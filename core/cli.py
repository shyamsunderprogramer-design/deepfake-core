"""Command line entry point for the deepfake toolchain.

Step 4 scope: `doctor` environment checks and the `consent` lifecycle
(enroll/list/show/revoke) wired to core/consent.py. Later milestones attach
models, swap, realtime, bench and eval subcommands onto the same parser.

    deepfake doctor
    deepfake consent enroll --id alice --name "Alice" --photos a.jpg b.jpg \
        --projects demo --uses realtime image --revocation-contact alice@x.io
    deepfake consent list
    deepfake consent show alice
    deepfake consent revoke alice
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __generator_id__, __version__
from . import config
from .consent import ConsentError
from . import consent as consent_mod


def _print_error(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# doctor


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK " if ok else "MISS"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    return ok


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"deepfake-core {__version__} ({__generator_id__})")
    all_ok = True

    all_ok &= _check(
        "python >= 3.10",
        sys.version_info >= (3, 10),
        sys.version.split()[0],
    )

    try:
        import numpy  # noqa: F401
        all_ok &= _check("numpy", True, numpy.__version__)
    except Exception:
        all_ok &= _check("numpy", False, "pip install numpy")
    try:
        import cv2  # noqa: F401
        all_ok &= _check("opencv (cv2)", True, cv2.__version__)
    except Exception:
        all_ok &= _check("opencv (cv2)", False, "pip install opencv-python")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        all_ok &= _check("onnxruntime", True, ort.__version__)
        all_ok &= _check(
            "CoreML execution provider (Apple Silicon acceleration)",
            "CoreMLExecutionProvider" in providers,
            ", ".join(providers),
        )
    except Exception:
        all_ok &= _check("onnxruntime", False, "pip install onnxruntime")
    try:
        import tqdm  # noqa: F401
        all_ok &= _check("tqdm progress bars", True, tqdm.__version__)
    except Exception:
        all_ok &= _check("tqdm progress bars", False, "optional; built-in fallback works")

    # model registry status — models are downloadable assets, never committed;
    # missing weights are reported but do not fail the consent-only commands.
    from .models import session as model_sessions

    entries = model_sessions.registry()
    if entries:
        print("\n  models (downloadable assets):")
    for entry in entries:
        if not entry.present():
            _check(f"model {entry.name}", False, f"missing {entry.file}")
        else:
            h = entry.hash_ok()
            detail = "hash ok" if h else ("NO sha256 pinned — supply-chain fail-open; pin it in registry.yaml" if h is None else "HASH MISMATCH")
            _check(f"model {entry.name}", h is not False, detail)

    try:
        config.CONSENT_DIR.mkdir(parents=True, exist_ok=True)
        all_ok &= _check("consent dir writable", True, str(config.CONSENT_DIR))
    except Exception as exc:
        all_ok &= _check("consent dir writable", False, str(exc))
    try:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        all_ok &= _check("models dir writable", True, str(config.MODELS_DIR))
    except Exception as exc:
        all_ok &= _check("models dir writable", False, str(exc))

    print("\ndoctor: " + ("all required checks passed" if all_ok else "issues found — see above"))
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# consent


def cmd_consent_enroll(args: argparse.Namespace) -> int:
    photos = [Path(p) for p in args.photos]
    missing = [str(p) for p in photos if not p.is_file()]
    if missing:
        return _print_error(f"photo file(s) not found: {', '.join(missing)}")
    try:
        rec = consent_mod.enroll(
            identity_id=args.id,
            display_name=args.name,
            photo_paths=photos,
            projects=args.projects,
            uses=args.uses,
            revocation_contact=args.revocation_contact,
            expires_days=args.expires_days,
        )
    except ConsentError as exc:
        return _print_error(str(exc))
    print(f"enrolled '{rec.identity_id}' ({rec.display_name})")
    print(f"  identity_hash: {rec.identity_hash}")
    print(f"  projects: {rec.scope_projects or ['*']}")
    print(f"  uses:     {rec.scope_uses or ['*']}")
    print(f"  expires:  {rec.expires_at or 'never'}")
    print(f"  record:   {rec.path}")
    return 0


def cmd_consent_list(args: argparse.Namespace) -> int:
    try:
        records = consent_mod.list_records()
    except ConsentError as exc:
        return _print_error(str(exc))
    if not records:
        print(f"no consent records in {config.CONSENT_DIR}")
        return 0
    for rec in records:
        flags = []
        if rec.revoked:
            flags.append("REVOKED")
        if rec.expired():
            flags.append("EXPIRED")
        status = f"  [{' '.join(flags)}]" if flags else ""
        print(f"{rec.identity_id}  ({rec.display_name}){status}")
    return 0


def cmd_consent_show(args: argparse.Namespace) -> int:
    try:
        rec = consent_mod.load(args.id)
    except ConsentError as exc:
        return _print_error(str(exc))
    print(json.dumps(rec.to_dict(), indent=2))
    return 0


def cmd_consent_revoke(args: argparse.Namespace) -> int:
    try:
        rec = consent_mod.load(args.id)
    except ConsentError as exc:
        return _print_error(str(exc))
    if not args.yes:
        print(
            f"This revokes consent for '{rec.identity_id}' ({rec.display_name}) and "
            f"permanently deletes assets/models/trained/{rec.identity_id}/."
        )
        answer = input("Type the identity_id to confirm: ").strip()
        if answer != rec.identity_id:
            print("aborted (identity_id did not match)")
            return 1
    result = consent_mod.revoke(args.id)
    print(f"revoked '{result['identity_id']}'")
    if result["deleted_model_dir"]:
        print(f"  deleted trained model dir: {result['deleted_model_dir']}")
    else:
        print("  no trained model dir existed")
    return 0


# ---------------------------------------------------------------------------
# swap / live / bench


def _build_pipeline(args: argparse.Namespace, use: str):
    """Consent-gated pipeline construction; returns SwapPipeline or exits."""
    import cv2

    from .models import session as model_sessions
    from .pipeline.detect import Detector
    from .pipeline.pipeline import SwapPipeline
    from .pipeline.swap import IdentityEncoder, Swapper

    # Consent is checked before any model is touched (and again inside
    # SwapPipeline construction — defense in depth).
    try:
        consent_mod.check(args.identity, project=args.project, use=use)
    except ConsentError as exc:
        print(f"consent refused: {exc}", file=sys.stderr)
        raise SystemExit(3)

    try:
        unpinned = getattr(args, "allow_unpinned_models", False)
        det_session = model_sessions.get_session("detector", allow_unpinned=unpinned)
        enc_session = model_sessions.get_session("identity_encoder", allow_unpinned=unpinned)
        swp_session = model_sessions.get_session("swap", allow_unpinned=unpinned)
    except (model_sessions.ModelMissing, model_sessions.ModelUnpinned) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

    pipeline = SwapPipeline(
        identity_id=args.identity,
        project=args.project,
        use=use,
        detector=Detector(det_session),
        encoder=IdentityEncoder(enc_session),
        swapper=Swapper(swp_session),
    )

    # Source encoding: detect the identity's face in each --source photo and
    # average the embeddings (plan §6.2).
    frames, kps_list = [], []
    for path in args.source:
        img = cv2.imread(str(path))
        if img is None:
            print(f"error: cannot read source photo {path}", file=sys.stderr)
            raise SystemExit(2)
        face = pipeline.detector.detect_one(img)
        if face is None or face.score < 0.5:
            print(f"error: no clear face found in {path}", file=sys.stderr)
            raise SystemExit(2)
        frames.append(img)
        kps_list.append(face.kps)
    pipeline.set_source(frames, kps_list)
    return pipeline


def cmd_swap_image(args: argparse.Namespace) -> int:
    import cv2

    pipeline = _build_pipeline(args, use="image")
    target = cv2.imread(str(args.target))
    if target is None:
        return _print_error(f"cannot read target image {args.target}")
    out = pipeline.process_frame(target)
    result = pipeline.finalize(out, Path(args.out))
    print(f"swapped '{args.identity}' into {args.target}")
    print(f"  output:     {result['file']}")
    print(f"  provenance: {result['provenance']['mode']}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from .realtime.loop import run_live

    pipeline = _build_pipeline(args, use="realtime")
    return run_live(pipeline, camera_index=args.camera)


def cmd_bench(args: argparse.Namespace) -> int:
    from .pipeline.bench import run_bench

    pipeline = _build_pipeline(args, use="image")
    report = run_bench(pipeline, frames=args.frames)
    print(f"bench: {report['effective_fps']} FPS effective "
          f"(detect {report['stages']['detect']['avg_ms']} ms, "
          f"swap+blend {report['stages']['swap_blend']['avg_ms']} ms)")
    print(f"  budget ≤{report['budgets_ms']['core_total']} ms: "
          + ("OK" if report["budget_ok"] else "OVER BUDGET"))
    print(f"  report: {report['report_file']}")
    return 0 if report["budget_ok"] else 1


# ---------------------------------------------------------------------------
# parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepfake",
        description="Consent-gated face-swap toolchain (see docs/DEEPFAKE_BUILD_PLAN.md).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="check the local environment")
    p_doctor.set_defaults(func=cmd_doctor)

    p_consent = sub.add_parser("consent", help="consent lifecycle (enroll/list/show/revoke)")
    csub = p_consent.add_subparsers(dest="consent_command", required=True)

    p_enroll = csub.add_parser("enroll", help="enroll an identity with consent")
    p_enroll.add_argument("--id", required=True, help="identity id ([A-Za-z0-9][A-Za-z0-9_.-]{0,63})")
    p_enroll.add_argument("--name", required=True, help="display name")
    p_enroll.add_argument(
        "--photos", required=True, nargs="+",
        help="reference photo(s); hashed together into identity_hash",
    )
    p_enroll.add_argument(
        "--projects", default=[], nargs="*",
        help="allowed project scopes (empty = any project)",
    )
    p_enroll.add_argument(
        "--uses", default=[], nargs="*",
        choices=config.USE_ORDER,
        help="allowed uses (empty = any use)",
    )
    p_enroll.add_argument(
        "--revocation-contact", required=True,
        help="contact the person can use to revoke consent",
    )
    p_enroll.add_argument(
        "--expires-days", type=int, default=None,
        help="consent expires after this many days (omit = never)",
    )
    p_enroll.set_defaults(func=cmd_consent_enroll)

    p_list = csub.add_parser("list", help="list all consent records")
    p_list.set_defaults(func=cmd_consent_list)

    p_show = csub.add_parser("show", help="show one consent record as JSON")
    p_show.add_argument("id")
    p_show.set_defaults(func=cmd_consent_show)

    p_revoke = csub.add_parser("revoke", help="revoke consent and delete trained assets")
    p_revoke.add_argument("id")
    p_revoke.add_argument("--yes", action="store_true", help="skip interactive confirmation")
    p_revoke.set_defaults(func=cmd_consent_revoke)

    def _add_source_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--identity", required=True, help="consented source identity id")
        p.add_argument("--source", required=True, nargs="+", type=Path,
                       help="photo(s) of the consented identity (embeddings are averaged)")
        p.add_argument("--project", required=True, help="project scope (must be in the consent record)")
        p.add_argument("--allow-unpinned-models", dest="allow_unpinned_models",
                       action="store_true",
                       help="dev only: load model files whose registry sha256 is not pinned")

    p_swap = sub.add_parser("swap", help="swap a consented identity into a target image")
    _add_source_args(p_swap)
    p_swap.add_argument("--target", required=True, type=Path, help="target image")
    p_swap.add_argument("--out", required=True, type=Path, help="output image path")
    p_swap.set_defaults(func=cmd_swap_image)

    p_live = sub.add_parser("live", help="real-time camera swap (press q to quit)")
    _add_source_args(p_live)
    p_live.add_argument("--camera", type=int, default=0)
    p_live.set_defaults(func=cmd_live)

    p_bench = sub.add_parser("bench", help="stage-level benchmark; report saved to runs/")
    _add_source_args(p_bench)
    p_bench.add_argument("--frames", type=int, default=200)
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
