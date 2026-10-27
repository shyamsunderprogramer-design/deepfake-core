# deepfake-core

Consent-gated face-swap toolchain for Apple Silicon. See the authoritative
build specification in [`docs/DEEPFAKE_BUILD_PLAN.md`](docs/DEEPFAKE_BUILD_PLAN.md).

Every output path enforces three things:

- **Consent** — an identity cannot be swapped unless a consent record exists
  in `consent/<identity_id>.json`, covers the requested project and use, and
  is neither expired nor revoked (`core/consent.py`). Revocation also deletes
  that identity's trained models.
- **Provenance** — every output is signed with C2PA claims
  (`core/watermark/c2pa.py`), carrying a `synthetic_media: true` claim, the
  consent id, and the model version. When the optional embedded signer is not
  configured, a `<file>.c2pa.json` sidecar is written instead.
- **Disclosure** — outputs carry an invisible frequency-domain watermark
  (`core/watermark/watermark.py`) and a visible "SYNTHETIC MEDIA" badge
  (`core/watermark/badge.py`).

## Quick start

```bash
pip install -e ".[dev]"
deepfake doctor
```

Enroll an identity (photos are hashed locally; the hashes identify the
person, not the files):

```bash
deepfake consent enroll \
    --id alice --name "Alice Example" \
    --photos ~/photos/alice-1.jpg ~/photos/alice-2.jpg \
    --projects demo --uses realtime image video \
    --revocation-contact alice@example.com
```

Manage consent:

```bash
deepfake consent list          # shows REVOKED / EXPIRED flags
deepfake consent show alice    # full record as JSON
deepfake consent revoke alice  # asks for confirmation; also deletes her trained model
```

Swap a consented identity into an image (requires model weights — see
`assets/models/registry.yaml`; models are downloadable assets, never committed):

```bash
deepfake swap --identity alice --source ~/photos/alice-1.jpg \
    --target group-photo.png --out runs/alice-swap.png --project demo
deepfake live --identity alice --source ~/photos/alice-1.jpg --project demo  # camera
deepfake bench --identity alice --source ~/photos/alice-1.jpg --project demo
```

Every `swap`/`live`/`bench` run is refusal-first on consent: the CLI checks
the ledger before loading any model, and `SwapPipeline` re-checks at
construction (defense in depth). A swap invoked without valid consent exits
with code 3 and touches nothing.

## Layout

```
core/
  cli.py          # deepfake CLI: doctor, consent lifecycle, swap, live, bench
  consent.py      # consent ledger: enroll / check / revoke
  config.py       # paths, quality gates, stage budgets
  progress.py     # tqdm with a stdlib fallback for live progress bars
  models/         # ONNX session wrapper, CoreML provider routing, registry
  pipeline/       # detect, align, swap, blend (+ color match), orchestrator, bench
  realtime/       # threaded camera loop, drop-oldest queues, face tracker
  watermark/      # DCT watermark, C2PA signing, visible badge
assets/models/    # registry.yaml + downloadable weights (gitignored)
consent/          # consent records (gitignored) + _template.json
docs/             # DEEPFAKE_BUILD_PLAN.md (authoritative), compliance docs
tests/            # pytest suite (61 tests; runs without model weights)
```

## License

MIT (see `LICENSE`). Model weights referenced by `assets/models/registry.yaml`
are third-party assets with their own licenses (several non-commercial) and
are not covered. DeepFaceLab / DeepFaceLive / Deep-Live-Cam were studied as
references only — all are GPL/AGPL, so nothing is vendored from them.

## Responsible use

This tool only produces media of people who have explicitly consented, it
marks every output as synthetic (visible + invisible + C2PA), and it refuses
to run otherwise. It contains no detection-evasion or impersonation
functionality by design. Legal compliance notes live in
[`docs/compliance.md`](docs/compliance.md) (planned, M7).
