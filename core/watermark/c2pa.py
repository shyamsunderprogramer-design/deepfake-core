"""Content provenance: C2PA manifest embedding, with a sidecar-JSON fallback.

Embedded C2PA signing needs `c2pa-python` plus signing credentials, pointed
to by DEEPFAKE_C2PA_CERT / DEEPFAKE_C2PA_KEY. Without either, we write a
sidecar ``<output>.c2pa.json`` carrying the same claims plus the file's
sha256; verifiers and the release gate accept the sidecar.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import c2pa  # type: ignore

    _HAS_C2PA = True
except Exception:  # pragma: no cover
    c2pa: Any = None  # optional dependency missing; sidecar mode
    _HAS_C2PA = False


def has_c2pa() -> bool:
    return _HAS_C2PA


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign_output(
    path: Path,
    *,
    generator_id: str,
    consent_id: str,
    model_version: str,
    use: str,
) -> dict:
    """Attach provenance claims to an output file. Returns what was written."""
    path = Path(path)
    claims = {
        "standard": "C2PA",
        "claim_generator": generator_id,
        "synthetic_media": True,
        "consent_id": consent_id,
        "model_version": model_version,
        "use": use,
        "signed_at": datetime.now(timezone.utc).isoformat(),
        "file_sha256": _sha256_file(path),
    }

    cert = os.environ.get("DEEPFAKE_C2PA_CERT")
    key = os.environ.get("DEEPFAKE_C2PA_KEY")
    if cert and key:
        # Credentials were explicitly configured: failing over to a sidecar
        # would silently downgrade a promised embedded signature. Fail closed.
        if not _HAS_C2PA:
            raise RuntimeError(
                "DEEPFAKE_C2PA_CERT/DEEPFAKE_C2PA_KEY are set but c2pa-python "
                "is not installed; refusing to silently downgrade to a sidecar."
            )
        try:  # pragma: no cover - exercised only with credentials configured
            manifest = {
                "claim_generator": generator_id,
                "assertions": [
                    {"label": "deepfake.disclosure", "data": claims},
                    {
                        "label": "c2pa.actions",
                        "data": {"actions": [{"action": "c2pa.created"}]},
                    },
                ],
            }
            builder = c2pa.Builder(json.dumps(manifest))
            with open(cert, "rb") as cf, open(key, "rb") as kf:
                signer = c2pa.Signer.from_pem(cf.read(), kf.read(), "es256")
            dest = path.with_name(path.stem + ".signed" + path.suffix)
            builder.sign_file(signer, str(path), str(dest))
            return {"mode": "embedded", "file": str(dest), "claims": claims}
        except Exception as exc:  # fail closed with the cause attached
            raise RuntimeError(
                f"embedded C2PA signing failed for {path}: {exc}"
            ) from exc

    sidecar = path.with_name(path.name + ".c2pa.json")
    sidecar.write_text(json.dumps(claims, indent=2) + "\n", encoding="utf-8")
    return {"mode": "sidecar", "sidecar": str(sidecar), "claims": claims}
