"""Tests for C2PA provenance signing (sidecar fallback mode)."""
from __future__ import annotations

import hashlib
import json

from core.watermark import c2pa


class TestSidecarMode:
    """Without c2pa-python + credentials, sign_output must write a sidecar."""

    def test_sidecar_written_with_claims(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEEPFAKE_C2PA_CERT", raising=False)
        monkeypatch.delenv("DEEPFAKE_C2PA_KEY", raising=False)
        out = tmp_path / "frame.png"
        out.write_bytes(b"fake-png-bytes")

        result = c2pa.sign_output(
            out,
            generator_id="deepfake-core/0.1.0",
            consent_id="alice",
            model_version="v1",
            use="image",
        )
        assert result["mode"] in ("sidecar", "embedded")
        if result["mode"] == "embedded":
            return  # environment has a real signer; sidecar asserts don't apply

        sidecar = tmp_path / "frame.png.c2pa.json"
        assert sidecar.exists()
        claims = json.loads(sidecar.read_text())
        assert claims["synthetic_media"] is True
        assert claims["consent_id"] == "alice"
        assert claims["model_version"] == "v1"
        assert claims["use"] == "image"
        assert claims["claim_generator"] == "deepfake-core/0.1.0"
        assert claims["standard"] == "C2PA"
        assert claims["file_sha256"] == hashlib.sha256(b"fake-png-bytes").hexdigest()

    def test_no_export_without_provenance(self, tmp_path, monkeypatch):
        """Every call must produce either an embedded manifest or a sidecar."""
        monkeypatch.delenv("DEEPFAKE_C2PA_CERT", raising=False)
        monkeypatch.delenv("DEEPFAKE_C2PA_KEY", raising=False)
        out = tmp_path / "x.bin"
        out.write_bytes(b"data")
        result = c2pa.sign_output(
            out, generator_id="g", consent_id="c", model_version="m", use="video"
        )
        assert result["mode"] in ("sidecar", "embedded")
