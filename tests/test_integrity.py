"""Tests for the security-review hardening: fail-closed model integrity,
fail-closed C2PA signing, and keyable watermark payloads."""
from __future__ import annotations

import hashlib

import pytest

from core import config
from core.models import session as ms
from core.watermark import c2pa
from core.watermark.watermark import make_payload


@pytest.fixture()
def model_env(tmp_path, monkeypatch):
    """A temp models dir with a registry pointing at a dummy weight file."""
    models = tmp_path / "models"
    models.mkdir()
    weights = models / "thing.onnx"
    weights.write_bytes(b"not-a-real-onnx")
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "MODEL_REGISTRY", models / "registry.yaml")
    monkeypatch.setattr(ms, "_sessions", {})
    return models, weights


def _write_registry(models, sha256_line: str):
    if sha256_line != "null":
        sha256_line = f"'{sha256_line}'"  # keep all-digit hashes as strings
    (models / "registry.yaml").write_text(
        "models:\n"
        "  - name: thing\n"
        "    role: swap\n"
        f"    file: thing.onnx\n"
        f"    sha256: {sha256_line}\n"
    )


class TestModelIntegrity:
    def test_unpinned_refuses_by_default(self, model_env):
        models, _ = model_env
        _write_registry(models, "null")
        with pytest.raises(ms.ModelUnpinned, match="no sha256 pinned"):
            ms.get_session("swap")

    def test_wrong_pin_refuses(self, model_env):
        models, _ = model_env
        _write_registry(models, "0" * 64)
        with pytest.raises(ms.ModelHashMismatch):
            ms.get_session("swap")

    def test_unpinned_override_gets_past_pin_check(self, model_env):
        """--allow-unpinned-models skips the pin check (the garbage file then
        fails at ONNX parse, which is not the check under test)."""
        models, _ = model_env
        _write_registry(models, "null")
        with pytest.raises(Exception) as excinfo:
            ms.get_session("swap", allow_unpinned=True)
        assert not isinstance(excinfo.value, (ms.ModelUnpinned, ms.ModelHashMismatch))

    def test_matching_pin_gets_past_hash_check(self, model_env):
        models, weights = model_env
        _write_registry(models, hashlib.sha256(weights.read_bytes()).hexdigest())
        with pytest.raises(Exception) as excinfo:
            ms.get_session("swap")  # fails later at ONNX parse, not at hashing
        assert not isinstance(excinfo.value, (ms.ModelUnpinned, ms.ModelHashMismatch))


class TestC2paFailClosed:
    def test_configured_credentials_never_downgrade(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPFAKE_C2PA_CERT", str(tmp_path / "cert.pem"))
        monkeypatch.setenv("DEEPFAKE_C2PA_KEY", str(tmp_path / "key.pem"))
        out = tmp_path / "x.png"
        out.write_bytes(b"png")
        # c2pa-python is absent in this env -> must raise, never sidecar
        with pytest.raises(RuntimeError, match="downgrade|signing failed"):
            c2pa.sign_output(
                out, generator_id="g", consent_id="c", model_version="m", use="image"
            )
        assert not (tmp_path / "x.png.c2pa.json").exists()

    def test_sidecar_still_works_without_credentials(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEEPFAKE_C2PA_CERT", raising=False)
        monkeypatch.delenv("DEEPFAKE_C2PA_KEY", raising=False)
        out = tmp_path / "x.png"
        out.write_bytes(b"png")
        result = c2pa.sign_output(
            out, generator_id="g", consent_id="c", model_version="m", use="image"
        )
        assert result["mode"] == "sidecar"


class TestKeyedWatermarkPayload:
    def test_keyed_differs_from_unkeyed(self):
        a = make_payload("g", "alice", "v1", "t")
        b = make_payload("g", "alice", "v1", "t", key=b"install-secret")
        assert a != b and len(b) == 32

    def test_key_deterministic_and_sensitive(self):
        a = make_payload("g", "alice", "v1", "t", key=b"k1")
        b = make_payload("g", "alice", "v1", "t", key=b"k1")
        c = make_payload("g", "alice", "v1", "t", key=b"k2")
        assert a == b and a != c
