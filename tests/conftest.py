"""Shared fixtures: isolate consent records and trained-model dirs per test."""
from __future__ import annotations

from pathlib import Path

import pytest

from core import config


@pytest.fixture()
def consent_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point consent + trained-model storage at a temp dir.

    config constants are read at call time, so monkeypatching them fully
    isolates the test from the real repo consent/ and assets/.
    """
    consent_dir = tmp_path / "consent"
    models_dir = tmp_path / "assets" / "models"
    trained_dir = models_dir / "trained"
    monkeypatch.setattr(config, "CONSENT_DIR", consent_dir)
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config, "TRAINED_MODELS_DIR", trained_dir)
    consent_dir.mkdir(parents=True)
    return {"consent": consent_dir, "models": models_dir, "trained": trained_dir}


@pytest.fixture()
def photo_files(tmp_path: Path):
    """Two dummy reference 'photos' — hashing only needs bytes, not images."""
    paths = []
    for i in (1, 2):
        p = tmp_path / f"photo-{i}.jpg"
        p.write_bytes(b"fake-jpeg-bytes-%d" % i)
        paths.append(p)
    return paths
