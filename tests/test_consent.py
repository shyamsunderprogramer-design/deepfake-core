"""Tests for the consent ledger and the hard gate (plan §2 / M1)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from core import consent
from core.consent import ConsentError


def _enroll(photo_files, **overrides: Any) -> consent.ConsentRecord:
    args: dict[str, Any] = dict(
        identity_id="alice",
        display_name="Alice Example",
        photo_paths=photo_files,
        projects=["demo"],
        uses=["realtime", "image"],
        revocation_contact="alice@example.com",
    )
    args.update(overrides)
    return consent.enroll(**args)


class TestEnrollAndLoad:
    def test_round_trip(self, consent_env, photo_files):
        rec = _enroll(photo_files, expires_days=30)
        assert rec.path.exists()
        loaded = consent.load("alice")
        assert loaded.identity_id == "alice"
        assert loaded.display_name == "Alice Example"
        assert loaded.identity_hash == rec.identity_hash
        assert loaded.scope_projects == ["demo"]
        assert loaded.scope_uses == ["realtime", "image"]
        assert loaded.revoked is False
        assert loaded.expires_at is not None

    def test_identity_hash_is_content_based_not_filename(self, tmp_path):
        a = tmp_path / "a.jpg"
        b = tmp_path / "b.jpg"
        a.write_bytes(b"same-bytes")
        b.write_bytes(b"same-bytes")
        h1 = consent.identity_hash_from_photos([a])
        h2 = consent.identity_hash_from_photos([b])
        assert h1 == h2

    def test_requires_photos(self):
        with pytest.raises(ConsentError):
            _enroll([], photo_paths=[])

    def test_requires_revocation_contact(self, consent_env, photo_files):
        with pytest.raises(ConsentError):
            _enroll(photo_files, revocation_contact="")

    @pytest.mark.parametrize("bad_id", ["", "-x", "a b", "a/b", "x" * 65])
    def test_rejects_invalid_identity_id(self, consent_env, photo_files, bad_id):
        with pytest.raises(ConsentError):
            _enroll(photo_files, identity_id=bad_id)

    def test_no_expiry_when_omitted(self, consent_env, photo_files):
        rec = _enroll(photo_files)
        assert rec.expires_at is None
        assert rec.expired() is False


class TestHardGate:
    def test_valid_consent_passes(self, consent_env, photo_files):
        _enroll(photo_files)
        rec = consent.check("alice", project="demo", use="realtime")
        assert rec.identity_id == "alice"

    def test_missing_record_refuses(self, consent_env):
        with pytest.raises(ConsentError, match="no consent record"):
            consent.check("mallory", project="demo", use="realtime")

    def test_revoked_refuses(self, consent_env, photo_files):
        _enroll(photo_files)
        consent.revoke("alice")
        with pytest.raises(ConsentError, match="revoked"):
            consent.check("alice", project="demo", use="realtime")

    def test_expired_refuses(self, consent_env, photo_files):
        _enroll(photo_files, expires_days=-1)  # expiry in the past
        assert consent.load("alice").expired() is True
        with pytest.raises(ConsentError, match="expired"):
            consent.check("alice", project="demo", use="realtime")

    def test_project_outside_scope_refuses(self, consent_env, photo_files):
        _enroll(photo_files)
        with pytest.raises(ConsentError, match="project"):
            consent.check("alice", project="other", use="realtime")

    def test_use_outside_scope_refuses(self, consent_env, photo_files):
        _enroll(photo_files)
        with pytest.raises(ConsentError, match="use"):
            consent.check("alice", project="demo", use="video")

    def test_empty_scope_is_wildcard(self, consent_env, photo_files):
        _enroll(photo_files, projects=[], uses=[])
        rec = consent.check("alice", project="anything", use="video")
        assert rec.identity_id == "alice"

    def test_expiry_boundary(self, consent_env, photo_files):
        rec = _enroll(photo_files, expires_days=10)
        future = datetime.now(timezone.utc) + timedelta(days=9)
        assert rec.expired(future) is False
        future = datetime.now(timezone.utc) + timedelta(days=11)
        assert rec.expired(future) is True


class TestRevoke:
    def test_revoke_deletes_trained_models(self, consent_env, photo_files):
        _enroll(photo_files)
        model_dir = consent_env["trained"] / "alice"
        model_dir.mkdir(parents=True)
        (model_dir / "model.onnx").write_bytes(b"weights")
        result = consent.revoke("alice")
        assert result["revoked"] is True
        assert result["deleted_model_dir"] == str(model_dir)
        assert not model_dir.exists()
        assert consent.load("alice").revoked is True

    def test_revoke_without_models(self, consent_env, photo_files):
        _enroll(photo_files)
        result = consent.revoke("alice")
        assert result["deleted_model_dir"] is None

    def test_revoke_unknown_identity_refuses(self, consent_env):
        with pytest.raises(ConsentError):
            consent.revoke("ghost")


class TestListRecords:
    def test_skips_underscore_templates(self, consent_env, photo_files):
        _enroll(photo_files)
        (consent_env["consent"] / "_template.json").write_text("{}")
        ids = [r.identity_id for r in consent.list_records()]
        assert ids == ["alice"]

    def test_corrupt_record_raises(self, consent_env, photo_files):
        _enroll(photo_files)
        (consent_env["consent"] / "alice.json").write_text("{not json")
        with pytest.raises(ConsentError, match="corrupt"):
            consent.list_records()

    def test_empty_dir(self, consent_env):
        assert consent.list_records() == []
