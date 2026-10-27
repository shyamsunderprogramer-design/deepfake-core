"""Consent ledger and hard gate.

No swap, source-embedding, or training job may run without a valid consent
record for the identity involved (plan §2 / M1 exit criterion). Records live
in ``consent/<identity_id>.json``; see ``consent/_template.json``.

A record is valid when it exists, matches the requested project/use scope,
is unexpired, and is not revoked. Revocation marks the record AND deletes
any trained model for the identity at ``assets/models/trained/<identity_id>/``.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Union

from . import config

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


class ConsentError(PermissionError):
    """Raised whenever an action lacks a valid consent record."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _validate_identity_id(identity_id: str) -> None:
    if not _ID_RE.match(identity_id or ""):
        raise ConsentError(
            f"invalid identity_id {identity_id!r}: 1-64 chars of [A-Za-z0-9_.-], "
            "starting with a letter or digit"
        )


def identity_hash_from_photos(photo_paths: List[Union[str, Path]]) -> str:
    """sha256 over the sorted per-photo sha256 hex digests."""
    paths = [Path(p) for p in photo_paths]
    if not paths:
        raise ConsentError("enrollment requires at least one reference photo")
    digests = sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths)
    return hashlib.sha256("".join(digests).encode("ascii")).hexdigest()


@dataclass
class ConsentRecord:
    identity_id: str
    display_name: str
    identity_hash: str
    scope_projects: List[str] = field(default_factory=list)
    scope_uses: List[str] = field(default_factory=list)
    issued_at: str = ""
    expires_at: Optional[str] = None
    revoked: bool = False
    revocation_contact: str = ""
    signature: Optional[str] = None  # optional ed25519 signature of the record

    def to_dict(self) -> dict:
        return {
            "identity_id": self.identity_id,
            "display_name": self.display_name,
            "identity_hash": self.identity_hash,
            "scope": {
                "projects": list(self.scope_projects),
                "uses": list(self.scope_uses),
            },
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "revoked": self.revoked,
            "revocation_contact": self.revocation_contact,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConsentRecord":
        scope = data.get("scope", {}) or {}
        return cls(
            identity_id=data["identity_id"],
            display_name=data.get("display_name", ""),
            identity_hash=data.get("identity_hash", ""),
            scope_projects=list(scope.get("projects", [])),
            scope_uses=list(scope.get("uses", [])),
            issued_at=data.get("issued_at", ""),
            expires_at=data.get("expires_at"),
            revoked=bool(data.get("revoked", False)),
            revocation_contact=data.get("revocation_contact", ""),
            signature=data.get("signature"),
        )

    @property
    def path(self) -> Path:
        return config.CONSENT_DIR / f"{self.identity_id}.json"

    def expired(self, now: Optional[datetime] = None) -> bool:
        """True when expires_at is set and is not in the future."""
        if not self.expires_at:
            return False
        return _parse_ts(self.expires_at) <= (now or _utc_now())

    def save(self) -> Path:
        config.CONSENT_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return self.path


def enroll(
    identity_id: str,
    display_name: str,
    photo_paths: List[Union[str, Path]],
    projects: List[str],
    uses: List[str],
    revocation_contact: str,
    expires_days: Optional[int] = None,
) -> ConsentRecord:
    """Create and persist a consent record for an identity."""
    _validate_identity_id(identity_id)
    if not revocation_contact:
        raise ConsentError("a revocation contact is required (plan §2)")
    now = _utc_now()
    rec = ConsentRecord(
        identity_id=identity_id,
        display_name=display_name,
        identity_hash=identity_hash_from_photos(photo_paths),
        scope_projects=list(projects),
        scope_uses=list(uses),
        issued_at=_iso(now),
        expires_at=_iso(now + timedelta(days=expires_days)) if expires_days else None,
        revoked=False,
        revocation_contact=revocation_contact,
        signature=None,
    )
    rec.save()
    return rec


def load(identity_id: str) -> ConsentRecord:
    path = config.CONSENT_DIR / f"{identity_id}.json"
    if not path.exists():
        raise ConsentError(f"no consent record for identity '{identity_id}'")
    return ConsentRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_records() -> List[ConsentRecord]:
    if not config.CONSENT_DIR.exists():
        return []
    out: List[ConsentRecord] = []
    for path in sorted(config.CONSENT_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            out.append(ConsentRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ConsentError(f"corrupt consent record {path.name}: {exc}") from exc
    return out


def check(
    identity_id: str,
    project: str,
    use: str,
    now: Optional[datetime] = None,
) -> ConsentRecord:
    """Hard gate. Returns the record only if consent is valid; otherwise raises.

    `use` should be one of ``config.USE_ORDER`` entries (realtime, image,
    video, training). Empty scope lists mean "not scoped" (any project/use);
    the record's existence, validity window and revocation status still gate.
    """
    now = now or _utc_now()
    rec = load(identity_id)
    if rec.revoked:
        raise ConsentError(
            f"consent for '{identity_id}' was revoked; trained models were deleted"
        )
    if rec.expired(now):
        raise ConsentError(
            f"consent for '{identity_id}' expired at {rec.expires_at}"
        )
    if rec.scope_projects and project not in rec.scope_projects:
        raise ConsentError(
            f"project '{project}' is outside the consented scope "
            f"{rec.scope_projects} for '{identity_id}'"
        )
    if rec.scope_uses and use not in rec.scope_uses:
        raise ConsentError(
            f"use '{use}' is outside the consented scope "
            f"{rec.scope_uses} for '{identity_id}'"
        )
    return rec


def revoke(identity_id: str) -> dict:
    """Revoke consent and delete any trained model for the identity."""
    rec = load(identity_id)
    rec.revoked = True
    rec.save()
    deleted = None
    model_dir = config.TRAINED_MODELS_DIR / identity_id
    if model_dir.exists():
        shutil.rmtree(model_dir)
        deleted = str(model_dir)
    return {
        "identity_id": identity_id,
        "revoked": True,
        "deleted_model_dir": deleted,
    }
