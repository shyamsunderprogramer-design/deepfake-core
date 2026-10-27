"""ONNX model loading with deliberate hardware-unit routing (plan §7.1).

Swap/restore models get the ANE (MLComputeUnits=ALL); the detector is routed
to CPU+GPU so it runs concurrently with the ANE-bound swap model on a
different hardware unit. Sessions are built lazily and cached per model.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import onnxruntime as ort
import yaml

from .. import config

# CoreML EP options per role — the det-off-ANE / swap-on-ANE split.
_COREML_BASE = {
    "ModelFormat": "MLProgram",
    "AllowLowPrecisionAccumulationOnGPU": "1",
}
_COREML_SWAP = {
    **_COREML_BASE,
    "MLComputeUnits": "ALL",
    "SpecializationStrategy": "FastPrediction",
    "EnableOnSubgraphs": "1",
}
_COREML_DETECTOR = {**_COREML_BASE, "MLComputeUnits": "CPUAndGPU"}

ROLE_PROVIDERS: Dict[str, tuple] = {
    "detector": ((-1, _COREML_DETECTOR),),
    "identity_encoder": ((-1, _COREML_SWAP),),
    "swap": ((-1, _COREML_SWAP),),
    "restore": ((-1, _COREML_SWAP),),
}


class ModelMissing(FileNotFoundError):
    """A registry model has no local file (assets/models is gitignored)."""


class ModelHashMismatch(RuntimeError):
    """Local file's sha256 does not match the registry entry."""


class ModelEntry:
    """One registry row."""

    def __init__(self, raw: dict):
        self.name: str = raw["name"]
        self.role: str = raw["role"]
        self.file: str = raw["file"]
        self.input_size: tuple = tuple(raw.get("input_size", (0, 0)))
        self.sha256: Optional[str] = raw.get("sha256") or None

    @property
    def path(self) -> Path:
        return config.MODELS_DIR / self.file

    def present(self) -> bool:
        return self.path.is_file()

    def hash_ok(self) -> Optional[bool]:
        """True/False when a sha256 is pinned; None when unpinned."""
        if self.sha256 is None:
            return None
        if not self.present():
            return False
        return hashlib.sha256(self.path.read_bytes()).hexdigest() == self.sha256


def registry() -> List[ModelEntry]:
    path = config.MODEL_REGISTRY
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [ModelEntry(m) for m in data.get("models", [])]


def entry_by_role(role: str) -> ModelEntry:
    for e in registry():
        if e.role == role:
            return e
    raise ModelMissing(f"no registry entry for role '{role}'")


def providers_for_role(role: str) -> List:
    """ort provider list for a role; falls back to plain CoreML/CPU when the
    tuned options raise (e.g. model uses ops CoreML would reject)."""
    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        return ["CPUExecutionProvider"]
    entries = ROLE_PROVIDERS.get(role, ROLE_PROVIDERS["swap"])
    out = []
    for _, ep_opts in entries:
        out.append(("CoreMLExecutionProvider", dict(ep_opts)))
    out.append("CPUExecutionProvider")
    return out


_sessions: Dict[str, "ort.InferenceSession"] = {}


def get_session(role: str, *, force_reload: bool = False) -> "ort.InferenceSession":
    """Lazily build (and cache) a session for the registry model of `role`."""
    e = entry_by_role(role)
    if not e.present():
        raise ModelMissing(
            f"{e.name} ({e.file}) is not in {config.MODELS_DIR}. Download the "
            "weights per assets/models/registry.yaml — they are not committed."
        )
    if e.hash_ok() is False:
        raise ModelHashMismatch(f"{e.file} sha256 does not match registry.yaml")
    key = f"{role}:{e.path}"
    if force_reload or key not in _sessions:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _sessions[key] = ort.InferenceSession(
            str(e.path), sess_options=opts, providers=providers_for_role(role)
        )
    return _sessions[key]
