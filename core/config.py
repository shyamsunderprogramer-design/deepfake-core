"""Central paths and constants."""
from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of this package directory.
ROOT = Path(__file__).resolve().parents[1]

CONSENT_DIR = Path(os.environ.get("DEEPFAKE_CONSENT_DIR", ROOT / "consent"))
MODELS_DIR = Path(os.environ.get("DEEPFAKE_MODELS_DIR", ROOT / "assets" / "models"))
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
MODEL_REGISTRY = MODELS_DIR / "registry.yaml"
MODEL_LOCK = MODELS_DIR / "lock.yaml"
RUNS_DIR = ROOT / "runs"
EVAL_PACKS_DIR = ROOT / "eval" / "benchmark_packs"
EVAL_REPORTS_DIR = ROOT / "eval" / "reports"

USE_ORDER = ["realtime", "image", "video", "training"]

# §11.1 accuracy gates
GATES = {
    "offline": {"id_cosine_min": 0.60, "id_retrieval_min": 0.95},
    "realtime": {"id_cosine_min": 0.50, "id_retrieval_min": 0.90},
}

# §7.2 stage budgets at 1080p, milliseconds (core pipeline, restore optional)
STAGE_BUDGETS_MS = {
    "detect": 4.0,
    "align": 1.0,
    "swap": 12.0,
    "blend_color_encode": 3.0,
    "core_total": 16.0,
}
TARGET_FPS_1080P = 30.0
