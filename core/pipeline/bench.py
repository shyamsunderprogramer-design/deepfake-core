"""Stage-level benchmark harness (plan §7.2, §11.5 precursor).

Runs synthetic 1080p frames through detect -> swap -> blend, records
avg/min/max per stage plus effective FPS, and persists the report to
runs/bench-<timestamp>.json so regressions surface in CI.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Dict

import numpy as np

from .. import __version__, config
from .pipeline import SwapPipeline


def _synthetic_frame(w: int = 1920, h: int = 1080) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    frame = (128 + 60 * np.sin(x / 23) + 40 * np.cos(y / 31)).clip(0, 255)
    return np.stack([frame] * 3, axis=-1).astype(np.uint8)


def run_bench(pipeline: SwapPipeline, frames: int = 200) -> Dict:
    """Time each stage; requires a warmed pipeline (set_source already done)."""
    if not pipeline.ready:
        raise RuntimeError("bench needs a source identity: call set_source first")
    stats: Dict[str, list] = {"detect": [], "swap_blend": [], "total": []}
    for i in range(frames):
        frame = _synthetic_frame()
        t0 = time.perf_counter()
        faces = pipeline.detector.detect(frame)
        t1 = time.perf_counter()
        if faces:
            pipeline._swap_one(frame, faces[0])
        t2 = time.perf_counter()
        stats["detect"].append((t1 - t0) * 1e3)
        stats["swap_blend"].append((t2 - t1) * 1e3)
        stats["total"].append((t2 - t0) * 1e3)
        if (i + 1) % 50 == 0:
            print(f"  bench: {i + 1}/{frames}")

    def summarize(xs):
        a = np.asarray(xs)
        return {"avg_ms": round(float(a.mean()), 2),
                "min_ms": round(float(a.min()), 2),
                "max_ms": round(float(a.max()), 2)}

    report = {
        "version": __version__,
        "model_version": pipeline.model_version,
        "frames": frames,
        "resolution": [1920, 1080],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "stages": {k: summarize(v) for k, v in stats.items()},
        "budgets_ms": config.STAGE_BUDGETS_MS,
    }
    total_avg = report["stages"]["total"]["avg_ms"]
    report["effective_fps"] = round(1000.0 / total_avg, 1) if total_avg else None
    report["budget_ok"] = total_avg <= config.STAGE_BUDGETS_MS["core_total"]

    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = config.RUNS_DIR / f"bench-{ts}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_file"] = str(path)
    return report
