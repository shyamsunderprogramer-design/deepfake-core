"""Face alignment: similarity transform onto a canonical crop (plan §4 stage 2).

5-point template alignment in the insightface norm_crop family, implemented
directly on cv2 (umeyama) so we carry no insightface/AGPL dependency.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

# Standard 5-point canonical templates (x, y) per crop size.
TEMPLATE_112 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)
_TEMPLATES: Dict[int, np.ndarray] = {
    112: TEMPLATE_112,
    128: TEMPLATE_112 / 112.0 * 128.0,
    512: np.array(
        [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
         [201.26117, 371.41043], [313.08905, 371.15118]],
        dtype=np.float32,
    ),
}


def template_for(size: int) -> np.ndarray:
    if size not in _TEMPLATES:
        _TEMPLATES[size] = TEMPLATE_112 / 112.0 * float(size)
    return _TEMPLATES[size]


def estimate_transform(kps: np.ndarray, size: int) -> np.ndarray:
    """Similarity matrix M (2x3) mapping source kps onto the canonical crop."""
    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape != (5, 2):
        raise ValueError(f"expected 5 landmarks, got shape {kps.shape}")
    M, _ = cv2.estimateAffinePartial2D(
        kps[None, ...], template_for(size)[None, ...], method=cv2.LMEDS
    )
    if M is None:
        raise ValueError("alignment failed: degenerate landmarks")
    return M


def align_crop(bgr: np.ndarray, kps: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """(aligned crop, M). Keep M — paste-back needs its inverse."""
    M = estimate_transform(kps, size)
    return cv2.warpAffine(bgr, M, (size, size), borderValue=0.0), M
