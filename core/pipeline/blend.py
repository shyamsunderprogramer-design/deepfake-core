"""Blending & color matching (plan §4 stages 5 & 7).

Tight-crop paste-back: invert the alignment transform and warp only the
face's bounding box back into the frame, with a cached feathered alpha mask
— the whole blend runs on a small ROI instead of the full frame.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

_alpha_cache: dict[Tuple[int, int], np.ndarray] = {}


def soft_alpha(size: int) -> np.ndarray:
    """Cached elliptical feathered alpha mask for a crop of `size` px."""
    key = (size, size)
    if key not in _alpha_cache:
        mask = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        axes = (int(size * 0.44), int(size * 0.44))
        cv2.ellipse(mask, (center, center), axes, 0, 0, 360, 255, -1)
        alpha_cache_blur = max(3, size // 8) | 1  # odd kernel
        _alpha_cache[key] = cv2.GaussianBlur(
            mask, (alpha_cache_blur, alpha_cache_blur), 0
        ).astype(np.float32) / 255.0
    return _alpha_cache[key]


def _tight_bbox(M_inv: np.ndarray, size: int, frame_shape) -> tuple:
    """Bounding box of the inverse-warped crop, clamped to the frame."""
    h, w = frame_shape[:2]
    corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    pts = cv2.transform(corners[None, ...], M_inv)[0]
    x0 = max(0, int(np.floor(pts[:, 0].min())))
    y0 = max(0, int(np.floor(pts[:, 1].min())))
    x1 = min(w, int(np.ceil(pts[:, 0].max())))
    y1 = min(h, int(np.ceil(pts[:, 1].max())))
    return x0, y0, x1, y1


def paste_back(
    frame: np.ndarray, crop: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """Inverse-warp `crop` into `frame` through M on a tight ROI only."""
    out = frame.copy()
    size = crop.shape[0]
    M_inv = cv2.invertAffineTransform(M)
    # crop corners forward through M (crop->frame) give the ROI on the frame
    x0, y0, x1, y1 = _tight_bbox(M, size, frame.shape)
    if x1 <= x0 or y1 <= y0:
        return out
    roi_shift = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float32)
    # warp sampling: roi pixel p -> frame point p+offset -> crop coord M^-1(·)
    M_roi = np.vstack([M_inv, [0, 0, 1]]) @ roi_shift
    warped = cv2.warpAffine(crop, M_roi[:2], (x1 - x0, y1 - y0))
    alpha = cv2.warpAffine(
        soft_alpha(size), M_roi[:2], (x1 - x0, y1 - y0)
    )[..., None]
    roi = out[y0:y1, x0:x1].astype(np.float32)
    blended = warped.astype(np.float32) * alpha + roi * (1.0 - alpha)
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def reinhard_transfer(face: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Plan §10 ct_mode `rct`: Reinhard per-channel mean/std transfer in LAB.

    `face` is the swapped crop; `reference` is the original aligned crop it
    replaces. Stats are computed on the face region only.
    """
    f = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    mf, sf = cv2.meanStdDev(f)
    mr, sr = cv2.meanStdDev(r)
    sf = np.maximum(sf, 1.0)
    out = (f - mf.reshape(1, 1, 3)) / sf.reshape(1, 1, 3)
    out = out * sr.reshape(1, 1, 3) + mr.reshape(1, 1, 3)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
