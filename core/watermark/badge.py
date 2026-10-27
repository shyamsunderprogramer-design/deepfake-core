"""Visible disclosure badge drawn onto output frames (plan §2).

Complements the invisible DCT watermark: a small, unobtrusive but legible
' SYNTHETIC MEDIA ' strip so every rendered frame/image is labelled on
sight, not just in metadata.
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_TEXT = "SYNTHETIC MEDIA"


def add_badge(
    bgr: np.ndarray, text: str = DEFAULT_TEXT, alpha: float = 0.65
) -> np.ndarray:
    """Draw a translucent bottom strip with centred disclosure text.

    Input/output are BGR uint8 (OpenCV convention). Returns a new array.
    """
    out = bgr.copy()
    h, w = out.shape[:2]
    bar = max(28, h // 22)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - bar), (w, h), (20, 20, 20), thickness=-1)
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = bar / 34.0
    thick = max(1, int(bar / 16))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = max(4, (w - tw) // 2)
    y = h - (bar - th) // 2
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out
