"""Tests for the visible 'SYNTHETIC MEDIA' disclosure badge."""
from __future__ import annotations

import numpy as np

from core.watermark.badge import DEFAULT_TEXT, add_badge


def _frame(h=480, w=640):
    return np.full((h, w, 3), 128, dtype=np.uint8)


class TestBadge:
    def test_shape_and_dtype_preserved(self):
        frame = _frame()
        out = add_badge(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_bottom_strip_is_drawn(self):
        frame = _frame()
        out = add_badge(frame)
        bar = max(28, 480 // 22)
        assert not np.array_equal(out[-bar:, :], frame[-bar:, :])

    def test_top_region_untouched(self):
        frame = _frame()
        out = add_badge(frame)
        bar = max(28, 480 // 22)
        assert np.array_equal(out[: 480 - bar], frame[: 480 - bar])

    def test_input_not_mutated(self):
        frame = _frame()
        snapshot = frame.copy()
        add_badge(frame)
        assert np.array_equal(frame, snapshot)

    def test_custom_text(self):
        out = add_badge(_frame(), text="AI GENERATED")
        assert out.shape == _frame().shape

    def test_small_frame_no_crash(self):
        out = add_badge(np.zeros((40, 60, 3), dtype=np.uint8))
        assert out.shape == (40, 60, 3)
        assert DEFAULT_TEXT  # default disclosed
