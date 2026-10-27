"""Unit tests for M2 pipeline stages — all run with stub sessions (no weights)."""
from __future__ import annotations

import numpy as np
import pytest

from core.pipeline import align, blend
from core.pipeline.detect import Decoder, Detector, letterbox


def _synthetic_scrfd_outputs():
    """One confident anchor near frame centre at stride 8 (raw head outputs:
    logit scores, distances in stride units — the decoder scales)."""
    outs = []
    for stride, side in ((8, 80), (16, 40), (32, 20)):
        n = side * side * 2
        score = np.full((n, 1), -5.0, dtype=np.float32)
        bbox = np.zeros((n, 4), dtype=np.float32)
        kps = np.zeros((n, 10), dtype=np.float32)
        if stride == 8:
            # grid cell (40, 40) -> centre (320, 320); num_anchors=2 means the
            # cell occupies two consecutive output rows. Keypoints are face-
            # like (five distinct offsets) so similarity alignment can fit.
            idx = 2 * (40 * side + 40)
            score[idx] = score[idx + 1] = 5.0
            bbox[idx] = bbox[idx + 1] = (20.0, 20.0, 20.0, 20.0)
            five = np.array([[-2, -1], [2, -1], [0, 0], [-1.5, 2], [1.5, 2]], dtype=np.float32)
            kps[idx] = kps[idx + 1] = five.reshape(-1)
        outs.extend([score, bbox, kps])
    return outs


class _FakeInput:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class TestAlign:
    def test_identity_when_kps_match_template(self):
        M = align.estimate_transform(align.TEMPLATE_112, 112)
        assert np.allclose(M, np.eye(2, 3), atol=1e-4)

    def test_rejects_wrong_landmark_count(self):
        with pytest.raises(ValueError):
            align.estimate_transform(np.zeros((4, 2)), 128)

    def test_crop_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop, M = align.align_crop(frame, align.TEMPLATE_112 * 3, 128)
        assert crop.shape == (128, 128, 3)
        assert M.shape == (2, 3)


class TestDetect:
    def test_decoder_finds_positive_anchor(self):
        boxes, kpss, scores = Decoder().decode(_synthetic_scrfd_outputs())
        assert scores.shape[0] >= 1
        best = boxes[0]
        assert abs((best[0] + best[2]) / 2 - 320.0) < 16  # within 2 strides
        assert kpss.shape == boxes.shape[0:1] + (5, 2)

    def test_decoder_empty_when_all_negative(self):
        outs = _synthetic_scrfd_outputs()
        outs = [-np.abs(o) if o.shape[-1] == 1 else o for o in outs]
        _, _, scores = Decoder().decode(outs)
        assert scores.shape[0] == 0

    def test_letterbox_preserves_aspect(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas, scale, dx, dy = letterbox(frame, 640)
        assert canvas.shape == (640, 640, 3)
        assert scale == pytest.approx(640 / 1920)
        assert dx == 0 and dy > 0  # wide frame pads vertically

    def test_detector_end_to_end_with_stub(self):
        class Stub:
            def get_inputs(self):
                return [_FakeInput("input.1", [1, 3, 640, 640])]

            def run(self, _outputs, feed):
                return _synthetic_scrfd_outputs()

        det = Detector(Stub())
        faces = det.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
        assert len(faces) == 1
        f = faces[0]
        # un-letterboxed: 640-coords map back to 1920x1080
        cx = (f.bbox[0] + f.bbox[2]) / 2
        assert cx == pytest.approx(960.0, abs=30.0)


class TestBlend:
    def test_soft_alpha_center_full_edges_zero(self):
        a = blend.soft_alpha(128)
        assert a.shape == (128, 128)
        assert a[64, 64] > 0.99
        assert a[0, 0] == 0.0

    def test_paste_back_translation(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = np.full((64, 64, 3), 200, dtype=np.uint8)
        M = np.array([[1, 0, 100], [0, 1, 80]], dtype=np.float32)  # translate only
        out = blend.paste_back(frame, crop, M)
        assert out[112, 132].max() > 100  # ROI centre took the crop
        assert not out[:80].any() and not out[:, :100].any()  # nothing outside

    def test_paste_back_clamps_offscreen(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = np.full((64, 64, 3), 255, dtype=np.uint8)
        M = np.array([[1, 0, 90], [0, 1, 90]], dtype=np.float32)  # mostly offscreen
        out = blend.paste_back(frame, crop, M)
        assert out.shape == frame.shape

    def test_reinhard_preserves_shape_dtype(self):
        rng = np.random.default_rng(7)
        face = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        ref = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        out = blend.reinhard_transfer(face, ref)
        assert out.shape == face.shape and out.dtype == np.uint8
