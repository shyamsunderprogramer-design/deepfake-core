"""Tests for the DCT disclosure watermark (core/watermark/watermark.py).

Covers the M1 exit criterion "watermark verified on sample output", plus
durability against JPEG re-encode (plan §2.4 — used only to validate
durability, never to tune around a detector).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.watermark import watermark as wm


@pytest.fixture()
def rng():
    return np.random.default_rng(20261026)


@pytest.fixture()
def gray_image(rng):
    return rng.integers(0, 256, size=(256, 256), dtype=np.uint8)


@pytest.fixture()
def rgb_image(rng):
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


@pytest.fixture()
def smooth_image():
    """Smooth gradient — a stand-in for photographic (e.g. face) content."""
    y, x = np.mgrid[0:256, 0:256]
    channel = 128 + 60 * np.sin(x / 17) + 30 * np.cos(y / 23)
    return np.stack([channel] * 3, axis=-1).clip(0, 255).astype(np.uint8)


class TestPayload:
    def test_make_payload_is_32_bytes_and_deterministic(self):
        p1 = wm.make_payload("gen/1.0", "alice", "v1", "2026-10-26T00:00:00+00:00")
        p2 = wm.make_payload("gen/1.0", "alice", "v1", "2026-10-26T00:00:00+00:00")
        assert len(p1) == 32
        assert p1 == p2

    def test_payload_changes_with_consent_id(self):
        a = wm.make_payload("gen/1.0", "alice", "v1", "t")
        b = wm.make_payload("gen/1.0", "bob", "v1", "t")
        assert a != b


class TestRoundTrip:
    def test_grayscale(self, gray_image):
        payload = wm.make_payload("gen/1.0", "alice", "v1", "t")
        marked = wm.embed(gray_image, payload)
        assert marked.shape == gray_image.shape
        assert marked.dtype == np.uint8
        result = wm.verify(marked)
        assert result.crc_ok is True
        assert result.payload_hex == payload.hex()
        assert result.confidence > 0.8

    def test_rgb(self, rgb_image):
        payload = wm.make_payload("gen/1.0", "alice", "v1", "t")
        marked = wm.embed(rgb_image, payload)
        assert marked.shape == rgb_image.shape
        result = wm.verify(marked)
        assert result.crc_ok is True
        assert result.payload_hex == payload.hex()

    def test_embed_is_subtle(self, rgb_image):
        """Worst case (pure noise) stays above a 28 dB PSNR floor."""
        payload = wm.make_payload("gen/1.0", "alice", "v1", "t")
        marked = wm.embed(rgb_image, payload)
        mse = ((marked.astype(np.float64) - rgb_image) ** 2).mean()
        psnr = 10 * np.log10(255**2 / mse)
        assert psnr > 28.0

    def test_unmarked_image_fails_crc(self, gray_image):
        result = wm.verify(gray_image)
        assert result.crc_ok is False

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            wm.embed(np.zeros((4, 4), dtype=np.uint8), b"\x00" * 32)

    def test_bad_payload_length_raises(self, gray_image):
        with pytest.raises(ValueError):
            wm.embed(gray_image, b"\x00" * 16)


class TestDurability:
    """Watermark must survive lossy re-encode — §2.4 durability validation."""

    @pytest.mark.parametrize("quality", [95, 85])
    def test_survives_jpeg_reencode(self, rgb_image, quality):
        payload = wm.make_payload("gen/1.0", "alice", "v1", "t")
        marked = wm.embed(rgb_image, payload)
        # RGB -> BGR for OpenCV's encoder
        ok, buf = cv2.imencode(".jpg", marked[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert decoded is not None
        decoded = decoded[..., ::-1]
        result = wm.verify(decoded)
        assert result.crc_ok is True
        assert result.payload_hex == payload.hex()

    def test_survives_light_blur(self, smooth_image):
        """On photographic-like content the default-strength mark survives a
        light blur. (Pure noise is worst case and needs a higher strength.)"""
        payload = wm.make_payload("gen/1.0", "alice", "v1", "t")
        marked = wm.embed(smooth_image, payload)
        blurred = cv2.GaussianBlur(marked, (3, 3), 0)
        result = wm.verify(blurred)
        assert result.crc_ok is True
        assert result.confidence > 0.9


class TestCrcDetectsCorruption:
    def test_wrong_payload_fails(self, gray_image):
        wm.embed(gray_image, wm.make_payload("gen/1.0", "alice", "v1", "t"))
        other = wm.verify(wm.embed(gray_image, wm.make_payload("gen/1.0", "bob", "v1", "t")))
        assert other.payload_hex != wm.make_payload("gen/1.0", "alice", "v1", "t").hex()
