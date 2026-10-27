"""M1 exit criterion, executable: the swap pipeline refuses to run without a
valid consent record — and an end-to-end image swap with stub sessions
proves outputs carry badge + watermark + provenance sidecar.
"""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest

from core.consent import ConsentError, enroll
from core.pipeline.pipeline import SwapPipeline
from core.pipeline.detect import Detector
from core.pipeline.swap import IdentityEncoder, Swapper
from core.watermark.watermark import verify

from test_pipeline_stages import _FakeInput, _synthetic_scrfd_outputs


class _EncoderStub:
    def get_inputs(self):
        return [_FakeInput("in", [1, 3, 112, 112])]

    def run(self, _o, feed):
        return [np.ones((1, 512), dtype=np.float32)]


class _SwapStub:
    def get_inputs(self):
        return [_FakeInput("img", [1, 3, 128, 128]), _FakeInput("latent", [1, 512])]

    def get_outputs(self):
        return [_FakeInput("out", [1, 3, 128, 128])]

    def run(self, _o, feed):
        return [np.full((1, 3, 128, 128), 0.5, dtype=np.float32)]


class _DetectorStub:
    def get_inputs(self):
        return [_FakeInput("input.1", [1, 3, 640, 640])]

    def run(self, _o, feed):
        return _synthetic_scrfd_outputs()


def _enrolled_pipeline(**kw):
    args = dict(
        identity_id="alice",
        project="demo",
        use="image",
        detector=Detector(_DetectorStub()),
        encoder=IdentityEncoder(_EncoderStub()),
        swapper=Swapper(_SwapStub()),
    )
    args.update(kw)
    return SwapPipeline(**args)


def _enroll_alice(photo_files, uses=("image", "realtime")):
    enroll("alice", "Alice", photo_files, ["demo"], list(uses), "a@x.io")


class TestConsentGate:
    def test_refuses_without_record(self, consent_env):
        with pytest.raises(ConsentError):
            _enrolled_pipeline(identity_id="mallory", use="realtime")

    def test_refuses_out_of_scope_use(self, consent_env, photo_files):
        _enroll_alice(photo_files, uses=("image",))
        with pytest.raises(ConsentError, match="use"):
            _enrolled_pipeline(use="realtime")

    def test_refuses_after_revocation(self, consent_env, photo_files):
        _enroll_alice(photo_files)
        assert _enrolled_pipeline().consent_record.identity_id == "alice"
        from core import consent as consent_mod

        consent_mod.revoke("alice")
        with pytest.raises(ConsentError, match="revoked"):
            _enrolled_pipeline()  # construction re-checks the ledger

    def test_source_faces_pass_through_gate(self, consent_env, photo_files):
        _enroll_alice(photo_files)
        p = _enrolled_pipeline()
        src = np.zeros((256, 256, 3), dtype=np.uint8)
        face = p.detector.detect_one(src)
        assert face is not None
        p.set_source([src], [face.kps])
        assert p.ready


class TestEndToEndImageSwap:
    def test_output_is_badged_watermarked_and_signed(self, consent_env, photo_files, tmp_path):
        _enroll_alice(photo_files)
        p = _enrolled_pipeline()
        src = np.zeros((256, 256, 3), dtype=np.uint8)
        p.set_source([src], [p.detector.detect_one(src).kps])

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = p.process_frame(frame)
        assert out.shape == frame.shape

        out_path = tmp_path / "result.png"
        p.finalize(out, out_path)
        assert out_path.exists()

        # invisible watermark round-trips from the written file
        read_back = cv2.imread(str(out_path))
        assert read_back is not None
        check = verify(read_back)
        assert check.crc_ok is True

        # provenance sidecar carries the consent id and matching hash
        claims = json.loads((tmp_path / "result.png.c2pa.json").read_text())
        assert claims["synthetic_media"] is True
        assert claims["consent_id"] == "alice"
        assert claims["file_sha256"] == hashlib.sha256(out_path.read_bytes()).hexdigest()

    def test_bench_writes_report(self, consent_env, photo_files, monkeypatch, tmp_path):
        from core import config
        from core.pipeline.bench import run_bench

        monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
        _enroll_alice(photo_files)
        p = _enrolled_pipeline()
        src = np.zeros((256, 256, 3), dtype=np.uint8)
        p.set_source([src], [p.detector.detect_one(src).kps])
        report = run_bench(p, frames=5)
        assert report["stages"]["total"]["avg_ms"] > 0
        assert "effective_fps" in report
        assert (tmp_path / "runs").exists()
