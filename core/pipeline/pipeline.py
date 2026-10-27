"""Pipeline orchestrator with the consent gate at the entry point (plan §2).

Constructing a SwapPipeline requires a valid consent record for the source
identity covering this project and use — ConsentError propagates out of
__init__, so no unconsented frame can ever reach the swap stage. Every
finalized output passes through the disclosure gate: badge + DCT watermark
+ C2PA sidecar (see finalize()).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .. import __generator_id__, consent
from ..watermark import badge as badge_mod
from ..watermark import c2pa as c2pa_mod
from ..watermark import watermark as wm_mod
from . import blend as blend_mod
from .detect import Detector, Face
from .swap import IdentityEncoder, Swapper


class SwapPipeline:
    """detect -> align -> swap -> color-match -> paste-back, consent-gated."""

    def __init__(
        self,
        *,
        identity_id: str,
        project: str,
        use: str,
        detector: Detector,
        encoder: IdentityEncoder,
        swapper: Swapper,
        model_version: str = "inswapper_128/v1",
    ):
        # Hard gate — raises ConsentError if the record is missing, outside
        # scope, expired or revoked. Nothing after this line can run ungated.
        self.consent_record = consent.check(identity_id, project=project, use=use)
        self.detector = detector
        self.encoder = encoder
        self.swapper = swapper
        self.model_version = model_version
        self._source_latent: Optional[np.ndarray] = None

    def set_source(self, frames: List[np.ndarray], kps_list: List[np.ndarray]) -> None:
        """Encode the consented source identity (embedding-averaged, plan §6.2)."""
        if len(frames) != len(kps_list) or not frames:
            raise ValueError("frames and kps_list must be non-empty and aligned")
        self._source_latent = self.encoder.encode_frames(frames, kps_list)

    @property
    def ready(self) -> bool:
        return self._source_latent is not None

    def process_frame(self, bgr: np.ndarray, *, color_match: bool = True) -> np.ndarray:
        """Swap every detected face onto the consented identity."""
        if self._source_latent is None:
            raise RuntimeError("set_source() must run before process_frame()")
        out = bgr
        for face in self.detector.detect(bgr):
            out = self._swap_one(out, face, color_match=color_match)
        return out

    def _swap_one(self, bgr: np.ndarray, face: Face, *, color_match: bool = True) -> np.ndarray:
        fake, M = self.swapper.swap_face(bgr, face, self._source_latent)
        if color_match:
            # match the swapped crop to the target's own aligned crop
            ref = cv2.warpAffine(
                bgr, M, (fake.shape[0], fake.shape[0]), borderValue=0.0
            )
            fake = blend_mod.reinhard_transfer(fake, ref)
        return blend_mod.paste_back(bgr, fake, M)

    # ------------------------------------------------------------------
    # Disclosure gate — every finalized output goes through here (plan §2.2)

    def finalize(
        self, bgr: np.ndarray, out_path: Path, *, visible_badge: bool = True
    ) -> dict:
        """Badge + invisible watermark + provenance, then write the file."""
        stamped = badge_mod.add_badge(bgr) if visible_badge else bgr
        ts = datetime.now(timezone.utc).isoformat()
        payload = wm_mod.make_payload(
            __generator_id__, self.consent_record.identity_id, self.model_version, ts
        )
        marked = wm_mod.embed(stamped, payload)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), marked)  # pipeline is BGR end-to-end
        prov = c2pa_mod.sign_output(
            out_path,
            generator_id=__generator_id__,
            consent_id=self.consent_record.identity_id,
            model_version=self.model_version,
            use="image",
        )
        return {"file": str(out_path), "provenance": prov}
