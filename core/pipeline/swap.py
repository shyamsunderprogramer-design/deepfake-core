"""Identity swap stages (plan §4 stage 3): ArcFace encoder + inswapper.

The swapper is *not* opinionated about sessions — inject any object with a
.onnxruntime-compatible .run/.get_inputs interface (used by unit tests).
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from . import align as align_mod
from .detect import Face


class IdentityEncoder:
    """ArcFace 112x112 -> 512-d unit-normalized embedding."""

    def __init__(self, session, size: int = 112):
        self.session = session
        self.size = size
        self.input_name = session.get_inputs()[0].name

    def encode_face(self, bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        crop, _ = align_mod.align_crop(bgr, kps, self.size)
        blob = ((crop.astype(np.float32) - 127.5) / 127.5)[..., ::-1]
        blob = blob.transpose(2, 0, 1)[None]
        feat = self.session.run(None, {self.input_name: blob})[0]
        vec = feat.reshape(-1).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def encode_frames(
        self, frames: List[np.ndarray], kps_list: List[np.ndarray]
    ) -> np.ndarray:
        """Plan §6.2 accuracy upgrade: average embeddings over the source set."""
        vecs = [self.encode_face(f, k) for f, k in zip(frames, kps_list)]
        mean = np.mean(vecs, axis=0)
        return mean / (np.linalg.norm(mean) + 1e-12)


class Swapper:
    """inswapper_128-class model: (aligned target crop, latent) -> swapped crop."""

    def __init__(self, session, size: int = 128):
        self.session = session
        self.size = size
        inputs = session.get_inputs()
        # inswapper exposes (target image, latent vector); identify by rank.
        self._img_input = next(i for i in inputs if len(i.shape) == 4).name
        self._latent_input = next(i for i in inputs if len(i.shape) == 2).name
        self.output_name = session.get_outputs()[0].name

    def swap_face(
        self, bgr: np.ndarray, face: Face, latent: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """(swapped crop 128x128 BGR uint8, M) — paste-back is blend's job."""
        crop, M = align_mod.align_crop(bgr, face.kps, self.size)
        blob = cv2.dnn.blobFromImage(
            crop, 1.0 / 255.0, (self.size, self.size), (0, 0, 0), swapRB=True
        )
        latent = latent.reshape(1, -1).astype(np.float32)
        latent /= np.linalg.norm(latent) + 1e-12
        out = self.session.run(
            [self.output_name], {self._img_input: blob, self._latent_input: latent}
        )[0]
        fake = np.clip(out.transpose(0, 2, 3, 1)[0] * 255.0, 0, 255).astype(np.uint8)
        return fake[..., ::-1], M  # model output is RGB; pipeline is BGR
