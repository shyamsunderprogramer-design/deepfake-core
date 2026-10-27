"""SCRFD face detection (plan §4 stage 1).

Fresh numpy/cv2 implementation of the SCRFD decode (buffalo_l det_10g:
strides 8/16/32, num_anchors=2, distance-based bbox + 5-point kps decode).
Detection is routed to CPU+GPU ComputeUnits (see core.models.session) so it
runs concurrently with the ANE-bound swap model.

The Decoder is a pure function of raw ONNX outputs, so the decode can be
unit-tested without the model file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

_STRIDES = (8, 16, 32)
_NUM_ANCHORS = 2


@dataclass
class Face:
    bbox: np.ndarray  # (4,) x1,y1,x2,y2 in original-frame coords
    kps: np.ndarray   # (5, 2) in original-frame coords
    score: float

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


def _distance_to_bbox(anchor: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = anchor[:, 0] - distance[:, 0]
    y1 = anchor[:, 1] - distance[:, 1]
    x2 = anchor[:, 0] + distance[:, 2]
    y2 = anchor[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance_to_kps(anchor: np.ndarray, distance: np.ndarray) -> np.ndarray:
    return anchor[:, None, :] + distance.reshape(-1, 5, 2)


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
    xywh = np.stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]],
        axis=-1,
    )
    idx = cv2.dnn.NMSBoxes(
        bboxes=[list(map(float, b)) for b in xywh],
        scores=[float(s) for s in scores],
        score_threshold=0.0,
        nms_threshold=thresh,
    )
    return [int(i) for i in (np.asarray(idx).reshape(-1) if len(idx) else [])]


def _maybe_sigmoid(x: np.ndarray) -> np.ndarray:
    """det_10g variants differ: some emit logits, others probabilities.
    Values outside [0,1] mean logits -> squash; otherwise pass through."""
    flat = x.reshape(-1)
    if flat.size and (flat.min() < 0.0 or flat.max() > 1.0):
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    return x.astype(np.float32)


class Decoder:
    """Anchor-based SCRFD decode for the 10g_bnkps head."""

    def __init__(self, input_size: int = 640):
        self.input_size = input_size
        self._anchors = self._make_anchors()

    def _make_anchors(self) -> np.ndarray:
        cols = []
        for stride in _STRIDES:
            side = self.input_size // stride
            ys, xs = np.mgrid[0:side, 0:side]
            base = np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride
            cols.append(np.repeat(base, _NUM_ANCHORS, axis=0))
        return np.concatenate(cols, axis=0).astype(np.float32)

    def decode(
        self, outputs: List[np.ndarray], score_thresh: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(boxes, kpss, scores) in 640-input coords, sorted by score desc.

        `outputs` is the model's raw output list; heads are identified by
        their last-dim (1=score, 4=bbox, 10=kps) rather than by position.
        """
        scores_all, boxes_all, kps_all = [], [], []
        # group heads by their last-dim signature: 1=score, 4=bbox, 10=kps
        heads: dict[int, dict[str, np.ndarray]] = {}
        for out in outputs:
            o = np.asarray(out)
            o = o.reshape(o.shape[0], -1) if o.ndim == 3 else o
            n = o.shape[-1]
            # each stride block: side^2 * num_anchors rows
            kind = {1: "score", 4: "bbox", 10: "kps"}.get(n)
            if kind is None:
                continue
            rows = o.shape[0]
            side = int(round((rows / _NUM_ANCHORS) ** 0.5))
            stride = self.input_size // side
            heads.setdefault(stride, {})[kind] = o
        offset = 0
        for stride in _STRIDES:
            h = heads.get(stride)
            if h is None or "score" not in h or "bbox" not in h:
                continue
            count = h["score"].shape[0]
            anchors = self._anchors[offset : offset + count]
            offset += count
            scores = _maybe_sigmoid(h["score"]).reshape(-1)
            keep = scores > score_thresh
            if not keep.any():
                continue
            scores_all.append(scores[keep])
            boxes_all.append(
                _distance_to_bbox(anchors[keep], h["bbox"][keep] * stride)
            )
            if "kps" in h:
                kps_all.append(_distance_to_kps(anchors[keep], h["kps"][keep] * stride))
            else:
                kps_all.append(np.zeros((int(keep.sum()), 5, 2), dtype=np.float32))
        if not scores_all:
            return np.empty((0, 4)), np.empty((0, 5, 2)), np.empty((0,))
        boxes = np.concatenate(boxes_all)
        kpss = np.concatenate(kps_all)
        scores = np.concatenate(scores_all)
        order = np.argsort(-scores)
        return boxes[order], kpss[order], scores[order]


def letterbox(bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize preserving aspect; pad to square. Returns (canvas, scale, dx, dy)."""
    h, w = bgr.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(bgr, (nw, nh))
    canvas = np.full((size, size, 3), 127.5, dtype=np.float32)
    dy, dx = (size - nh) // 2, (size - nw) // 2
    canvas[dy : dy + nh, dx : dx + nw] = resized
    return canvas, scale, float(dx), float(dy)


class Detector:
    """Session-backed SCRFD detector. Pass a stub with .run for tests."""

    def __init__(self, session, input_size: int = 640, score_thresh: float = 0.5):
        self.session = session
        self.input_size = input_size
        self.score_thresh = score_thresh
        self.decoder = Decoder(input_size)
        self.input_name = session.get_inputs()[0].name

    def detect(self, bgr: np.ndarray, nms_thresh: float = 0.4) -> List[Face]:
        canvas, scale, dx, dy = letterbox(bgr, self.input_size)
        blob = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None].astype(np.float32)
        outputs = self.session.run(None, {self.input_name: blob})
        boxes, kpss, scores = self.decoder.decode(outputs, self.score_thresh)
        keep = nms(boxes, scores, nms_thresh)
        faces: List[Face] = []
        for i in keep:
            box = (boxes[i] - [dx, dy, dx, dy]) / scale
            kps = (kpss[i] - [dx, dy]) / scale
            faces.append(Face(bbox=box, kps=kps, score=float(scores[i])))
        return faces

    def detect_one(self, bgr: np.ndarray, **kw) -> Optional[Face]:
        faces = self.detect(bgr, **kw)
        if not faces:
            return None
        return max(faces, key=lambda f: f.width * f.height)
