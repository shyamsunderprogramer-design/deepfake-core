"""Real-time threaded swap loop (plan §6.1, §7.1 rules 5).

Three stages decoupled by drop-oldest queues: capture -> process -> display.
Detection runs every `detect_interval` frames; between runs, the last face
is extrapolated by its estimated velocity (the paste found frozen boxes
visibly snap between detections — velocity extrapolation avoids that).
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..pipeline.detect import Face


class DropOldestQueue(queue.Queue):
    """maxsize-limited queue: on full, drop the oldest item (never blocks)."""

    def put_latest(self, item) -> None:
        try:
            self.put_nowait(item)
        except queue.Full:
            try:
                self.get_nowait()
            except queue.Empty:
                pass
            self.put_nowait(item)


@dataclass
class TrackedFace:
    face: Face
    vx: float = 0.0  # bbox-centre velocity, px/frame
    vy: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        b = self.face.bbox
        return (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2))


class FaceTracker:
    """Velocity extrapolation between detector runs (replaces frozen boxes)."""

    def __init__(self, detect_interval: int = 3, max_missed: int = 6):
        self.detect_interval = max(1, detect_interval)
        self.max_missed = max_missed
        self._tracked: Optional[TrackedFace] = None
        self._missed = 0
        self._count = 0

    def update(self, detector, frame: np.ndarray) -> Optional[Face]:
        run_detect = self._tracked is None or (self._count % self.detect_interval) == 0
        self._count += 1
        if run_detect:
            face = detector.detect_one(frame)
            if face is None:
                self._missed += 1
                if self._missed > self.max_missed:
                    self._tracked = None
                return self._predict()
            self._missed = 0
            if self._tracked is None:
                self._tracked = TrackedFace(face=face)
            else:
                prev = self._tracked.face
                oc = ((prev.bbox[0] + prev.bbox[2]) / 2, (prev.bbox[1] + prev.bbox[3]) / 2)
                nc = ((face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2)
                n = max(1, self.detect_interval)
                self._tracked = TrackedFace(
                    face=face,
                    vx=float((nc[0] - oc[0]) / n),
                    vy=float((nc[1] - oc[1]) / n),
                )
            return face
        return self._predict()

    def _predict(self) -> Optional[Face]:
        t = self._tracked
        if t is None:
            return None
        dx, dy = t.vx, t.vy
        b = t.face.bbox + np.array([dx, dy, dx, dy])
        k = t.face.kps + np.array([dx, dy])
        return Face(bbox=b, kps=k, score=t.face.score)


def run_live(pipeline, camera_index: int = 0, detect_interval: int = 3) -> int:
    """Camera -> swap -> preview window. Badge is applied per displayed frame."""
    from ..watermark.badge import add_badge

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"error: cannot open camera {camera_index}")
        return 2
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    raw_q: DropOldestQueue = DropOldestQueue(maxsize=2)
    stop = threading.Event()

    def capture():
        while not stop.is_set():
            ok, frame = cap.read()
            if ok:
                raw_q.put_latest(frame)

    tracker = FaceTracker(detect_interval=detect_interval)
    worker = threading.Thread(target=capture, daemon=True)
    worker.start()
    print("live: press q to quit")
    try:
        while True:
            try:
                frame = raw_q.get(timeout=0.5)
            except queue.Empty:
                continue
            face = tracker.update(pipeline.detector, frame)
            out = pipeline._swap_one(frame, face) if face is not None else frame
            cv2.imshow("deepfake-core (consent-gated)", add_badge(out))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
    return 0
