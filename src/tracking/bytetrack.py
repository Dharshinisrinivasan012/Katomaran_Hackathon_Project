"""Ultralytics BYTETracker adapter. Does not run YOLO or recognition."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from src.config.loader import TrackingConfig
from src.detection.results import BoundingBox, FaceDetection
from src.tracking.exceptions import TrackerBackendError
from src.tracking.results import TrackedFace

logger = logging.getLogger(__name__)


class _DetBatch:
    """Minimal Ultralytics Results/Boxes stand-in for BYTETracker.update."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, mask: np.ndarray) -> _DetBatch:
        if self.conf.size == 0:
            return _empty_batch()
        return _DetBatch(self.xywh[mask], self.conf[mask], self.cls[mask])


def _empty_batch() -> _DetBatch:
    return _DetBatch(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (x1 + w / 2.0, y1 + h / 2.0, w, h)


def detections_to_batch(detections: Sequence[FaceDetection]) -> _DetBatch:
    if not detections:
        return _empty_batch()
    xywh = np.zeros((len(detections), 4), dtype=np.float32)
    conf = np.zeros((len(detections),), dtype=np.float32)
    cls = np.zeros((len(detections),), dtype=np.float32)
    for i, det in enumerate(detections):
        xywh[i] = _xyxy_to_xywh(det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2)
        conf[i] = float(det.confidence)
        cls[i] = float(det.class_id if det.class_id is not None else 0)
    return _DetBatch(xywh, conf, cls)


class ByteTrackFaceTracker:
    """Wraps ``ultralytics.trackers.BYTETracker``.

    Detection frames (a list, including empty): full ByteTrack update — associate,
    spawn, and drop tracks.

    Skipped detection frames (``detections is None``): Kalman predict only on
    already-activated tracks. No new ``track_id`` values are created.
    """

    def __init__(self, config: TrackingConfig) -> None:
        if config.tracker_type != "bytetrack":
            raise TrackerBackendError(
                f"Unsupported tracking.tracker_type {config.tracker_type!r}; expected 'bytetrack'"
            )
        try:
            from ultralytics.trackers import BYTETracker
        except ImportError as exc:
            raise TrackerBackendError(
                "The 'ultralytics' package is required for ByteTrack. "
                "Install dependencies from requirements.txt."
            ) from exc

        low = min(0.1, float(config.track_thresh) * 0.5)
        if low >= config.track_thresh:
            low = max(0.0, float(config.track_thresh) - 1e-3)
        args = SimpleNamespace(
            track_high_thresh=float(config.track_thresh),
            track_low_thresh=low,
            new_track_thresh=float(config.track_thresh),
            track_buffer=int(config.track_buffer),
            match_thresh=float(config.match_thresh),
            fuse_score=True,
        )
        self._config = config
        self._byte = BYTETracker(args)
        logger.info(
            "Initialized ByteTrack (track_thresh=%s, match_thresh=%s, track_buffer=%s)",
            config.track_thresh,
            config.match_thresh,
            config.track_buffer,
        )

    def update(
        self,
        detections: Sequence[FaceDetection] | None,
        *,
        frame_index: int,
        image: np.ndarray | None = None,
    ) -> list[TrackedFace]:
        if detections is None:
            return self._predict_existing(frame_index)
        batch = detections_to_batch(detections)
        raw = self._byte.update(batch, img=image)
        return self._parse_output(raw, frame_index, state="tracked")

    def reset(self) -> None:
        self._byte.reset()

    def _predict_existing(self, frame_index: int) -> list[TrackedFace]:
        from ultralytics.trackers.byte_tracker import STrack
        from ultralytics.trackers.basetrack import TrackState

        active = [
            t
            for t in self._byte.tracked_stracks
            if t.is_activated and t.state == TrackState.Tracked
        ]
        if active:
            STrack.multi_predict(active)
        out: list[TrackedFace] = []
        for track in active:
            x1, y1, x2, y2 = (float(v) for v in track.xyxy.tolist())
            out.append(
                TrackedFace(
                    track_id=int(track.track_id),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=float(track.score),
                    frame_index=frame_index,
                    state="predicted",
                )
            )
        return out

    def _parse_output(
        self, raw: np.ndarray, frame_index: int, *, state: str
    ) -> list[TrackedFace]:
        if raw is None or len(raw) == 0:
            return []
        tracks: list[TrackedFace] = []
        for row in np.atleast_2d(raw):
            x1, y1, x2, y2 = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            track_id = int(row[4])
            score = float(row[5])
            tracks.append(
                TrackedFace(
                    track_id=track_id,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=score,
                    frame_index=frame_index,
                    state=state,
                )
            )
        return tracks
