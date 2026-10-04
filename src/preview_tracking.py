"""Development preview: YOLO faces + ByteTrack IDs. Not recognition or a product UI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from src.config.loader import load_config
from src.detection.exceptions import DetectionError, ModelNotFoundError
from src.detection.factory import create_face_detector
from src.detection.skip import DetectionSkipPolicy
from src.input.exceptions import (
    RtspNotConfiguredError,
    VideoFileNotFoundError,
    VideoPathNotConfiguredError,
    VideoSourceError,
)
from src.input.factory import create_frame_source
from src.tracking.exceptions import TrackingError
from src.tracking.factory import create_face_tracker
from src.tracking.results import TrackedFace

logger = logging.getLogger(__name__)


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    rng = (track_id * 47) % 255, (track_id * 97) % 255, (track_id * 151) % 255
    return (int(40 + rng[0] * 0.8), int(40 + rng[1] * 0.8), int(40 + rng[2] * 0.8))


def _draw_tracks(image, tracks: list[TrackedFace]):
    annotated = image.copy()
    for tr in tracks:
        x1, y1, x2, y2 = tr.bbox.as_int_tuple()
        color = _color_for_id(tr.track_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tr.track_id} {tr.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_preview(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preview ByteTrack IDs on YOLO face detections (not face identity)."
    )
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    parser.add_argument(
        "--save-dir",
        default="",
        help="Optional directory to write annotated frames (jpg)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a GUI window (use with --save-dir on headless machines)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    try:
        source = create_frame_source(config)
    except (VideoPathNotConfiguredError, VideoFileNotFoundError, RtspNotConfiguredError) as exc:
        logger.error("%s", exc)
        return 2
    except VideoSourceError as exc:
        logger.error("Video source error: %s", exc)
        return 2

    try:
        detector = create_face_detector(config)
    except ModelNotFoundError as exc:
        logger.error("%s", exc)
        return 3
    except DetectionError as exc:
        logger.error("Detector error: %s", exc)
        return 3

    try:
        tracker = create_face_tracker(config)
    except TrackingError as exc:
        logger.error("Tracker error: %s", exc)
        return 4

    skip = DetectionSkipPolicy(config.detection.detection_skip_frames)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    detection_count = 0
    unique_ids: set[int] = set()
    prev_ids: set[int] | None = None
    persisted_steps = 0
    comparable_steps = 0
    try:
        for frame in source:
            if skip.should_detect(frame.index):
                dets = list(detector.detect(frame))
                detection_count += len(dets)
                tracks = list(tracker.update(dets, frame_index=frame.index, image=frame.image))
                logger.info(
                    "Frame %s: detect %s face(s) → %s track(s) %s",
                    frame.index,
                    len(dets),
                    len(tracks),
                    [t.track_id for t in tracks],
                )
            else:
                tracks = list(tracker.update(None, frame_index=frame.index, image=frame.image))
                logger.info(
                    "Frame %s: skipped YOLO; predicted %s track(s) %s",
                    frame.index,
                    len(tracks),
                    [t.track_id for t in tracks],
                )

            ids = {t.track_id for t in tracks}
            unique_ids.update(ids)
            if prev_ids is not None and prev_ids and ids:
                comparable_steps += 1
                if prev_ids & ids:
                    persisted_steps += 1
            prev_ids = ids

            annotated = _draw_tracks(frame.image, tracks)
            if save_dir is not None:
                out = save_dir / f"frame_{frame.index:06d}.jpg"
                cv2.imwrite(str(out), annotated)
            if not args.no_display:
                cv2.imshow("ByteTrack preview", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            if args.max_frames > 0 and processed >= args.max_frames:
                break
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()

    logger.info(
        "Tracking preview finished: frames=%s detections=%s unique_track_ids=%s "
        "id_overlap_steps=%s/%s (track IDs are not face identities)",
        processed,
        detection_count,
        len(unique_ids),
        persisted_steps,
        comparable_steps,
    )
    return 0


def main() -> None:
    sys.exit(run_preview())


if __name__ == "__main__":
    main()
