"""Development preview: video frames + YOLO boxes. Not the full application."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from src.config.loader import load_config
from src.detection.exceptions import DetectionError, ModelNotFoundError
from src.detection.factory import create_face_detector
from src.detection.results import FaceDetection
from src.detection.skip import DetectionSkipPolicy
from src.input.exceptions import (
    RtspNotConfiguredError,
    VideoFileNotFoundError,
    VideoPathNotConfiguredError,
    VideoSourceError,
)
from src.input.factory import create_frame_source

logger = logging.getLogger(__name__)


def _draw_detections(image, detections: list[FaceDetection]):
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox.as_int_tuple()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_preview(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preview YOLO face detections on the configured video source."
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

    skip = DetectionSkipPolicy(config.detection.detection_skip_frames)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    last_detections: list[FaceDetection] = []
    processed = 0
    try:
        for frame in source:
            if skip.should_detect(frame.index):
                last_detections = list(detector.detect(frame))
                logger.info(
                    "Frame %s: ran detector, %s face(s)",
                    frame.index,
                    len(last_detections),
                )
            else:
                logger.debug(
                    "Frame %s: skipped detector (detection_skip_frames=%s)",
                    frame.index,
                    skip.skip_frames,
                )

            annotated = _draw_detections(frame.image, last_detections)
            if save_dir is not None:
                out = save_dir / f"frame_{frame.index:06d}.jpg"
                cv2.imwrite(str(out), annotated)
            if not args.no_display:
                cv2.imshow("YOLO face preview", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            if args.max_frames > 0 and processed >= args.max_frames:
                break
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()

    logger.info("Preview finished after %s frame(s)", processed)
    return 0


def main() -> None:
    sys.exit(run_preview())


if __name__ == "__main__":
    main()
