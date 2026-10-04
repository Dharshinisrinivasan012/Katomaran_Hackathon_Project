"""Development preview: YOLO → ByteTrack → InsightFace embedding → gallery lookup.

This preview demonstrates the Phase 5 pipeline segment:
  video frame → YOLO detection → ByteTrack track_id → face crop → InsightFace
  embedding → gallery search → existing face_id / new registration

It does NOT implement:
  - entry/exit logic
  - visitor counting
  - events persisted from the video loop to SQLite
  - the final production pipeline

IMPORTANT:
  track_id  = temporary ByteTrack motion identity
  face_id   = persistent identity assigned by InsightFace + gallery

Run::

    python -m src.preview_recognition --no-display --max-frames 30 \\
        --save-dir tmp/preview_recognition

"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

from src.config.loader import load_config
from src.database.connection import open_initialized
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
from src.recognition.exceptions import ModelLoadError, RecognitionError
from src.recognition.factory import create_face_embedder, create_gallery_matcher
from src.registration.registrar import FaceRegistrar
from src.tracking.exceptions import TrackingError
from src.tracking.factory import create_face_tracker
from src.tracking.results import TrackedFace

logger = logging.getLogger(__name__)


def _color_for_id(face_id: str) -> tuple[int, int, int]:
    h = hash(face_id) & 0xFFFFFF
    return (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)


def _draw_results(
    image: np.ndarray,
    tracks: list[TrackedFace],
    track_to_face: dict[int, str],
    *,
    new_ids: set[str],
) -> np.ndarray:
    annotated = image.copy()
    for tr in tracks:
        x1, y1, x2, y2 = tr.bbox.as_int_tuple()
        face_id = track_to_face.get(tr.track_id, "?")
        is_new = face_id in new_ids
        color = _color_for_id(face_id) if face_id != "?" else (128, 128, 128)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        tag = f"ID {tr.track_id} {'NEW' if is_new else 'RE-ID'}"
        short_fid = face_id[-8:] if len(face_id) > 8 else face_id
        cv2.putText(
            annotated, tag,
            (x1, max(0, y1 - 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
        cv2.putText(
            annotated, short_fid,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )
    return annotated


def run_preview(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 5 preview: YOLO + ByteTrack + InsightFace recognition."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--db-path", default="",
        help="SQLite path override (default: from config). Use :memory: for ephemeral run.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

    config = load_config(args.config)
    db_path = args.db_path if args.db_path else config.database.sqlite_path

    # Open DB (in-memory for ephemeral preview run to avoid polluting production DB)
    conn = open_initialized(db_path)
    conn.commit()

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

    try:
        embedder = create_face_embedder(config)
    except ModelLoadError as exc:
        logger.error("InsightFace load error: %s", exc)
        return 5
    except RecognitionError as exc:
        logger.error("Recognition error: %s", exc)
        return 5

    matcher = create_gallery_matcher(config)
    registrar = FaceRegistrar(config.recognition, config.registration, matcher)

    skip = DetectionSkipPolicy(config.detection.detection_skip_frames)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    track_to_face: dict[int, str] = {}   # track_id → face_id (session cache)
    new_registered_ids: set[str] = set()  # face_ids registered this run

    processed = 0
    detect_frames = 0
    total_detections = 0
    total_embeddings = 0
    total_new = 0
    total_reid = 0

    try:
        for frame in source:
            if skip.should_detect(frame.index):
                dets = list(detector.detect(frame))
                total_detections += len(dets)
                detect_frames += 1
                tracks = list(tracker.update(dets, frame_index=frame.index, image=frame.image))

                for tr in tracks:
                    # Pass the full frame + bbox hint — the engine applies
                    # padding internally so InsightFace SCRFD has face context.
                    bbox = (
                        float(tr.bbox.x1), float(tr.bbox.y1),
                        float(tr.bbox.x2), float(tr.bbox.y2),
                    )
                    emb = embedder.embed(frame.image, bbox=bbox)
                    if emb is None:
                        continue
                    total_embeddings += 1

                    # Extract the padded crop for saving alongside registration
                    x1, y1, x2, y2 = tr.bbox.as_int_tuple()
                    fh, fw = frame.image.shape[:2]
                    bw = x2 - x1; bh = y2 - y1
                    px = int(bw * 0.3); py = int(bh * 0.3)
                    cx1 = max(0, x1 - px); cy1 = max(0, y1 - py)
                    cx2 = min(fw, x2 + px); cy2 = min(fh, y2 + py)
                    crop = frame.image[cy1:cy2, cx1:cx2]

                    identity = registrar.identify_or_register(
                        conn, emb,
                        crop_image=crop,
                        source_label=f"frame_{frame.index}",
                    )
                    conn.commit()

                    if identity.face_id:
                        track_to_face[tr.track_id] = identity.face_id
                        if identity.is_new:
                            new_registered_ids.add(identity.face_id)
                            total_new += 1
                            logger.info(
                                "Frame %d  track_id=%d  → NEW face_id=%s",
                                frame.index, tr.track_id, identity.face_id,
                            )
                        else:
                            total_reid += 1
                            logger.info(
                                "Frame %d  track_id=%d  → RE-ID face_id=%s (sim=%.4f)",
                                frame.index, tr.track_id, identity.face_id, identity.similarity,
                            )
            else:
                tracks = list(tracker.update(None, frame_index=frame.index, image=frame.image))

            annotated = _draw_results(
                frame.image, tracks, track_to_face, new_ids=new_registered_ids
            )
            if save_dir:
                cv2.imwrite(str(save_dir / f"frame_{frame.index:06d}.jpg"), annotated)
            if not args.no_display:
                cv2.imshow("Phase 5 recognition preview", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            if args.max_frames > 0 and processed >= args.max_frames:
                break
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
        conn.close()

    logger.info(
        "Recognition preview finished: frames=%d  detect_frames=%d  "
        "detections=%d  embeddings=%d  new_registered=%d  re_identified=%d  "
        "unique_face_ids=%d",
        processed, detect_frames, total_detections, total_embeddings,
        total_new, total_reid, len(set(track_to_face.values())),
    )
    return 0


def main() -> None:
    sys.exit(run_preview())


if __name__ == "__main__":
    main()
