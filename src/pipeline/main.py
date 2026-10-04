"""Complete end-to-end pipeline entry point.

Usage::

    python -m src.pipeline.main                          # full video from config
    python -m src.pipeline.main --max-frames 60          # bounded run
    python -m src.pipeline.main --no-display             # headless
    python -m src.pipeline.main --save-dir tmp/run       # save annotated frames
    python -m src.pipeline.main --config path/config.json

Pipeline stages (in order per frame):
    Input → YOLO detection (every N+1 frames)
         → ByteTrack (all frames, Kalman-predict on skip)
         → InsightFace embedding (detection frames only)
         → FaceRegistrar (register new / re-id known)
         → EntryExitFSM (line-crossing detection)
         → VisitManager (open/close visits, save snapshots)
         → events.log
         → SQLite commit

Identity rules (enforced throughout):
    track_id  = temporary ByteTrack motion identity
    face_id   = persistent SQLite identity (InsightFace gallery)
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

from src.config.loader import PROJECT_ROOT, load_config
from src.database.connection import open_initialized, utc_now_iso
from src.database.repositories import get_person_by_face_id
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
from src.logging.setup import get_event_logger, setup_logging
from src.pipeline.entry_exit import EntryExitFSM
from src.pipeline.visit_manager import VisitManager
from src.recognition.exceptions import ModelLoadError, RecognitionError
from src.recognition.factory import create_face_embedder, create_gallery_matcher
from src.registration.registrar import FaceRegistrar
from src.tracking.exceptions import TrackingError
from src.tracking.factory import create_face_tracker
from src.tracking.results import TrackedFace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _color_for_face(face_id: str) -> tuple[int, int, int]:
    h = hash(face_id) & 0xFFFFFF
    b = max(40, h & 0xFF)
    g = max(40, (h >> 8) & 0xFF)
    r = max(40, (h >> 16) & 0xFF)
    return (b, g, r)


def _draw_frame(
    image: np.ndarray,
    tracks: list[TrackedFace],
    track_to_face: dict[int, str],
    inside_face_ids: set[str],
    *,
    entry_line: list,
    exit_line: list,
    visitor_count: int,
) -> np.ndarray:
    ann = image.copy()
    h, w = ann.shape[:2]

    # Draw virtual entry/exit lines
    def _pt(norm):
        return (int(norm[0] * w), int(norm[1] * h))

    cv2.line(ann, _pt(entry_line[0]), _pt(entry_line[1]), (0, 255, 0), 2)
    cv2.putText(ann, "ENTRY", _pt(entry_line[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.line(ann, _pt(exit_line[0]), _pt(exit_line[1]), (0, 0, 255), 2)
    cv2.putText(ann, "EXIT", _pt(exit_line[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Draw unique visitor counter
    cv2.putText(
        ann, f"Unique visitors: {visitor_count}",
        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2,
    )
    cv2.putText(
        ann, f"Inside now: {len(inside_face_ids)}",
        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2,
    )

    for tr in tracks:
        face_id = track_to_face.get(tr.track_id, "")
        color = _color_for_face(face_id) if face_id else (128, 128, 128)
        x1, y1, x2, y2 = tr.bbox.as_int_tuple()
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
        label1 = f"T{tr.track_id}"
        label2 = (face_id[-8:] if len(face_id) > 8 else face_id) or "?"
        inside_mark = " [IN]" if face_id in inside_face_ids else ""
        cv2.putText(ann, label1 + inside_mark,
                    (x1, max(0, y1 - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(ann, label2,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return ann


# ---------------------------------------------------------------------------
# DB summary helpers
# ---------------------------------------------------------------------------

def _db_summary(conn) -> dict:
    from src.database.connection import connect
    try:
        persons   = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        embeds    = conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
        events    = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        entries   = conn.execute("SELECT COUNT(*) FROM events WHERE event_type='ENTRY'").fetchone()[0]
        exits     = conn.execute("SELECT COUNT(*) FROM events WHERE event_type='EXIT'").fetchone()[0]
        visits    = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        open_v    = conn.execute("SELECT COUNT(*) FROM visits WHERE status='open'").fetchone()[0]
        tot_vis   = conn.execute("SELECT SUM(total_visits) FROM persons").fetchone()[0] or 0
        return {
            "persons": persons,
            "embeddings": embeds,
            "events": events,
            "entries": entries,
            "exits": exits,
            "visits": visits,
            "open_visits": open_v,
            "total_visits_sum": tot_vis,
        }
    except Exception as exc:
        logger.warning("Could not query DB summary: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Padded crop helper (mirrors insightface_engine._crop_to_bbox)
# ---------------------------------------------------------------------------

def _padded_crop(image: np.ndarray, x1: float, y1: float, x2: float, y2: float,
                 pad_ratio: float = 0.3) -> np.ndarray:
    h, w = image.shape[:2]
    bw = x2 - x1; bh = y2 - y1
    cx1 = max(0, int(x1 - bw * pad_ratio))
    cy1 = max(0, int(y1 - bh * pad_ratio))
    cx2 = min(w, int(x2 + bw * pad_ratio))
    cy2 = min(h, int(y2 + bh * pad_ratio))
    if cx2 <= cx1 or cy2 <= cy1:
        return image.copy()
    return image[cy1:cy2, cx1:cx2].copy()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Intelligent Face Tracker — full end-to-end pipeline."
    )
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--video", default="",
                        help="Override input video path (overrides config input.video_path)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = full video)")
    parser.add_argument("--save-dir", default="",
                        help="Save annotated frames to this directory")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode — no GUI window")
    parser.add_argument("--db-path", default="",
                        help="Override SQLite database path (use :memory: for isolated demo)")
    args = parser.parse_args(argv)

    # ── Config ────────────────────────────────────────────────────────
    config = load_config(args.config)

    # Apply CLI overrides
    if args.video:
        from dataclasses import replace as _replace
        config = _replace(config, input=_replace(config.input, video_path=args.video))

    # ── Logging ───────────────────────────────────────────────────────
    warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
    event_log = setup_logging(config.logging)
    event_log.info('{"event":"SYSTEM","message":"pipeline starting","ts":"%s"}', utc_now_iso())
    logger.info("=== Intelligent Face Tracker starting ===")
    logger.info("Config: %s", config.config_path)
    logger.info("Source: %s  path=%s", config.input.source_type,
                config.input.video_path or config.input.rtsp_url)

    # ── Database ──────────────────────────────────────────────────────
    db_path_str = args.db_path if args.db_path else str(config.resolve_path(config.database.sqlite_path))
    conn = open_initialized(db_path_str, wal_mode=config.database.wal_mode)
    conn.commit()
    logger.info("Database: %s", db_path_str)

    # ── Input ─────────────────────────────────────────────────────────
    try:
        source = create_frame_source(config)
    except (VideoPathNotConfiguredError, VideoFileNotFoundError, RtspNotConfiguredError) as exc:
        logger.error("%s", exc)
        conn.close()
        return 2
    except VideoSourceError as exc:
        logger.error("Video source error: %s", exc)
        conn.close()
        return 2
    event_log.info('{"event":"SYSTEM","message":"video source opened","ts":"%s"}', utc_now_iso())

    # ── YOLO ──────────────────────────────────────────────────────────
    try:
        detector = create_face_detector(config)
    except ModelNotFoundError as exc:
        logger.error("%s", exc)
        conn.close()
        return 3
    except DetectionError as exc:
        logger.error("Detector error: %s", exc)
        conn.close()
        return 3

    # ── ByteTrack ─────────────────────────────────────────────────────
    try:
        tracker = create_face_tracker(config)
    except TrackingError as exc:
        logger.error("Tracker error: %s", exc)
        conn.close()
        return 4

    # ── InsightFace ───────────────────────────────────────────────────
    try:
        embedder = create_face_embedder(config)
    except ModelLoadError as exc:
        logger.error("InsightFace load error: %s", exc)
        conn.close()
        return 5
    except RecognitionError as exc:
        logger.error("Recognition error: %s", exc)
        conn.close()
        return 5

    matcher = create_gallery_matcher(config)
    registrar = FaceRegistrar(config.recognition, config.registration, matcher)

    # ── Entry/Exit FSM + Visit Manager ────────────────────────────────
    fsm = EntryExitFSM(config.entry_exit, track_buffer=config.tracking.track_buffer)
    vm  = VisitManager(conn, fsm, config.paths, source_label=config.input.source_type)

    skip = DetectionSkipPolicy(config.detection.detection_skip_frames)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Per-session state
    track_to_face:  dict[int, str]   = {}   # track_id → face_id
    inside_face_ids: set[str]        = set() # face_ids currently INSIDE
    registered_this_run: set[str]    = set()

    # Counters
    processed         = 0
    detect_frames     = 0
    total_detections  = 0
    total_embeddings  = 0
    total_new         = 0
    total_reid        = 0
    unique_visitor_count = 0  # backed by DB persons.total_visits aggregate

    event_log.info('{"event":"SYSTEM","message":"main loop started","ts":"%s"}', utc_now_iso())

    try:
        for frame in source:
            fh, fw = frame.image.shape[:2]

            # ── Detection + Tracking ──────────────────────────────────
            if skip.should_detect(frame.index):
                dets = list(detector.detect(frame))
                total_detections += len(dets)
                detect_frames += 1
                if config.logging.log_detections:
                    logger.debug("Frame %d: %d detections", frame.index, len(dets))
                tracks = list(
                    tracker.update(dets, frame_index=frame.index, image=frame.image)
                )
            else:
                tracks = list(
                    tracker.update(None, frame_index=frame.index, image=frame.image)
                )

            current_frame_face_ids: set[str] = set()

            # ── Embedding + Registration (detection frames only) ──────
            if skip.should_detect(frame.index):
                for tr in tracks:
                    bbox = (
                        float(tr.bbox.x1), float(tr.bbox.y1),
                        float(tr.bbox.x2), float(tr.bbox.y2),
                    )
                    emb = embedder.embed(frame.image, bbox=bbox)
                    if emb is None:
                        continue
                    total_embeddings += 1

                    crop = _padded_crop(
                        frame.image,
                        tr.bbox.x1, tr.bbox.y1, tr.bbox.x2, tr.bbox.y2,
                    )

                    identity = registrar.identify_or_register(
                        conn, emb,
                        crop_image=crop,
                        source_label=f"frame_{frame.index}",
                    )
                    conn.commit()

                    if not identity.face_id:
                        continue

                    track_to_face[tr.track_id] = identity.face_id

                    if identity.is_new:
                        registered_this_run.add(identity.face_id)
                        total_new += 1
                        unique_visitor_count += 1
                        event_log.info(
                            '{"event":"REGISTER","face_id":"%s","track_id":%d,'
                            '"frame":%d,"ts":"%s"}',
                            identity.face_id, tr.track_id, frame.index, utc_now_iso(),
                        )
                        logger.info(
                            "REGISTER  face_id=%s  track_id=%d  frame=%d",
                            identity.face_id, tr.track_id, frame.index,
                        )
                    else:
                        total_reid += 1
                        event_log.info(
                            '{"event":"RECOGNIZE","face_id":"%s","track_id":%d,'
                            '"sim":%.4f,"frame":%d,"ts":"%s"}',
                            identity.face_id, tr.track_id,
                            identity.similarity, frame.index, utc_now_iso(),
                        )

            # ── Entry/Exit FSM ────────────────────────────────────────
            for tr in tracks:
                face_id = track_to_face.get(tr.track_id, "")
                if not face_id:
                    continue

                current_frame_face_ids.add(face_id)
                crop = _padded_crop(
                    frame.image,
                    tr.bbox.x1, tr.bbox.y1, tr.bbox.x2, tr.bbox.y2,
                )
                fsm_events = fsm.update(
                    face_id, tr.track_id,
                    tr.bbox.x1, tr.bbox.y1, tr.bbox.x2, tr.bbox.y2,
                    fw, fh,
                    crop, frame.index,
                    quality=tr.confidence,
                )
                vm.handle_all(fsm_events)

                # Track inside set
                from src.pipeline.entry_exit import VisitState
                if fsm.get_state(face_id) == VisitState.INSIDE:
                    inside_face_ids.add(face_id)
                else:
                    inside_face_ids.discard(face_id)

            # ── Track-lost flush ──────────────────────────────────────
            lost_events = fsm.flush_lost_tracks(frame.index)
            if lost_events:
                vm.handle_all(lost_events)
                for lev in lost_events:
                    inside_face_ids.discard(lev.face_id)

            # ── Draw + save/display ───────────────────────────────────
            ann = _draw_frame(
                frame.image, tracks, track_to_face, inside_face_ids,
                entry_line=list(config.entry_exit.entry_line),
                exit_line=list(config.entry_exit.exit_line),
                visitor_count=unique_visitor_count,
            )
            if save_dir:
                cv2.imwrite(str(save_dir / f"frame_{frame.index:06d}.jpg"), ann)
            if not args.no_display:
                cv2.imshow("Face Tracker", ann)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User quit at frame %d", frame.index)
                    break

            processed += 1
            if args.max_frames > 0 and processed >= args.max_frames:
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down")
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()

        # Flush remaining inside faces as TRACK_LOST exits
        if processed > 0:
            final_lost = fsm.flush_lost_tracks(processed + config.tracking.track_buffer + 1)
            if final_lost:
                vm.handle_all(final_lost)

    # ── Final DB summary ──────────────────────────────────────────────
    conn.commit()
    summary = _db_summary(conn)
    conn.close()

    # Recompute unique visitors from DB (authoritative)
    unique_visitors_db = summary.get("persons", unique_visitor_count)

    # ── Print summary ─────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  INTELLIGENT FACE TRACKER — RUN SUMMARY")
    print(sep)
    print(f"  Frames processed      : {processed}")
    print(f"  Detection frames      : {detect_frames}")
    print(f"  Total YOLO detections : {total_detections}")
    print(f"  Embeddings extracted  : {total_embeddings}")
    print(f"  New registrations     : {total_new}")
    print(f"  Re-identifications    : {total_reid}")
    print(f"  Total entries (ENTRY) : {vm.summary.total_entries}")
    print(f"  Total exits   (EXIT)  : {vm.summary.total_exits}")
    print(f"  Unique visitors (DB)  : {unique_visitors_db}")
    print(f"  Total visits (visits) : {summary.get('visits', '?')}")
    print(f"  Open visits remaining : {summary.get('open_visits', '?')}")
    print(sep)
    print(f"  Database              : {db_path_str}")
    print(f"  Events log            : {config.resolve_path(config.logging.file_path)}")
    print(f"  Event snapshots       : {config.resolve_path(config.paths.events_snapshot_dir)}")
    if save_dir:
        print(f"  Annotated frames      : {save_dir}")
    print(f"{sep}\n")

    event_log.info(
        '{"event":"SYSTEM","message":"pipeline finished","frames":%d,'
        '"unique_visitors":%d,"entries":%d,"exits":%d,"ts":"%s"}',
        processed,
        unique_visitors_db,
        vm.summary.total_entries,
        vm.summary.total_exits,
        utc_now_iso(),
    )
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
