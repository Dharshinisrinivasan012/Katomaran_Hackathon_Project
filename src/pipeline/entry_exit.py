"""Entry/Exit state machine — line-crossing mode.

Operates on PERSISTENT face_ids (not temporary ByteTrack track_ids).

Entry/exit geometry (from config.json):
  entry_line: [[x1,y1],[x2,y2]] normalized [0,1] coordinates
  exit_line:  [[x1,y1],[x2,y2]] normalized [0,1] coordinates
  direction_entry: "top_to_bottom"  (signed-distance sign change negative→positive)
  direction_exit:  "bottom_to_top"  (signed-distance sign change positive→negative)
  debounce_sec:    minimum seconds between successive events for the same face_id
  min_track_age_frames: ignore tracks younger than this

Algorithm (per frame, per face_id that has a resolved identity):
  1. Compute bbox foot-point (bottom-center) in normalized [0,1] pixel coords.
  2. Compute signed distance of foot-point from entry_line and exit_line using
     the 2D cross-product.  Sign tells which side of the line the point is on.
  3. When the sign changes (previous sign recorded) and direction matches config:
       - ENTRY: signed distance flips negative→positive (top_to_bottom crossing)
       - EXIT:  signed distance flips positive→negative (bottom_to_top crossing)
  4. Gate by debounce_sec and min_track_age_frames.
  5. For faces that were Inside but whose track disappears for more than
     track_lost_timeout_frames: auto-issue EXIT (TRACK_LOST path).

Face state machine per face_id:
  Outside → (ENTRY crossing) → Inside → (EXIT crossing) → Outside
                                       → (track lost timeout) → Outside (auto EXIT)

CRITICAL: face_id ≠ track_id.
  - A new ByteTrack track_id for a returning face (after re-ID) must not create a
    new visit if the same face_id already has an open visit.
  - This FSM maintains state keyed by face_id, not track_id.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from src.config.loader import EntryExitConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


class _Point(NamedTuple):
    x: float  # normalized [0,1]
    y: float  # normalized [0,1]


def _signed_distance(point: _Point, p1: _Point, p2: _Point) -> float:
    """2D cross product of (p2-p1) × (point-p1).

    Positive = point is to the LEFT of the directed segment p1→p2 (or BELOW
    a horizontal line going left→right).
    Negative = point is to the RIGHT (or ABOVE a horizontal line).
    Zero = point is on the line.
    """
    return (p2.x - p1.x) * (point.y - p1.y) - (p2.y - p1.y) * (point.x - p1.x)


def _foot_point_normalized(bbox_x1: float, bbox_y1: float, bbox_x2: float, bbox_y2: float,
                           frame_w: int, frame_h: int) -> _Point:
    """Return bottom-centre of the bbox in normalized [0,1] coordinates."""
    cx = (bbox_x1 + bbox_x2) / 2.0 / frame_w
    cy = bbox_y2 / frame_h
    return _Point(x=min(1.0, max(0.0, cx)), y=min(1.0, max(0.0, cy)))


# ---------------------------------------------------------------------------
# Per-face tracking state
# ---------------------------------------------------------------------------


class VisitState(Enum):
    OUTSIDE = auto()
    INSIDE  = auto()


@dataclass
class _FaceState:
    """Runtime state for one face_id across frames."""
    visit_state:        VisitState = VisitState.OUTSIDE
    visit_id:           int | None = None          # open visit's DB id
    entry_sign:         float | None = None        # last signed-dist from entry_line
    exit_sign:          float | None = None        # last signed-dist from exit_line
    last_event_ts:      float = 0.0                # monotonic, for debounce
    last_seen_frame:    int   = -1                 # for track-lost detection
    track_age_frames:   int   = 0                  # frames this face has been seen
    best_crop:          np.ndarray | None = None   # best quality crop seen while inside
    best_quality:       float = 0.0
    current_track_id:   int | None = None


# ---------------------------------------------------------------------------
# Event result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryExitEvent:
    """Emitted by the FSM; caller persists to DB and filesystem."""
    event_type:  str           # "ENTRY" or "EXIT"
    face_id:     str
    track_id:    int | None
    crop:        np.ndarray | None
    timestamp:   str           # ISO-8601 UTC string (caller provides)
    frame_index: int


# ---------------------------------------------------------------------------
# Main FSM
# ---------------------------------------------------------------------------


class EntryExitFSM:
    """Line-crossing entry/exit state machine keyed by persistent face_id.

    Usage::

        fsm = EntryExitFSM(config.entry_exit, track_buffer=config.tracking.track_buffer)
        events = fsm.update(face_id, track_id, bbox, frame_w, frame_h, crop, frame_index)
        for ev in events:
            # persist ev to DB and filesystem

        # After the main loop, flush any faces still inside whose tracks are lost:
        lost_events = fsm.flush_lost_tracks(current_frame_index)
    """

    def __init__(self, config: EntryExitConfig, *, track_buffer: int = 30) -> None:
        self._cfg = config
        self._track_buffer = track_buffer
        # Parse line geometry once
        e = config.entry_line
        x = config.exit_line
        self._ep1 = _Point(*e[0])
        self._ep2 = _Point(*e[1])
        self._xp1 = _Point(*x[0])
        self._xp2 = _Point(*x[1])
        # State per face_id
        self._states: dict[str, _FaceState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        face_id: str,
        track_id: int,
        bbox_x1: float,
        bbox_y1: float,
        bbox_x2: float,
        bbox_y2: float,
        frame_w: int,
        frame_h: int,
        crop: np.ndarray | None,
        frame_index: int,
        *,
        quality: float = 0.0,
    ) -> list[EntryExitEvent]:
        """Process one face observation and return any triggered events.

        Args:
            face_id:    Persistent identity from InsightFace gallery.
            track_id:   Current ByteTrack id (transient; used for metadata only).
            bbox_*:     Bounding box in pixel coordinates.
            frame_w/h:  Frame dimensions for normalization.
            crop:       Face crop image for saving with ENTRY/EXIT event.
            frame_index: Current frame number.
            quality:    Detection confidence for selecting the best crop.
        """
        if not face_id:
            return []

        st = self._get_or_create(face_id)
        st.last_seen_frame = frame_index
        st.current_track_id = track_id
        st.track_age_frames += 1

        # Update best crop for this face (used for ENTRY/EXIT snapshots)
        if crop is not None and quality >= st.best_quality:
            st.best_crop = crop.copy()
            st.best_quality = quality

        # Don't evaluate crossing for brand-new tracks
        if st.track_age_frames < self._cfg.min_track_age_frames:
            return []

        foot = _foot_point_normalized(bbox_x1, bbox_y1, bbox_x2, bbox_y2, frame_w, frame_h)
        events: list[EntryExitEvent] = []

        entry_sd = _signed_distance(foot, self._ep1, self._ep2)
        exit_sd  = _signed_distance(foot, self._xp1, self._xp2)

        now = time.monotonic()

        if st.visit_state == VisitState.OUTSIDE:
            # Check for ENTRY crossing
            if self._crosses_entry(st.entry_sign, entry_sd):
                if now - st.last_event_ts >= self._cfg.debounce_sec:
                    events.append(EntryExitEvent(
                        event_type="ENTRY",
                        face_id=face_id,
                        track_id=track_id,
                        crop=st.best_crop,
                        timestamp="",          # caller fills ISO timestamp
                        frame_index=frame_index,
                    ))
                    st.visit_state = VisitState.INSIDE
                    st.last_event_ts = now
                    st.best_crop = crop.copy() if crop is not None else None
                    st.best_quality = quality
                    logger.debug("face_id=%s ENTRY at frame %d", face_id, frame_index)

            # First appearance already inside the entry zone (prev_sign was None):
            # treat as implicit entry — face appeared without us seeing the crossing.
            elif st.entry_sign is None and entry_sd > 0:
                events.append(EntryExitEvent(
                    event_type="ENTRY",
                    face_id=face_id,
                    track_id=track_id,
                    crop=st.best_crop,
                    timestamp="",
                    frame_index=frame_index,
                ))
                st.visit_state = VisitState.INSIDE
                st.last_event_ts = now
                st.best_crop = crop.copy() if crop is not None else None
                st.best_quality = quality
                logger.debug("face_id=%s implicit ENTRY (first sighting inside zone) frame %d", face_id, frame_index)

        elif st.visit_state == VisitState.INSIDE:
            # Check for EXIT crossing
            if self._crosses_exit(st.exit_sign, exit_sd):
                if now - st.last_event_ts >= self._cfg.debounce_sec:
                    events.append(EntryExitEvent(
                        event_type="EXIT",
                        face_id=face_id,
                        track_id=track_id,
                        crop=st.best_crop,
                        timestamp="",
                        frame_index=frame_index,
                    ))
                    st.visit_state = VisitState.OUTSIDE
                    st.last_event_ts = now
                    st.best_crop = None
                    st.best_quality = 0.0
                    logger.debug("face_id=%s EXIT at frame %d", face_id, frame_index)

        # Update sign memory after crossing check
        st.entry_sign = entry_sd
        st.exit_sign  = exit_sd

        return events

    def flush_lost_tracks(
        self,
        current_frame_index: int,
    ) -> list[EntryExitEvent]:
        """Issue EXIT for faces that are INSIDE but whose track has been lost.

        Called once per frame (or at end of processing).  A track is considered
        lost when it hasn't been seen for more than ``track_buffer`` frames.
        """
        events: list[EntryExitEvent] = []
        for face_id, st in self._states.items():
            if st.visit_state != VisitState.INSIDE:
                continue
            frames_since = current_frame_index - st.last_seen_frame
            if frames_since > self._track_buffer:
                events.append(EntryExitEvent(
                    event_type="EXIT",
                    face_id=face_id,
                    track_id=st.current_track_id,
                    crop=st.best_crop,
                    timestamp="",
                    frame_index=current_frame_index,
                ))
                st.visit_state = VisitState.OUTSIDE
                st.last_event_ts = time.monotonic()
                st.best_crop = None
                st.best_quality = 0.0
                logger.debug(
                    "face_id=%s track-lost EXIT at frame %d (unseen for %d frames)",
                    face_id, current_frame_index, frames_since,
                )
        return events

    def get_state(self, face_id: str) -> VisitState:
        st = self._states.get(face_id)
        return st.visit_state if st else VisitState.OUTSIDE

    def set_visit_id(self, face_id: str, visit_id: int) -> None:
        """Store the DB visit_id after open_visit() returns."""
        st = self._states.get(face_id)
        if st:
            st.visit_id = visit_id

    def get_visit_id(self, face_id: str) -> int | None:
        st = self._states.get(face_id)
        return st.visit_id if st else None

    def reset_face(self, face_id: str) -> None:
        """Remove state for a face_id (e.g. on explicit reset)."""
        self._states.pop(face_id, None)

    def reset_all(self) -> None:
        self._states.clear()

    # ------------------------------------------------------------------
    # Direction helpers — only line_crossing mode implemented
    # ------------------------------------------------------------------

    def _crosses_entry(self, prev_sign: float | None, curr_sign: float) -> bool:
        if prev_sign is None:
            return False
        direction = self._cfg.direction_entry
        if direction == "top_to_bottom":
            # foot moves from above line (negative sd) to below (positive sd)
            return prev_sign < 0 and curr_sign >= 0
        if direction == "bottom_to_top":
            return prev_sign > 0 and curr_sign <= 0
        # Fallback: any crossing
        return (prev_sign < 0 and curr_sign >= 0) or (prev_sign > 0 and curr_sign <= 0)

    def _crosses_exit(self, prev_sign: float | None, curr_sign: float) -> bool:
        if prev_sign is None:
            return False
        direction = self._cfg.direction_exit
        if direction == "bottom_to_top":
            return prev_sign > 0 and curr_sign <= 0
        if direction == "top_to_bottom":
            return prev_sign < 0 and curr_sign >= 0
        return (prev_sign < 0 and curr_sign >= 0) or (prev_sign > 0 and curr_sign <= 0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create(self, face_id: str) -> _FaceState:
        if face_id not in self._states:
            self._states[face_id] = _FaceState()
        return self._states[face_id]
