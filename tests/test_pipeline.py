"""Tests for EntryExitFSM, VisitManager, and related pipeline components.

Unit tests use synthetic data — no real video, YOLO, or InsightFace required.

FSM geometry used in these tests (hardcoded, independent of config.json):
  entry_line: y=0.5  (horizontal across full width)
  exit_line:  y=0.7  (horizontal across full width)
  direction_entry: top_to_bottom
  direction_exit:  bottom_to_top
  debounce_sec: 0 (overridden in tests that need it)
  min_track_age_frames: 3 (fast enough for 30-frame tests)
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config.loader import EntryExitConfig, load_config
from src.database.connection import open_initialized
from src.database.repositories import (
    get_open_visit,
    get_person_by_face_id,
    register_person,
)
from src.pipeline.entry_exit import (
    EntryExitFSM,
    VisitState,
    _foot_point_normalized,
    _signed_distance,
    _Point,
)
from src.pipeline.visit_manager import VisitManager


# ---------------------------------------------------------------------------
# Fixed test geometry (independent of config.json's calibrated lines)
# ---------------------------------------------------------------------------

def _test_entry_exit_config() -> EntryExitConfig:
    """Return a stable test config with entry line at y=0.5, exit at y=0.7."""
    return EntryExitConfig(
        mode="line_crossing",
        entry_line=[(0.0, 0.5), (1.0, 0.5)],
        exit_line=[(0.0, 0.7), (1.0, 0.7)],
        direction_entry="top_to_bottom",
        direction_exit="bottom_to_top",
        debounce_sec=0.0,
        min_track_age_frames=3,
    )


def _fsm_for_test() -> EntryExitFSM:
    return EntryExitFSM(_test_entry_exit_config(), track_buffer=30)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn(tmp_path: Path) -> sqlite3.Connection:
    conn = open_initialized(tmp_path / "test_pipeline.db", wal_mode=False)
    yield conn
    conn.close()


@pytest.fixture
def fsm() -> EntryExitFSM:
    return _fsm_for_test()


def _registered_face(conn: sqlite3.Connection, face_id: str = "FT-test001") -> str:
    register_person(conn, face_id)
    conn.commit()
    return face_id


def _blank_crop() -> np.ndarray:
    return np.zeros((80, 60, 3), dtype=np.uint8)


def _zero_debounce(cfg: EntryExitConfig) -> EntryExitConfig:
    return replace(cfg, debounce_sec=0.0)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def test_signed_distance_below_horizontal_line() -> None:
    p1 = _Point(0.0, 0.5)
    p2 = _Point(1.0, 0.5)
    assert _signed_distance(_Point(0.5, 0.7), p1, p2) > 0   # below → positive
    assert _signed_distance(_Point(0.5, 0.3), p1, p2) < 0   # above → negative


def test_foot_point_normalized_center() -> None:
    pt = _foot_point_normalized(280, 200, 360, 300, 640, 480)
    assert abs(pt.x - 0.5) < 0.02
    assert abs(pt.y - 300 / 480) < 0.02


def test_foot_point_clamped() -> None:
    pt = _foot_point_normalized(-10, -10, 650, 490, 640, 480)
    assert 0.0 <= pt.x <= 1.0
    assert 0.0 <= pt.y <= 1.0


# ---------------------------------------------------------------------------
# EntryExitFSM — initial state and basic guards
# ---------------------------------------------------------------------------

def test_fsm_initial_state_outside(fsm: EntryExitFSM) -> None:
    assert fsm.get_state("FT-new") == VisitState.OUTSIDE


def test_fsm_no_event_before_min_track_age() -> None:
    """Faces below entry line must not fire before min_track_age_frames is reached.

    With min_track_age_frames=3, frames 0 and 1 (track_age 1 and 2) must be silent.
    Frame 2 (track_age 3) is when events first become eligible.
    """
    f = EntryExitFSM(_test_entry_exit_config(), track_buffer=30)
    # Bbox at y=600–700 in a 1000px frame → foot_y=0.70 > entry_line_y=0.5 → inside zone
    for i in range(2):          # frames 0 and 1 → track_age 1 and 2
        evs = f.update(
            "FT-x", 1,
            400, 600, 600, 700,   # below entry line
            1000, 1000,
            _blank_crop(), i,
        )
        assert evs == [], f"Expected no events at frame {i} (track_age={i+1} < 3)"


def test_fsm_implicit_entry_on_first_sighting_inside_zone() -> None:
    """Face that first appears below the entry line triggers implicit ENTRY at min_track_age."""
    f = _fsm_for_test()
    # Warm up frames 0 and 1 (below entry line, track_age < 3)
    for i in range(2):
        f.update("FT-implicit", 1, 400, 600, 600, 700, 1000, 1000, _blank_crop(), i)
    # Frame 2 → track_age reaches 3 → implicit ENTRY fires
    evs = f.update("FT-implicit", 1, 400, 600, 600, 700, 1000, 1000, _blank_crop(), 2)
    entry_evs = [e for e in evs if e.event_type == "ENTRY"]
    assert len(entry_evs) == 1
    assert f.get_state("FT-implicit") == VisitState.INSIDE


def test_fsm_reset_face_clears_state(fsm: EntryExitFSM) -> None:
    fsm._states["FT-ghost"] = MagicMock()
    fsm.reset_face("FT-ghost")
    assert "FT-ghost" not in fsm._states


def test_fsm_reset_all(fsm: EntryExitFSM) -> None:
    fsm._states["FT-a"] = MagicMock()
    fsm._states["FT-b"] = MagicMock()
    fsm.reset_all()
    assert len(fsm._states) == 0


def test_fsm_set_and_get_visit_id(fsm: EntryExitFSM) -> None:
    fsm._get_or_create("FT-v")
    fsm.set_visit_id("FT-v", 99)
    assert fsm.get_visit_id("FT-v") == 99


# ---------------------------------------------------------------------------
# EntryExitFSM — explicit line crossing ENTRY and EXIT
# ---------------------------------------------------------------------------

def _do_explicit_crossing(face_id: str) -> tuple[list, list, EntryExitFSM]:
    """
    Simulate a face:
      1. Starting ABOVE the entry line (y=0.3 → foot above 0.5).
      2. Warm up 3 frames to pass min_track_age_frames.
      3. Walk downward through the entry line (y goes 0.3 → 0.6).
      4. Walk upward through the exit line (y goes 0.75 → 0.65).

    Frame size: 1000×1000. Entry line y=0.5, exit line y=0.7.
    """
    f = _fsm_for_test()
    entry_events: list = []
    exit_events: list = []

    # Warm up: face ABOVE entry line (foot_y=0.3 → entry_sd < 0)
    for i in range(3):
        f.update(face_id, 1, 400, 200, 600, 300, 1000, 1000, _blank_crop(), i)

    # Walk downward: foot_y goes 0.30 → 0.40 → 0.49 → 0.51 → 0.60
    for step, fy in enumerate([0.30, 0.40, 0.49, 0.51, 0.60]):
        by2 = int(fy * 1000)
        evs = f.update(face_id, 1, 400, by2 - 50, 600, by2, 1000, 1000, _blank_crop(), 3 + step)
        entry_events.extend([e for e in evs if e.event_type == "ENTRY"])

    # Walk upward: foot_y goes 0.75 → 0.71 → 0.68 → 0.64
    for step, fy in enumerate([0.75, 0.71, 0.68, 0.64]):
        by2 = int(fy * 1000)
        evs = f.update(face_id, 1, 400, by2 - 50, 600, by2, 1000, 1000, _blank_crop(), 8 + step)
        exit_events.extend([e for e in evs if e.event_type == "EXIT"])

    return entry_events, exit_events, f


def test_fsm_entry_crossing_produces_entry_event() -> None:
    entry_events, _, _ = _do_explicit_crossing("FT-crossing")
    assert len(entry_events) == 1
    assert entry_events[0].event_type == "ENTRY"
    assert entry_events[0].face_id == "FT-crossing"


def test_fsm_exit_crossing_produces_exit_event() -> None:
    _, exit_events, _ = _do_explicit_crossing("FT-exiting")
    assert len(exit_events) == 1
    assert exit_events[0].event_type == "EXIT"


def test_fsm_state_transitions() -> None:
    face_id = "FT-state"
    _, _, f = _do_explicit_crossing(face_id)
    # After full crossing (ENTRY then EXIT), face should be OUTSIDE
    assert f.get_state(face_id) == VisitState.OUTSIDE


def test_fsm_no_duplicate_entry() -> None:
    """While INSIDE, re-crossing the entry line must not fire a second ENTRY."""
    face_id = "FT-dup-entry"
    f = _fsm_for_test()

    # Warm up above the line
    for i in range(3):
        f.update(face_id, 1, 400, 200, 600, 300, 1000, 1000, _blank_crop(), i)

    # Cross entry line (above → below)
    first_entries = []
    for step, fy in enumerate([0.40, 0.49, 0.51, 0.60]):
        by2 = int(fy * 1000)
        evs = f.update(face_id, 1, 400, by2 - 50, 600, by2, 1000, 1000, _blank_crop(), 3 + step)
        first_entries.extend([e for e in evs if e.event_type == "ENTRY"])

    assert len(first_entries) == 1, "Expected exactly one ENTRY on first crossing"
    assert f.get_state(face_id) == VisitState.INSIDE

    # Re-cross entry line while still INSIDE — must NOT fire again
    second_entries = []
    for step, fy in enumerate([0.40, 0.49, 0.51, 0.60]):
        by2 = int(fy * 1000)
        evs = f.update(face_id, 1, 400, by2 - 50, 600, by2, 1000, 1000, _blank_crop(), 20 + step)
        second_entries.extend([e for e in evs if e.event_type == "ENTRY"])

    assert second_entries == [], "Re-crossing entry while INSIDE must not produce another ENTRY"


def test_fsm_flush_lost_tracks() -> None:
    """Face inside whose track disappears should get EXIT via flush."""
    face_id = "FT-lost"
    f = _fsm_for_test()
    # Force INSIDE state directly
    st = f._get_or_create(face_id)
    st.visit_state = VisitState.INSIDE
    st.last_seen_frame = 5
    st.track_age_frames = 10
    st.current_track_id = 1

    lost_evs = f.flush_lost_tracks(5 + 31)
    assert any(e.event_type == "EXIT" and e.face_id == face_id for e in lost_evs)
    assert f.get_state(face_id) == VisitState.OUTSIDE


# ---------------------------------------------------------------------------
# VisitManager — DB integration
# ---------------------------------------------------------------------------

@pytest.fixture
def vm(db_conn: sqlite3.Connection, fsm: EntryExitFSM, valid_config_path: Path) -> VisitManager:
    config = load_config(valid_config_path)
    instance = VisitManager(db_conn, fsm, config.paths, source_label="test")
    instance.fsm = fsm
    return instance


def test_visit_manager_entry_creates_visit(
    db_conn: sqlite3.Connection, vm: VisitManager
) -> None:
    from src.pipeline.entry_exit import EntryExitEvent
    face_id = _registered_face(db_conn)
    vm.handle(EntryExitEvent("ENTRY", face_id, 1, _blank_crop(), "", 5))
    visit = get_open_visit(db_conn, face_id)
    assert visit is not None
    assert visit.status == "open"


def test_visit_manager_entry_increments_total_visits(
    db_conn: sqlite3.Connection, vm: VisitManager
) -> None:
    from src.pipeline.entry_exit import EntryExitEvent
    face_id = _registered_face(db_conn, "FT-tv001")
    vm.handle(EntryExitEvent("ENTRY", face_id, 1, _blank_crop(), "", 5))
    person = get_person_by_face_id(db_conn, face_id)
    assert person is not None
    assert person.total_visits == 1


def test_visit_manager_repeated_recognition_does_not_increment_visits(
    db_conn: sqlite3.Connection, vm: VisitManager
) -> None:
    from src.pipeline.entry_exit import EntryExitEvent
    face_id = _registered_face(db_conn, "FT-nodup")
    ev = EntryExitEvent("ENTRY", face_id, 1, _blank_crop(), "", 5)
    vm.handle(ev)
    vm.handle(ev)   # second ENTRY — suppressed by open-visit guard
    person = get_person_by_face_id(db_conn, face_id)
    assert person.total_visits == 1


def test_visit_manager_exit_closes_visit(
    db_conn: sqlite3.Connection, vm: VisitManager
) -> None:
    from src.pipeline.entry_exit import EntryExitEvent
    face_id = _registered_face(db_conn, "FT-exit001")
    vm.handle(EntryExitEvent("ENTRY", face_id, 1, _blank_crop(), "", 5))
    visit = get_open_visit(db_conn, face_id)
    assert visit is not None
    vm.fsm.set_visit_id(face_id, visit.visit_id)
    vm.handle(EntryExitEvent("EXIT", face_id, 1, _blank_crop(), "", 30))
    assert get_open_visit(db_conn, face_id) is None


def test_visit_manager_exit_without_entry_is_safe(
    db_conn: sqlite3.Connection, vm: VisitManager
) -> None:
    from src.pipeline.entry_exit import EntryExitEvent
    face_id = _registered_face(db_conn, "FT-noexit")
    vm.handle(EntryExitEvent("EXIT", face_id, 1, _blank_crop(), "", 10))


def test_visit_manager_second_visit_after_exit(
    db_conn: sqlite3.Connection, vm: VisitManager
) -> None:
    from src.pipeline.entry_exit import EntryExitEvent
    face_id = _registered_face(db_conn, "FT-return")
    vm.handle(EntryExitEvent("ENTRY", face_id, 1, _blank_crop(), "", 5))
    visit1 = get_open_visit(db_conn, face_id)
    vm.fsm.set_visit_id(face_id, visit1.visit_id)
    vm.handle(EntryExitEvent("EXIT", face_id, 1, _blank_crop(), "", 30))
    vm.fsm.reset_face(face_id)
    vm.handle(EntryExitEvent("ENTRY", face_id, 2, _blank_crop(), "", 100))
    person = get_person_by_face_id(db_conn, face_id)
    assert person.total_visits == 2


def test_visit_manager_unknown_face_id_is_safe(
    db_conn: sqlite3.Connection, vm: VisitManager
) -> None:
    from src.pipeline.entry_exit import EntryExitEvent
    vm.handle(EntryExitEvent("ENTRY", "FT-ghost-unknown", 99, None, "", 1))


def test_visit_summary_counts(
    db_conn: sqlite3.Connection, vm: VisitManager
) -> None:
    from src.pipeline.entry_exit import EntryExitEvent
    face_id = _registered_face(db_conn, "FT-summary")
    vm.handle(EntryExitEvent("ENTRY", face_id, 1, _blank_crop(), "", 1))
    visit = get_open_visit(db_conn, face_id)
    vm.fsm.set_visit_id(face_id, visit.visit_id)
    vm.handle(EntryExitEvent("EXIT", face_id, 1, _blank_crop(), "", 10))
    assert vm.summary.total_entries == 1
    assert vm.summary.total_exits == 1
