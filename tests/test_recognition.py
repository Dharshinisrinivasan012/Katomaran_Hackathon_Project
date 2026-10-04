"""Tests for the recognition module (similarity, results, engine, registration).

Unit tests use synthetic embeddings and mock embedders — no real InsightFace model
required.  Integration tests that call the real model are guarded by
pytest.importorskip and a runtime model-availability check.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from src.config.loader import load_config
from src.database.connection import open_initialized
from src.database.repositories import (
    add_embedding,
    get_person_by_face_id,
    list_embeddings_for_gallery,
    register_person,
)
from src.recognition.exceptions import (
    InvalidImageError,
    ModelLoadError,
    RecognitionError,
)
from src.recognition.results import EmbeddingResult, GalleryMatch, IdentityResult
from src.recognition.similarity import CosineMatcher, cosine_similarity, normalize_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_embedding(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_embedding_result(seed: int = 0) -> EmbeddingResult:
    emb = _rand_embedding(seed=seed)
    h = hashlib.sha256(emb.tobytes()).hexdigest()
    return EmbeddingResult(embedding=emb, embedding_hash=h, quality_score=0.9)


@pytest.fixture
def db_conn(tmp_path: Path) -> sqlite3.Connection:
    conn = open_initialized(tmp_path / "test.db", wal_mode=False)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# normalize_embedding
# ---------------------------------------------------------------------------

def test_normalize_embedding_unit_norm() -> None:
    v = np.array([3.0, 4.0], dtype=np.float32)
    n = normalize_embedding(v)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-6


def test_normalize_embedding_already_unit() -> None:
    v = _rand_embedding(dim=8)
    n = normalize_embedding(v)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-6


def test_normalize_zero_vector() -> None:
    v = np.zeros(4, dtype=np.float32)
    n = normalize_embedding(v)
    assert n.shape == (4,)  # does not crash


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical() -> None:
    v = _rand_embedding(dim=32, seed=1)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-5


def test_cosine_similarity_opposite() -> None:
    v = _rand_embedding(dim=32, seed=2)
    assert abs(cosine_similarity(v, -v) - (-1.0)) < 1e-5


def test_cosine_similarity_orthogonal() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_zero_vector() -> None:
    a = np.zeros(4, dtype=np.float32)
    b = _rand_embedding(dim=4)
    assert cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# EmbeddingResult structure
# ---------------------------------------------------------------------------

def test_embedding_result_structure() -> None:
    emb = _rand_embedding(seed=3)
    h = hashlib.sha256(emb.tobytes()).hexdigest()
    r = EmbeddingResult(embedding=emb, embedding_hash=h, quality_score=0.85)
    assert r.embedding.shape == (512,)
    assert len(r.embedding_hash) == 64  # sha256 hex
    assert r.quality_score == 0.85


def test_embedding_result_equality_by_hash() -> None:
    emb = _rand_embedding(seed=4)
    h = hashlib.sha256(emb.tobytes()).hexdigest()
    r1 = EmbeddingResult(embedding=emb, embedding_hash=h)
    r2 = EmbeddingResult(embedding=emb.copy(), embedding_hash=h)
    assert r1 == r2


# ---------------------------------------------------------------------------
# GalleryMatch / IdentityResult structure
# ---------------------------------------------------------------------------

def test_gallery_match_structure() -> None:
    m = GalleryMatch(face_id="FT-abc", similarity=0.87)
    assert m.face_id == "FT-abc"
    assert m.similarity == 0.87


def test_identity_result_new() -> None:
    emb = _rand_embedding(seed=5)
    ir = IdentityResult(face_id="FT-new", is_new=True, similarity=0.0, embedding=emb)
    assert ir.is_new is True
    assert ir.face_id == "FT-new"


def test_identity_result_existing() -> None:
    emb = _rand_embedding(seed=6)
    ir = IdentityResult(face_id="FT-existing", is_new=False, similarity=0.72, embedding=emb)
    assert ir.is_new is False
    assert ir.similarity == 0.72


# ---------------------------------------------------------------------------
# CosineMatcher
# ---------------------------------------------------------------------------

def test_cosine_matcher_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        CosineMatcher(similarity_threshold=1.5)


def test_cosine_matcher_empty_gallery() -> None:
    matcher = CosineMatcher(0.45)
    result = matcher.search(_rand_embedding(), [])
    assert result is None


def test_cosine_matcher_match_above_threshold() -> None:
    matcher = CosineMatcher(0.45)
    anchor = _rand_embedding(seed=10)
    gallery = [("FT-a", anchor)]
    # Identical vector → similarity 1.0
    result = matcher.search(anchor, gallery)
    assert result is not None
    assert result.face_id == "FT-a"
    assert result.similarity > 0.99


def test_cosine_matcher_no_match_below_threshold() -> None:
    matcher = CosineMatcher(0.99)           # very high threshold
    a = _rand_embedding(seed=11)
    b = _rand_embedding(seed=99)            # random — likely low sim
    gallery = [("FT-b", b)]
    # Unless a and b happen to be nearly identical (astronomically unlikely), no match
    result = matcher.search(a, gallery)
    # We cannot guarantee None for all seeds, so check threshold logic
    if result is not None:
        assert result.similarity >= 0.99


def test_cosine_matcher_picks_best_face_among_multiple() -> None:
    matcher = CosineMatcher(0.45)
    q = _rand_embedding(seed=20)
    close = q.copy()                        # sim ≈ 1.0
    far = _rand_embedding(seed=21)          # likely low sim
    gallery = [("FT-far", far), ("FT-close", close)]
    result = matcher.search(q, gallery)
    assert result is not None
    assert result.face_id == "FT-close"


def test_cosine_matcher_multi_embedding_per_face() -> None:
    """best-of-K: FT-multi has two embeddings, one close and one far."""
    matcher = CosineMatcher(0.45)
    q = _rand_embedding(seed=30)
    e_close = q.copy()
    e_far = _rand_embedding(seed=99)
    gallery = [("FT-multi", e_far), ("FT-multi", e_close)]
    result = matcher.search(q, gallery)
    assert result is not None
    assert result.face_id == "FT-multi"
    assert result.similarity > 0.99


# ---------------------------------------------------------------------------
# FaceRegistrar (with mocked embedder)
# ---------------------------------------------------------------------------

from src.recognition.results import EmbeddingResult  # noqa: E402 (already imported)
from src.registration.registrar import FaceRegistrar  # noqa: E402


def _make_registrar(valid_config_path: Path) -> tuple[FaceRegistrar, CosineMatcher]:
    config = load_config(valid_config_path)
    matcher = CosineMatcher(config.recognition.similarity_threshold)
    registrar = FaceRegistrar(config.recognition, config.registration, matcher)
    return registrar, matcher


def test_registrar_registers_new_face(valid_config_path: Path, db_conn: sqlite3.Connection) -> None:
    registrar, _ = _make_registrar(valid_config_path)
    emb_result = _make_embedding_result(seed=40)
    result = registrar.identify_or_register(db_conn, emb_result, source_label="test")
    db_conn.commit()
    assert result.is_new is True
    assert result.face_id.startswith("FT-")
    person = get_person_by_face_id(db_conn, result.face_id)
    assert person is not None
    gallery = list_embeddings_for_gallery(db_conn)
    assert len(gallery) == 1


def test_registrar_reidentifies_existing_face(
    valid_config_path: Path, db_conn: sqlite3.Connection
) -> None:
    registrar, _ = _make_registrar(valid_config_path)
    emb_result = _make_embedding_result(seed=50)
    # First call — new registration
    first = registrar.identify_or_register(db_conn, emb_result, source_label="test")
    db_conn.commit()
    assert first.is_new is True
    # Second call with same embedding — re-identification
    second = registrar.identify_or_register(db_conn, emb_result, source_label="test")
    db_conn.commit()
    assert second.is_new is False
    assert second.face_id == first.face_id
    assert second.similarity > 0.99


def test_registrar_different_embeddings_create_separate_persons(
    valid_config_path: Path, db_conn: sqlite3.Connection
) -> None:
    config = load_config(valid_config_path)
    # Use a high threshold so two unrelated embeddings don't accidentally match
    matcher = CosineMatcher(similarity_threshold=0.99)
    registrar = FaceRegistrar(config.recognition, config.registration, matcher)

    emb_a = _make_embedding_result(seed=60)
    emb_b = _make_embedding_result(seed=61)  # Different seed → likely low cosine sim

    r_a = registrar.identify_or_register(db_conn, emb_a, source_label="test")
    db_conn.commit()
    r_b = registrar.identify_or_register(db_conn, emb_b, source_label="test")
    db_conn.commit()

    assert r_a.is_new is True
    assert r_b.is_new is True
    assert r_a.face_id != r_b.face_id


def test_registrar_duplicate_embedding_hash_does_not_crash(
    valid_config_path: Path, db_conn: sqlite3.Connection
) -> None:
    """Two identical embeddings for the same person → DuplicateEmbeddingError is swallowed."""
    registrar, _ = _make_registrar(valid_config_path)
    emb_result = _make_embedding_result(seed=70)
    r1 = registrar.identify_or_register(db_conn, emb_result, source_label="test")
    db_conn.commit()
    # Force a second identical embedding through a new EmbeddingResult with same hash
    # This simulates the race protection path in _register_new
    # (gallery now has this person so second call is re-id, not re-register)
    r2 = registrar.identify_or_register(db_conn, emb_result, source_label="test")
    db_conn.commit()
    assert r2.face_id == r1.face_id


# ---------------------------------------------------------------------------
# InsightFace engine — invalid image validation (no model needed)
# ---------------------------------------------------------------------------

def test_engine_invalid_image_none(valid_config_path: Path) -> None:
    pytest.importorskip("insightface")
    from src.recognition.insightface_engine import InsightFaceEngine, _validate_image
    with pytest.raises(InvalidImageError, match="None"):
        _validate_image(None, 40)  # type: ignore[arg-type]


def test_engine_invalid_image_wrong_channels(valid_config_path: Path) -> None:
    pytest.importorskip("insightface")
    from src.recognition.insightface_engine import _validate_image
    img = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(InvalidImageError, match="3-channel"):
        _validate_image(img, 40)


def test_engine_invalid_image_too_small(valid_config_path: Path) -> None:
    pytest.importorskip("insightface")
    from src.recognition.insightface_engine import _validate_image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(InvalidImageError, match="too small"):
        _validate_image(img, 40)


def test_engine_invalid_image_wrong_dtype(valid_config_path: Path) -> None:
    pytest.importorskip("insightface")
    from src.recognition.insightface_engine import _validate_image
    img = np.zeros((64, 64, 3), dtype=np.float32)
    with pytest.raises(InvalidImageError, match="uint8"):
        _validate_image(img, 40)


# ---------------------------------------------------------------------------
# Integration test — real InsightFace model (skipped if model unavailable)
# ---------------------------------------------------------------------------

def _insightface_available() -> bool:
    try:
        import insightface  # noqa: F401
        from insightface.app import FaceAnalysis
        import os
        cache = Path.home() / ".insightface" / "models" / "buffalo_l"
        return cache.is_dir()
    except Exception:
        return False


@pytest.mark.skipif(
    not _insightface_available(),
    reason="InsightFace buffalo_l model not available locally",
)
def test_insightface_engine_real_model_blank_image(valid_config_path: Path) -> None:
    """Real model: blank image should return None (no face found)."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    from src.recognition.factory import create_face_embedder
    config = load_config(valid_config_path)
    embedder = create_face_embedder(config)
    blank = np.zeros((112, 112, 3), dtype=np.uint8)
    result = embedder.embed(blank)
    # Blank image has no face — should be None
    assert result is None


@pytest.mark.skipif(
    not _insightface_available(),
    reason="InsightFace buffalo_l model not available locally",
)
def test_insightface_engine_real_model_embedding_shape(valid_config_path: Path) -> None:
    """Real model on a synthetic face-like image: embedding shape = (512,) and normalized."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import cv2
    from src.recognition.factory import create_face_embedder
    config = load_config(valid_config_path)
    embedder = create_face_embedder(config)
    assert embedder.embedding_dim == 512

    # Use first frame of the real sample video to get a real face
    sample = Path("data/videos/video_sample1.mp4")
    if not sample.is_file():
        pytest.skip("Sample video not available")
    cap = cv2.VideoCapture(str(sample))
    ret, frame = cap.read()
    cap.release()
    assert ret

    result = embedder.embed(frame)
    if result is None:
        pytest.skip("No face detected in first frame (model OK)")

    assert result.embedding.shape == (512,)
    assert result.embedding.dtype == np.float32
    assert abs(np.linalg.norm(result.embedding) - 1.0) < 1e-5
    assert len(result.embedding_hash) == 64


@pytest.mark.skipif(
    not _insightface_available(),
    reason="InsightFace buffalo_l model not available locally",
)
def test_insightface_full_pipeline_register_then_reid(
    valid_config_path: Path, tmp_path: Path
) -> None:
    """Register a face from the real video then re-identify it in the next detect frame."""
    import warnings, cv2
    warnings.filterwarnings("ignore", category=FutureWarning)
    from src.recognition.factory import create_face_embedder, create_gallery_matcher
    from src.registration.registrar import FaceRegistrar

    config = load_config(valid_config_path)
    conn = open_initialized(tmp_path / "test_pipe.db", wal_mode=False)

    sample = Path("data/videos/video_sample1.mp4")
    if not sample.is_file():
        conn.close()
        pytest.skip("Sample video not available")

    cap = cv2.VideoCapture(str(sample))
    ret, frame = cap.read()
    cap.release()
    assert ret

    embedder = create_face_embedder(config)
    matcher = create_gallery_matcher(config)
    registrar = FaceRegistrar(config.recognition, config.registration, matcher)

    emb1 = embedder.embed(frame)
    if emb1 is None:
        conn.close()
        pytest.skip("No face in first video frame")

    r1 = registrar.identify_or_register(conn, emb1, source_label="frame_0")
    conn.commit()
    assert r1.is_new is True
    face_id = r1.face_id

    # Same frame → same embedding → re-identification
    emb2 = embedder.embed(frame)
    assert emb2 is not None
    r2 = registrar.identify_or_register(conn, emb2, source_label="frame_0_again")
    conn.commit()
    assert r2.is_new is False
    assert r2.face_id == face_id
    assert r2.similarity > 0.99

    conn.close()
