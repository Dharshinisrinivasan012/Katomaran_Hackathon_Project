"""Auto-registration of unknown faces and re-identification of known ones.

This module bridges the recognition layer and the database layer.

Responsibilities:
  1. Accept an EmbeddingResult from InsightFaceEngine.
  2. Load the current gallery from SQLite.
  3. Search the gallery via CosineMatcher.
  4. If a match is found → return the existing face_id (re-identification).
  5. If no match → register a new person:
       a. Generate a UUID-based face_id.
       b. INSERT into persons.
       c. INSERT into face_embeddings (with deduplication via embedding_hash).
       d. Optionally save the face crop to data/faces/{face_id}/.
       e. Log a REGISTER event.
  6. Return an IdentityResult describing what happened.

CRITICAL identity rules (documented here for every caller to read):
  - track_id  is a TEMPORARY ByteTrack motion identity.
              It changes when a person leaves and re-enters the frame.
  - face_id   is the PERSISTENT SQLite identity assigned here.
              It must survive track_id changes across frames and sessions.
  - Never use track_id as a face_id.
  - Never create a new face_id because a track_id changed.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path

import cv2
import numpy as np

from src.config.loader import PROJECT_ROOT, RecognitionConfig, RegistrationConfig
from src.database import (
    DuplicateEmbeddingError,
    add_embedding,
    create_event,
    list_embeddings_for_gallery,
    register_person,
    touch_person_seen,
)
from src.recognition.results import EmbeddingResult, IdentityResult
from src.recognition.similarity import CosineMatcher
from src.registration.exceptions import CropSaveError

logger = logging.getLogger(__name__)

_FACE_ID_PREFIX = "FT-"


def _new_face_id() -> str:
    """Generate a unique persistent face ID: FT-<uuid4-hex>."""
    return f"{_FACE_ID_PREFIX}{uuid.uuid4().hex}"


def _resolve_dir(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


class FaceRegistrar:
    """Register new faces and re-identify known ones against the SQLite gallery.

    All database access is done through the repository functions — no raw SQL here.
    """

    def __init__(
        self,
        rec_config: RecognitionConfig,
        reg_config: RegistrationConfig,
        matcher: CosineMatcher,
    ) -> None:
        self._rec_cfg = rec_config
        self._reg_cfg = reg_config
        self._matcher = matcher

    def identify_or_register(
        self,
        conn: sqlite3.Connection,
        embedding_result: EmbeddingResult,
        *,
        crop_image: np.ndarray | None = None,
        source_label: str = "video",
    ) -> IdentityResult:
        """Main entry point: match gallery or register a new person.

        Args:
            conn:             Open SQLite connection (caller owns transaction).
            embedding_result: Normalized embedding + quality from InsightFaceEngine.
            crop_image:       Optional BGR crop of the face; saved to filesystem if
                              registration occurs and crop_save_enabled is True.
            source_label:     Human-readable source tag for logging (e.g. 'video').

        Returns:
            IdentityResult with face_id, is_new flag, similarity, and embedding.
        """
        gallery = self._load_gallery(conn)
        match = self._matcher.search(embedding_result.embedding, gallery)

        if match is not None:
            # Known person — update last_seen and return existing face_id
            touch_person_seen(conn, match.face_id)
            logger.debug(
                "Re-identified face_id=%s (sim=%.4f) from %s",
                match.face_id,
                match.similarity,
                source_label,
            )
            return IdentityResult(
                face_id=match.face_id,
                is_new=False,
                similarity=match.similarity,
                embedding=embedding_result.embedding,
            )

        # Unknown face — register if policy allows
        if not self._reg_cfg.auto_register_unknown:
            logger.debug(
                "No gallery match and auto_register_unknown=False; returning unknown"
            )
            # Return a placeholder that won't be persisted — caller may choose to skip
            return IdentityResult(
                face_id="",
                is_new=False,
                similarity=0.0,
                embedding=embedding_result.embedding,
            )

        face_id = self._register_new(
            conn,
            embedding_result,
            crop_image=crop_image,
            source_label=source_label,
        )
        return IdentityResult(
            face_id=face_id,
            is_new=True,
            similarity=0.0,
            embedding=embedding_result.embedding,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_gallery(
        self, conn: sqlite3.Connection
    ) -> list[tuple[str, np.ndarray]]:
        """Load all embeddings from SQLite as (face_id, ndarray) pairs."""
        rows = list_embeddings_for_gallery(conn)
        gallery: list[tuple[str, np.ndarray]] = []
        for row in rows:
            vec = np.frombuffer(row.embedding, dtype=np.float32).copy()
            if vec.shape[0] == self._rec_cfg.embedding_dim:
                gallery.append((row.face_id, vec))
            else:
                logger.warning(
                    "Gallery embedding for face_id=%s has wrong dimension %d (expected %d), skipping",
                    row.face_id,
                    vec.shape[0],
                    self._rec_cfg.embedding_dim,
                )
        return gallery

    def _register_new(
        self,
        conn: sqlite3.Connection,
        embedding_result: EmbeddingResult,
        *,
        crop_image: np.ndarray | None,
        source_label: str,
    ) -> str:
        face_id = _new_face_id()
        register_person(conn, face_id)

        # Save crop to filesystem first so we have the path for the DB row
        crop_path: str | None = None
        if crop_image is not None and self._reg_cfg.crop_save_enabled:
            crop_path = self._save_crop(face_id, crop_image)

        # Store embedding — DuplicateEmbeddingError shouldn't happen on a brand-new
        # face_id, but if two frames race (future pipeline) we silently ignore it.
        try:
            add_embedding(
                conn,
                face_id,
                model_name=InsightFaceEngine.MODEL_NAME,
                dimension=int(embedding_result.embedding.shape[0]),
                embedding=embedding_result.embedding.tobytes(),
                embedding_hash=embedding_result.embedding_hash,
                quality_score=embedding_result.quality_score,
                source_image_path=crop_path,
            )
        except DuplicateEmbeddingError:
            logger.warning(
                "Duplicate embedding hash on new face_id=%s — embedding not stored again",
                face_id,
            )

        # REGISTER event
        create_event(
            conn,
            "REGISTER",
            face_id=face_id,
            confidence=embedding_result.quality_score,
            image_path=crop_path,
            metadata={"source": source_label},
        )

        logger.info("Registered new face_id=%s (source=%s)", face_id, source_label)
        return face_id

    def _save_crop(self, face_id: str, crop: np.ndarray) -> str | None:
        """Write crop to data/faces/{face_id}/crop_0.jpg. Returns relative path."""
        try:
            dest_dir = _resolve_dir(self._reg_cfg.crop_dir) / face_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / "crop_0.jpg"
            ok = cv2.imwrite(str(dest), crop)
            if not ok:
                raise CropSaveError(f"cv2.imwrite returned False for {dest}")
            rel = str(dest.relative_to(PROJECT_ROOT))
            logger.debug("Saved face crop to %s", rel)
            return rel
        except CropSaveError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save face crop for %s: %s", face_id, exc)
            return None


# Deferred import inside class method to avoid circular at module level
from src.recognition.insightface_engine import InsightFaceEngine  # noqa: E402
