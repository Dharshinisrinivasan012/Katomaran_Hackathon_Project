"""Cosine similarity utilities and gallery matcher.

Metric: cosine similarity on L2-normalized vectors.
  sim(a, b) = dot(a/||a||, b/||b||)

For pre-normalized vectors this reduces to a simple dot product, which is
the standard approach for ArcFace / InsightFace embeddings.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from src.recognition.results import GalleryMatch

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two arbitrary float vectors.

    Returns a value in [−1.0, 1.0].  If either vector is zero, returns 0.0.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return an L2-normalized copy.  Zero vectors are returned unchanged."""
    norm = float(np.linalg.norm(embedding))
    if norm < 1e-10:
        return embedding.copy().astype(np.float32)
    return (embedding / norm).astype(np.float32)


class CosineMatcher:
    """Gallery search using cosine similarity.

    When a face_id has multiple stored embeddings, the maximum similarity
    across all its embeddings is used (best-of-K matching).
    """

    def __init__(self, similarity_threshold: float) -> None:
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {similarity_threshold}"
            )
        self.similarity_threshold = similarity_threshold

    def search(
        self,
        query: np.ndarray,
        gallery: Sequence[tuple[str, np.ndarray]],
    ) -> GalleryMatch | None:
        """Return best match above threshold or None.

        Args:
            query:   L2-normalized float32, shape (D,).
            gallery: Sequence of (face_id, normalized_embedding) pairs.
        """
        if not gallery:
            return None

        # Aggregate max similarity per face_id
        best_per_face: dict[str, float] = {}
        for face_id, stored in gallery:
            sim = cosine_similarity(query, stored)
            if face_id not in best_per_face or sim > best_per_face[face_id]:
                best_per_face[face_id] = sim

        if not best_per_face:
            return None

        best_id = max(best_per_face, key=lambda k: best_per_face[k])
        best_sim = best_per_face[best_id]

        logger.debug(
            "Gallery search: best_face_id=%s sim=%.4f threshold=%.4f",
            best_id,
            best_sim,
            self.similarity_threshold,
        )

        if best_sim >= self.similarity_threshold:
            return GalleryMatch(face_id=best_id, similarity=best_sim)
        return None
