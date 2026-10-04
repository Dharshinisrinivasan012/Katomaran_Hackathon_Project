"""Structured recognition results.

A RecognitionResult decouples the recognition layer from ByteTrack internals:
callers receive a face_id (or None) and an embedding — they do not need to know
anything about the InsightFace or ArcFace implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbeddingResult:
    """A single face embedding extracted from a crop.

    embedding: L2-normalized float32 ndarray of shape (512,) for buffalo_l.
    embedding_hash: SHA-256 hex digest of the normalized embedding bytes
                    — used for deduplication in face_embeddings table.
    quality_score:  0.0–1.0; InsightFace det_score for the face in the crop,
                    or None when the caller provides a pre-cropped image.
    """

    embedding: np.ndarray          # shape (D,) float32, L2-normalized
    embedding_hash: str            # sha256 of embedding.tobytes()
    quality_score: float | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EmbeddingResult):
            return NotImplemented
        return (
            self.embedding_hash == other.embedding_hash
            and self.quality_score == other.quality_score
            and np.array_equal(self.embedding, other.embedding)
        )

    def __hash__(self) -> int:  # required because we defined __eq__
        return hash(self.embedding_hash)


@dataclass(frozen=True)
class GalleryMatch:
    """Result of a gallery cosine-similarity search.

    face_id:    Existing persistent identity from SQLite.
    similarity: Cosine similarity in [−1, 1]; typical match is > threshold.
    """

    face_id: str
    similarity: float


@dataclass(frozen=True)
class IdentityResult:
    """Combined output of embed → search → register/match.

    face_id:       Persistent identity (UUID-based).
                   Assigned from the gallery if similarity ≥ threshold,
                   otherwise newly registered.
    is_new:        True  → person was auto-registered now.
                   False → existing person re-identified.
    similarity:    Cosine similarity of the best gallery match, or 0.0 for
                   brand-new registrations with an empty gallery.
    embedding:     The normalized embedding used for this decision.
    """

    face_id: str
    is_new: bool
    similarity: float
    embedding: np.ndarray          # shape (D,) float32, L2-normalized
