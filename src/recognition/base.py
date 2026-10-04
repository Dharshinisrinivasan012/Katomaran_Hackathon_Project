"""Protocol interfaces for recognition and gallery matching.

Callers (pipeline, preview) depend only on these protocols — not on InsightFace
directly.  Swap the backend without touching anything outside src/recognition/.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from src.recognition.results import EmbeddingResult, GalleryMatch


class FaceEmbedder(Protocol):
    """Extract a normalized embedding from a BGR face image."""

    embedding_dim: int
    model_name: str

    def embed(
        self,
        image: np.ndarray,
        *,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> EmbeddingResult | None:
        """Return an embedding or None when no usable face is found.

        Args:
            image: BGR uint8 ndarray — can be a full frame or a pre-cropped face.
            bbox:  Optional (x1, y1, x2, y2) in pixel coords to hint which region
                   contains the face.  When None the embedder searches the whole image.
        """


class GalleryMatcher(Protocol):
    """Search a set of stored embeddings for the nearest match."""

    similarity_threshold: float

    def search(
        self,
        query: np.ndarray,
        gallery: Sequence[tuple[str, np.ndarray]],
    ) -> GalleryMatch | None:
        """Return the best match above threshold, or None.

        Args:
            query:   L2-normalized float32 embedding, shape (D,).
            gallery: Sequence of (face_id, normalized_embedding) pairs.
                     Multiple embeddings per face_id are allowed; the best
                     similarity per face_id is used.
        """
