"""Face recognition: InsightFace buffalo_l embeddings + cosine gallery matching.

Key distinction:
  track_id  — temporary ByteTrack motion identity (changes on re-entry)
  face_id   — persistent SQLite identity assigned by this module via embeddings

Public API:
  create_face_embedder()  → InsightFaceEngine
  create_gallery_matcher() → CosineMatcher
"""

from src.recognition.exceptions import (
    InvalidImageError,
    ModelLoadError,
    NoFaceFoundError,
    RecognitionBackendError,
    RecognitionError,
)
from src.recognition.factory import create_face_embedder, create_gallery_matcher
from src.recognition.insightface_engine import InsightFaceEngine
from src.recognition.results import EmbeddingResult, GalleryMatch, IdentityResult
from src.recognition.similarity import CosineMatcher, cosine_similarity, normalize_embedding

__all__ = [
    "CosineMatcher",
    "EmbeddingResult",
    "GalleryMatch",
    "IdentityResult",
    "InsightFaceEngine",
    "InvalidImageError",
    "ModelLoadError",
    "NoFaceFoundError",
    "RecognitionBackendError",
    "RecognitionError",
    "cosine_similarity",
    "create_face_embedder",
    "create_gallery_matcher",
    "normalize_embedding",
]
