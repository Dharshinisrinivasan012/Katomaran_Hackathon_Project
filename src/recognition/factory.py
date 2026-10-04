"""Create recognition engine and matcher from config."""

from __future__ import annotations

from src.config.loader import AppConfig, RecognitionConfig
from src.recognition.insightface_engine import InsightFaceEngine
from src.recognition.similarity import CosineMatcher


def create_face_embedder(config: AppConfig | RecognitionConfig) -> InsightFaceEngine:
    """Return an InsightFace buffalo_l embedder.

    Callers use this factory so they do not import InsightFaceEngine directly.
    """
    rec_cfg = config.recognition if isinstance(config, AppConfig) else config
    return InsightFaceEngine(rec_cfg)


def create_gallery_matcher(config: AppConfig | RecognitionConfig) -> CosineMatcher:
    """Return a CosineMatcher configured with the project similarity threshold."""
    rec_cfg = config.recognition if isinstance(config, AppConfig) else config
    return CosineMatcher(rec_cfg.similarity_threshold)
