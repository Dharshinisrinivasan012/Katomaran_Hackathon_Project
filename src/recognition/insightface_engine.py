"""InsightFace ArcFace embedding engine (buffalo_l, 512-dim).

Model details (verified):
  Package : insightface 2.0
  Bundle  : buffalo_l  (downloaded to ~/.insightface/models/buffalo_l/)
  Detector: SCRFD (bundled — used only to locate face landmarks for alignment)
  Recognizer: ArcFaceONNX  — input 3×112×112, output (1, 512) float32
  Embedding dim : 512
  Post-processing: L2 normalization applied by this engine

InsightFace auto-downloads the buffalo_l model bundle on first use.
The cache lives in ~/.insightface/models/ — not in this repository.

This class does NOT use:
  - the face_recognition library
  - dlib
  - any custom embedding model

IMPORTANT — track_id vs face_id:
  This engine is concerned with embeddings only.  It has no knowledge of
  ByteTrack track_ids.  The caller (recognizer/registrar) is responsible
  for mapping track_id → face_id via the gallery.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from typing import Any

import numpy as np

from src.config.loader import RecognitionConfig
from src.recognition.exceptions import (
    InvalidImageError,
    ModelLoadError,
    RecognitionBackendError,
)
from src.recognition.results import EmbeddingResult
from src.recognition.similarity import normalize_embedding

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 512        # ArcFaceONNX output for buffalo_l, verified


def _compute_hash(embedding: np.ndarray) -> str:
    """SHA-256 of the normalized embedding bytes for deduplication."""
    return hashlib.sha256(embedding.tobytes()).hexdigest()


def _validate_image(image: np.ndarray, min_size_px: int) -> None:
    if image is None:
        raise InvalidImageError("Image is None")
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"Expected np.ndarray, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(
            f"Expected 3-channel BGR image, got shape {image.shape}"
        )
    h, w = image.shape[:2]
    if h < min_size_px or w < min_size_px:
        raise InvalidImageError(
            f"Image too small ({w}×{h}); minimum is {min_size_px}×{min_size_px} px"
        )
    if image.dtype != np.uint8:
        raise InvalidImageError(
            f"Expected uint8 image, got dtype {image.dtype}"
        )


class InsightFaceEngine:
    """InsightFace buffalo_l wrapper — embedding extraction only.

    Usage::

        engine = InsightFaceEngine(config.recognition)
        result = engine.embed(bgr_image)
        if result is not None:
            # result.embedding: (512,) float32 L2-normalized
            # result.embedding_hash: sha256 hex string
            # result.quality_score: InsightFace det_score
    """

    # Model name used in the face_embeddings table — matches config.
    MODEL_NAME = "buffalo_l"
    EMBEDDING_DIM = _EMBEDDING_DIM

    def __init__(self, config: RecognitionConfig) -> None:
        self._config = config
        self.embedding_dim = _EMBEDDING_DIM
        self.model_name = self.MODEL_NAME
        self._app: Any = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ModelLoadError(
                "insightface is required for recognition. "
                "Install from requirements.txt."
            ) from exc

        providers = list(self._config.providers) or ["CPUExecutionProvider"]
        try:
            # Suppress the FutureWarning from insightface.utils.face_align
            # about SimilarityTransform.estimate — that is a library internals
            # issue unrelated to our usage.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                app = FaceAnalysis(name="buffalo_l", providers=providers)
                app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Failed to load InsightFace buffalo_l: {exc}"
            ) from exc

        if "recognition" not in app.models:
            raise RecognitionBackendError(
                "buffalo_l bundle is missing the 'recognition' model. "
                "Delete ~/.insightface/models/buffalo_l and re-run to re-download."
            )
        self._app = app
        logger.info(
            "InsightFace buffalo_l loaded (providers=%s, embedding_dim=%d)",
            providers,
            _EMBEDDING_DIM,
        )

    def embed(
        self,
        image: np.ndarray,
        *,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> EmbeddingResult | None:
        """Extract an L2-normalized 512-dim embedding from a BGR image.

        Args:
            image: Full video frame or a pre-cropped face region (BGR uint8).
            bbox:  Optional (x1,y1,x2,y2) hint.  When provided, the image is
                   cropped to that region before calling InsightFace, which
                   avoids running the SCRFD detector on the full 4K frame.

        Returns:
            EmbeddingResult with normalized embedding and quality_score,
            or None when InsightFace finds no face in the region.
        """
        _validate_image(image, self._config.min_face_size_px)

        # Optionally crop to bbox before passing to InsightFace
        roi = _crop_to_bbox(image, bbox) if bbox is not None else image

        # Validate the crop is still large enough
        if roi.shape[0] < self._config.min_face_size_px or roi.shape[1] < self._config.min_face_size_px:
            logger.debug(
                "Crop too small after bbox application (%dx%d), skipping",
                roi.shape[1],
                roi.shape[0],
            )
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            faces = self._app.get(roi)

        if not faces:
            logger.debug("InsightFace found no face in image region (%dx%d)", roi.shape[1], roi.shape[0])
            return None

        # Pick the highest-quality face detection in the region
        best = max(faces, key=lambda f: float(f.det_score))

        if best.embedding is None:
            logger.debug("InsightFace returned a face with no embedding, skipping")
            return None

        # InsightFace already normalizes the embedding, but we re-normalize
        # defensively to guarantee exact unit norm.
        normed = normalize_embedding(best.embedding.astype(np.float32))
        emb_hash = _compute_hash(normed)
        quality = float(best.det_score) if best.det_score is not None else None

        if quality is not None and quality < self._config.quality_min_score:
            logger.debug(
                "Face quality %.3f below threshold %.3f, skipping",
                quality,
                self._config.quality_min_score,
            )
            return None

        return EmbeddingResult(
            embedding=normed,
            embedding_hash=emb_hash,
            quality_score=quality,
        )


def _crop_to_bbox(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    pad_ratio: float = 0.3,
) -> np.ndarray:
    """Return a copy of the image cropped to bbox (x1,y1,x2,y2) with padding.

    InsightFace's internal SCRFD detector needs context around the face to
    detect landmarks for alignment.  A tight YOLO crop (no context) typically
    yields 0 faces from SCRFD.  Adding ~30% padding on each side resolves this.

    Args:
        bbox:      (x1, y1, x2, y2) in pixel coordinates.
        pad_ratio: Fraction of box size to add as padding on each side.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))
    if cx2 <= cx1 or cy2 <= cy1:
        return image.copy()
    return image[cy1:cy2, cx1:cx2].copy()
