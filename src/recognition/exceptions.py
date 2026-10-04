"""Recognition module exceptions."""


class RecognitionError(Exception):
    """Base error for the recognition module."""


class RecognitionBackendError(RecognitionError):
    """Raised when the configured recognition backend cannot be used."""


class ModelLoadError(RecognitionError):
    """Raised when the InsightFace model package fails to load."""


class InvalidImageError(RecognitionError):
    """Raised when an image is null, wrong dtype, or too small to process."""


class NoFaceFoundError(RecognitionError):
    """Raised (optionally) when InsightFace detects no face in a crop."""
