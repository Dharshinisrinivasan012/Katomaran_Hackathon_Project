"""Face detection errors."""


class DetectionError(Exception):
    """Base error for the detector module."""


class DetectorBackendError(DetectionError):
    """Raised when the configured detection backend cannot be used."""


class ModelNotFoundError(DetectionError):
    """Raised when detection.model_path does not exist on disk."""


class ModelLoadError(DetectionError):
    """Raised when YOLO weights cannot be loaded."""
