"""Auto-registration of unknown faces and re-identification of known persons.

Track ID → Face ID mapping rule (mandatory for correct visitor counting):
  track_id  = temporary ByteTrack motion identity (resets on re-entry)
  face_id   = persistent SQLite identity (survives track resets)
"""

from src.registration.exceptions import (
    CropSaveError,
    QualityGateError,
    RegistrationError,
)
from src.registration.registrar import FaceRegistrar

__all__ = [
    "CropSaveError",
    "FaceRegistrar",
    "QualityGateError",
    "RegistrationError",
]
