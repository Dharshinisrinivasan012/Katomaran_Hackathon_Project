"""Registration module exceptions."""


class RegistrationError(Exception):
    """Base error for the registration module."""


class QualityGateError(RegistrationError):
    """Raised when a candidate crop fails quality gates."""


class CropSaveError(RegistrationError):
    """Raised when a face crop cannot be written to the filesystem."""
