"""Tracking errors."""


class TrackingError(Exception):
    """Base error for the tracker module."""


class TrackerBackendError(TrackingError):
    """Raised when the configured tracker cannot be constructed."""
