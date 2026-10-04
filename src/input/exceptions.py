"""Video input errors."""


class VideoSourceError(Exception):
    """Base error for video file and RTSP sources."""


class VideoPathNotConfiguredError(VideoSourceError):
    """Raised when source_type is file but input.video_path is empty."""


class VideoFileNotFoundError(VideoSourceError):
    """Raised when the configured local video path does not exist."""


class VideoOpenError(VideoSourceError):
    """Raised when OpenCV cannot open the configured source."""


class RtspNotConfiguredError(VideoSourceError):
    """Raised when source_type is rtsp but input.rtsp_url is empty or a placeholder."""


class RtspConnectionError(VideoSourceError):
    """Raised when an RTSP stream cannot be opened or reconnected."""


class VideoReadError(VideoSourceError):
    """Raised when a frame cannot be read from an open source."""
