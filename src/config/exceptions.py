"""Configuration-related exceptions."""


class ConfigError(Exception):
    """Base error for configuration loading."""


class ConfigValidationError(ConfigError):
    """Raised when config.json fails validation."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        self.path = path
        if path:
            super().__init__(f"{message} (at `{path}`)")
        else:
            super().__init__(message)
