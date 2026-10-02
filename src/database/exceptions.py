"""Database layer exceptions."""


class DatabaseError(Exception):
    """Base error for database operations."""


class DuplicateEmbeddingError(DatabaseError):
    """Raised when embedding_hash violates UNIQUE constraint."""


class PersonNotFoundError(DatabaseError):
    """Raised when a face_id does not exist."""


class VisitNotFoundError(DatabaseError):
    """Raised when a visit_id does not exist or is not in expected state."""
