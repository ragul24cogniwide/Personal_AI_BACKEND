"""
AI and LLM layer exceptions.
"""


class AIException(Exception):
    """Base exception for AI/LLM operations."""
    pass


class LLMConfigurationError(AIException):
    """Raised when an LLM provider is misconfigured (e.g. missing API key)."""
    pass


class LLMAPIError(AIException):
    """Raised when the LLM provider API returns an error or failure."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class LLMTimeoutError(AIException):
    """Raised when an LLM call exceeds the configured timeout."""
    def __init__(self, message: str = "LLM request timed out"):
        super().__init__(message)
        self.status_code = 504


class LLMInvalidResponseError(AIException):
    """Raised when the LLM response is empty, blocked, or invalid."""
    def __init__(self, message: str = "Invalid or empty LLM response"):
        super().__init__(message)
        self.status_code = 502
