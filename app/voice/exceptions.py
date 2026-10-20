"""
Custom exceptions for the Voice subsystem.
"""

class VoiceException(Exception):
    """Base exception for all voice operations."""
    def __init__(self, message: str, provider: str = "voice", status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code


class STTException(VoiceException):
    """Exception raised during Speech-to-Text operations."""
    pass


class TTSException(VoiceException):
    """Exception raised during Text-to-Speech operations."""
    pass


class ProviderConfigurationError(VoiceException):
    """Raised when an API key or provider setting is missing or invalid."""
    pass
