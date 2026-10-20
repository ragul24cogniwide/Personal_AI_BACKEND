"""
LLM Provider Manager & Factory.
Dynamically resolves and caches active LLM provider instances.
"""

from typing import Dict, Optional, Type
from app.core.config import settings
from app.ai.exceptions import LLMConfigurationError
from app.ai.llm.base import LLMProvider
from app.ai.llm.gemini import GeminiProvider


class LLMManager:
    """Manages active LLM provider instances."""

    def __init__(self):
        self._registry: Dict[str, Type[LLMProvider]] = {
            "gemini": GeminiProvider,
        }
        self._instances: Dict[str, LLMProvider] = {}

    def register_provider(self, name: str, provider_cls: Type[LLMProvider]):
        """Allows registering custom LLM providers at runtime."""
        self._registry[name.lower()] = provider_cls

    def get_provider(self, name: Optional[str] = None) -> LLMProvider:
        """
        Resolves the requested or globally configured LLM provider.
        Defaults to settings.LLM_PROVIDER (e.g. 'gemini').
        """
        provider_name = (name or settings.LLM_PROVIDER).lower()
        if provider_name not in self._registry:
            raise LLMConfigurationError(
                f"Unsupported LLM provider '{provider_name}'. Supported: {list(self._registry.keys())}"
            )

        if provider_name not in self._instances:
            provider_cls = self._registry[provider_name]
            self._instances[provider_name] = provider_cls()

        return self._instances[provider_name]


# Global LLMManager singleton
llm_manager = LLMManager()
