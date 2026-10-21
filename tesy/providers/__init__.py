"""Chat providers available to Tesy.

Adding a backend means writing a module with a provider class and registering
it in ``PROVIDERS`` - nothing else in the app needs to change.
"""

from __future__ import annotations

from .base import ChatRequest, Message, Provider, ProviderError, StreamChunk
from .local import LocalProvider, LocalSettings
from .openrouter import OpenRouterProvider, OpenRouterSettings, list_models
from .snowflake import SnowflakeProvider, SnowflakeSettings

PROVIDERS: dict[str, type] = {
    OpenRouterProvider.name: OpenRouterProvider,
    SnowflakeProvider.name: SnowflakeProvider,
    LocalProvider.name: LocalProvider,
}

DEFAULT_PROVIDER = OpenRouterProvider.name


def get_provider(name: str) -> Provider:
    """Build the provider registered under ``name`` from the environment."""
    provider_class = PROVIDERS.get(name)
    if provider_class is None:
        known = ", ".join(sorted(PROVIDERS))
        raise ProviderError(f"Unknown provider '{name}'. Known providers: {known}")
    return provider_class.from_env()


__all__ = [
    "ChatRequest",
    "DEFAULT_PROVIDER",
    "LocalProvider",
    "LocalSettings",
    "Message",
    "OpenRouterProvider",
    "OpenRouterSettings",
    "PROVIDERS",
    "Provider",
    "ProviderError",
    "SnowflakeProvider",
    "SnowflakeSettings",
    "StreamChunk",
    "get_provider",
    "list_models",
]
