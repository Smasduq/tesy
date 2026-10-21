"""Local provider (Ollama).

This is the home for a model you fine-tuned yourself. OpenRouter does not
offer fine-tuning, so the usual route is:

1. fine-tune an open model for free (Unsloth notebooks on Colab, for example)
2. merge the LoRA adapter and save it as a GGUF
3. ``ollama create tesy ./tesy.gguf``
4. ``python main.py --provider local "hello"``

The model runs on your own machine, so it costs nothing and your data never
leaves it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import ChatRequest, ProviderError, StreamChunk
from .openrouter import parse_stream

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "tesy"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 300


@dataclass
class LocalSettings:
    """Connection settings for a local Ollama server."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    timeout: int = DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls) -> LocalSettings:
        api_key = (os.environ.get("OLLAMA_API_KEY") or "").strip() or None
        return cls(
            base_url=(os.environ.get("OLLAMA_BASE_URL") or "").strip()
            or DEFAULT_BASE_URL,
            model=(os.environ.get("OLLAMA_MODEL") or "").strip() or DEFAULT_MODEL,
            api_key=api_key,
        )


def list_models(base_url: str = DEFAULT_BASE_URL, timeout: int = 10) -> list[str]:
    """List the models the local server has available."""
    request = Request(
        f"{base_url.rstrip('/')}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise ProviderError(f"Ollama returned HTTP {error.code}: {details}") from error
    except URLError as error:
        raise ProviderError(f"Could not connect to Ollama at {base_url}: {error.reason}") from error

    return [model.get("name", "") for model in payload.get("models") or []]


class LocalProvider:
    """Chat backend backed by a local Ollama server."""

    name = "local"
    default_model = DEFAULT_MODEL
    model_env_vars = ("OLLAMA_MODEL",)
    default_max_tokens = DEFAULT_MAX_TOKENS

    def __init__(self, settings: LocalSettings) -> None:
        self.settings = settings

    @classmethod
    def from_env(cls) -> LocalProvider:
        return cls(LocalSettings.from_env())

    def resolve_model(self, requested: str | None) -> str:
        if requested:
            return requested
        for name in self.model_env_vars:
            value = (os.environ.get(name) or "").strip()
            if value:
                return value
        return self.default_model

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [message.to_dict() for message in request.messages],
            "max_tokens": request.max_tokens,
            "stream": True,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        http_request = Request(
            f"{self.settings.base_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(http_request, timeout=self.settings.timeout) as response:
                yield from parse_stream(response)
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Ollama returned HTTP {error.code}: {details}") from error
        except URLError as error:
            raise ProviderError(
                f"Could not connect to Ollama at {self.settings.base_url}: {error.reason}"
            ) from error
