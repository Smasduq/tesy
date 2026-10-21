"""OpenRouter provider.

OpenRouter exposes an OpenAI-compatible chat completions API, so the whole
conversation is sent as ``messages`` and the reply arrives as server-sent
events with incremental ``delta.content`` chunks.

Reasoning models (Nemotron, DeepSeek, ...) stream their thinking in
``delta.reasoning`` before the answer. Those chunks are surfaced separately so
they never end up in a spoken reply.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, replace
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import ChatRequest, ProviderError, StreamChunk, ToolCall

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "minimax/minimax-m3.1-flash:free"
DEFAULT_MAX_TOKENS = 1024
APP_TITLE = "Tesy"
DEFAULT_TIMEOUT = 120
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")


@dataclass
class OpenRouterSettings:
    """Connection settings for the OpenRouter API."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    referer: str | None = None
    app_title: str = APP_TITLE
    timeout: int = DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls) -> OpenRouterSettings:
        api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
        if not api_key:
            raise ProviderError(
                "OPENROUTER_API_KEY is not set. Create a key at "
                "https://openrouter.ai/keys and add it to your .env file "
                "(or export it in your shell)."
            )

        referer = (os.environ.get("OPENROUTER_HTTP_REFERER") or "").strip() or None
        app_title = (os.environ.get("OPENROUTER_APP_TITLE") or "").strip() or APP_TITLE

        return cls(
            api_key=api_key,
            base_url=(os.environ.get("OPENROUTER_BASE_URL") or "").strip()
            or DEFAULT_BASE_URL,
            referer=referer,
            app_title=app_title,
        )


def parse_stream(lines: Iterable[bytes | str]) -> Iterator[StreamChunk]:
    """Turn an OpenRouter SSE body into a stream of chunks.

    Tool calls arrive as fragments: the id and name come once, then the
    arguments are spread over several events. They are collected here and
    yielded as one complete ``tool_call`` chunk when the model finishes.
    """
    partial: dict[int, dict[str, str]] = {}

    def finished_tool_calls() -> Iterator[StreamChunk]:
        for index in sorted(partial):
            call = partial[index]
            if not call["name"]:
                continue
            yield StreamChunk(
                kind="tool_call",
                tool_call=ToolCall(
                    id=call["id"] or f"call_{index}",
                    name=call["name"],
                    arguments=call["arguments"] or "{}",
                ),
            )
        partial.clear()

    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line.startswith("data:"):
            continue

        data = line[len("data:") :].strip()
        if data == "[DONE]":
            yield from finished_tool_calls()
            return

        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        if "error" in event:
            raise ProviderError(f"OpenRouter error: {event['error']}")

        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                yield StreamChunk(text=reasoning, kind="reasoning")

            content = delta.get("content")
            if isinstance(content, str) and content:
                yield StreamChunk(text=content, kind="content")

            for call in delta.get("tool_calls") or []:
                index = call.get("index")
                if not isinstance(index, int):
                    index = 0
                entry = partial.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if isinstance(call.get("id"), str) and call["id"]:
                    entry["id"] = call["id"]
                function = call.get("function") or {}
                if isinstance(function.get("name"), str) and function["name"]:
                    entry["name"] = function["name"]
                if isinstance(function.get("arguments"), str):
                    entry["arguments"] += function["arguments"]

            if choice.get("finish_reason"):
                yield from finished_tool_calls()


def _tool_unsupported(error: ProviderError) -> bool:
    """Does this error mean the chosen model cannot use tools?"""
    message = str(error).lower()
    if "tool" not in message:
        return False
    return any(
        hint in message
        for hint in ("does not support", "not supported", "no endpoints", "unsupported")
    )


def list_models(base_url: str = DEFAULT_BASE_URL, timeout: int = 30) -> list[dict[str, Any]]:
    """List the models OpenRouter currently offers."""
    request = Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise ProviderError(f"OpenRouter returned HTTP {error.code}: {details}") from error
    except URLError as error:
        raise ProviderError(f"Could not connect to OpenRouter: {error.reason}") from error

    return list(payload.get("data") or [])


class OpenRouterProvider:
    """Chat backend backed by OpenRouter."""

    name = "openrouter"
    default_model = DEFAULT_MODEL
    model_env_vars = ("OPENROUTER_MODEL",)
    default_max_tokens = DEFAULT_MAX_TOKENS
    supports_tools = True

    def __init__(self, settings: OpenRouterSettings) -> None:
        self.settings = settings

    @classmethod
    def from_env(cls) -> OpenRouterProvider:
        return cls(OpenRouterSettings.from_env())

    def resolve_model(self, requested: str | None) -> str:
        if requested:
            return requested
        for name in self.model_env_vars:
            value = (os.environ.get(name) or "").strip()
            if value:
                return value
        return self.default_model

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Stream a reply, retrying without tools if the model refuses them."""
        yielded = False
        try:
            for chunk in self._stream(request):
                yielded = True
                yield chunk
        except ProviderError as error:
            if yielded or not request.tools or not _tool_unsupported(error):
                raise
            print(
                f"note: {request.model} does not support file tools; "
                "carrying on without them.",
                file=sys.stderr,
            )
            yield from self._stream(replace(request, tools=None))

    def _stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [message.to_dict() for message in request.messages],
            "max_tokens": request.max_tokens,
            "stream": True,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.reasoning_effort:
            payload["reasoning"] = {"effort": request.reasoning_effort}
        if request.tools:
            payload["tools"] = request.tools

        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Title": self.settings.app_title,
        }
        if self.settings.referer:
            headers["HTTP-Referer"] = self.settings.referer

        http_request = Request(
            f"{self.settings.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(http_request, timeout=self.settings.timeout) as response:
                yield from parse_stream(response)
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise ProviderError(f"OpenRouter returned HTTP {error.code}: {details}") from error
        except URLError as error:
            raise ProviderError(f"Could not connect to OpenRouter: {error.reason}") from error
