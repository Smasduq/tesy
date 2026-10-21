"""Snowflake Cortex provider.

Kept around so Tesy can switch back to Snowflake with a single flag once
access is available again. The Cortex messages endpoint speaks the Anthropic
message format, which differs from OpenRouter's OpenAI-compatible one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import ChatRequest, ProviderError, StreamChunk

API_PATH = "/api/v2/cortex/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 120


@dataclass
class SnowflakeSettings:
    """Connection settings for Snowflake Cortex."""

    account_url: str
    token: str
    timeout: int = DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls) -> SnowflakeSettings:
        account_url = (os.environ.get("SNOWFLAKE_ACCOUNT_URL") or "").strip()
        if not account_url:
            account = (os.environ.get("account") or os.environ.get("ACCOUNT") or "").strip()
            if account:
                account_url = (
                    account
                    if "://" in account
                    else f"https://{account}.snowflakecomputing.com"
                )

        token = (
            (os.environ.get("SNOWFLAKE_PAT") or "").strip()
            or (os.environ.get("api_key") or "").strip()
            or (os.environ.get("API_KEY") or "").strip()
        )

        if not account_url or not token:
            raise ProviderError(
                "set SNOWFLAKE_ACCOUNT_URL (or account) and SNOWFLAKE_PAT "
                "(or api_key) in the environment or .env file"
            )

        return cls(account_url=account_url, token=token)


def parse_stream(lines: Iterable[bytes | str]) -> Iterator[StreamChunk]:
    """Turn a Snowflake Cortex SSE body into a stream of chunks."""
    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line.startswith("data:"):
            continue

        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return

        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        if "error" in event:
            raise ProviderError(f"Snowflake API error: {event['error']}")

        delta = event.get("delta", {})
        text = delta.get("text")
        if event.get("type") == "content_block_delta" and isinstance(text, str):
            yield StreamChunk(text=text, kind="content")


class SnowflakeProvider:
    """Chat backend backed by Snowflake Cortex."""

    name = "snowflake"
    default_model = DEFAULT_MODEL
    model_env_vars = ("SNOWFLAKE_MODEL",)
    default_max_tokens = DEFAULT_MAX_TOKENS
    supports_tools = False

    def __init__(self, settings: SnowflakeSettings) -> None:
        self.settings = settings

    @classmethod
    def from_env(cls) -> SnowflakeProvider:
        return cls(SnowflakeSettings.from_env())

    def resolve_model(self, requested: str | None) -> str:
        if requested:
            return requested
        for name in self.model_env_vars:
            value = (os.environ.get(name) or "").strip()
            if value:
                return value
        return self.default_model

    def stream(self, request: ChatRequest) -> Iterator[str]:
        account_url = self.settings.account_url.strip()
        if "://" not in account_url:
            account_url = f"https://{account_url}"
        url = f"{account_url.rstrip('/')}{API_PATH}"

        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [message.to_dict() for message in request.messages],
            "stream": True,
        }

        http_request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urlopen(http_request, timeout=self.settings.timeout) as response:
                yield from parse_stream(response)
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Snowflake returned HTTP {error.code}: {details}") from error
        except URLError as error:
            raise ProviderError(f"Could not connect to Snowflake: {error.reason}") from error
