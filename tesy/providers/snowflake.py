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

from .base import ChatRequest, Message, ProviderError, StreamChunk, ToolCall

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


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite OpenAI-style tool schemas the way Cortex expects them."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(function, dict) or not function.get("name"):
            continue
        converted.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "input_schema": function.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return converted


def _to_anthropic_message(message: Message) -> dict[str, Any]:
    """Turn one conversation message into an Anthropic-style entry.

    Assistant tool calls become ``tool_use`` blocks, and a tool result
    becomes a user message holding a ``tool_result`` block. That is how the
    Anthropic format pairs a request with its result.
    """
    if message.role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": message.content,
                }
            ],
        }

    if message.role == "assistant" and message.tool_calls:
        blocks: list[dict[str, Any]] = []
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        for call in message.tool_calls:
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            blocks.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": arguments}
            )
        return {"role": "assistant", "content": blocks}

    return {"role": message.role, "content": message.content}


def build_payload(request: ChatRequest) -> dict[str, Any]:
    """Turn a chat request into the Anthropic-style Cortex payload.

    Cortex takes the system prompt as a top-level ``system`` field and
    rejects ``system`` messages inside ``messages``.
    """
    system_parts = [
        message.content
        for message in request.messages
        if message.role == "system" and message.content
    ]
    messages = [
        _to_anthropic_message(message)
        for message in request.messages
        if message.role != "system"
    ]

    payload: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": messages,
        "stream": True,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.tools:
        converted = _openai_tools_to_anthropic(request.tools)
        if converted:
            payload["tools"] = converted
    return payload


def parse_stream(lines: Iterable[bytes | str]) -> Iterator[StreamChunk]:
    """Turn a Snowflake Cortex SSE body into a stream of chunks.

    Text arrives as ``content_block_delta`` events. A tool call arrives as a
    ``tool_use`` block whose arguments stream in as ``input_json_delta``
    fragments, so they are collected and yielded as one finished chunk.
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
            raise ProviderError(f"Snowflake API error: {event['error']}")

        kind = event.get("type")

        if kind == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                index = event.get("index")
                if not isinstance(index, int):
                    index = 0
                partial[index] = {
                    "id": block.get("id") or "",
                    "name": block.get("name") or "",
                    "arguments": "",
                }
            continue

        if kind == "content_block_stop":
            yield from finished_tool_calls()
            continue

        if kind != "content_block_delta":
            continue

        delta = event.get("delta") or {}

        if delta.get("type") == "input_json_delta":
            index = event.get("index")
            if not isinstance(index, int):
                index = 0
            entry = partial.setdefault(index, {"id": "", "name": "", "arguments": ""})
            fragment = delta.get("partial_json")
            if isinstance(fragment, str):
                entry["arguments"] += fragment
            continue

        text = delta.get("text")
        if isinstance(text, str):
            yield StreamChunk(text=text, kind="content")


class SnowflakeProvider:
    """Chat backend backed by Snowflake Cortex."""

    name = "snowflake"
    default_model = DEFAULT_MODEL
    model_env_vars = ("SNOWFLAKE_MODEL",)
    default_max_tokens = DEFAULT_MAX_TOKENS
    supports_tools = True

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

        payload = build_payload(request)

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
