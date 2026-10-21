"""Shared types for the chat provider layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal, Protocol, runtime_checkable

ChunkKind = Literal["content", "reasoning", "tool_call"]


class ProviderError(RuntimeError):
    """Raised when a provider cannot fulfil a request."""


@dataclass(frozen=True)
class ToolCall:
    """A tool the model asked to run.

    ``arguments`` stays raw JSON text, because that is what the model emits
    and what has to go back to it in the conversation.
    """

    id: str
    name: str
    arguments: str = "{}"


@dataclass(frozen=True)
class Message:
    """A single chat message.

    Most messages are just a role and some text. An assistant message can
    also carry ``tool_calls``; the matching results come back as messages
    with ``role="tool"`` and a ``tool_call_id``.
    """

    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        return data


@dataclass(frozen=True)
class StreamChunk:
    """A piece of a streamed reply.

    ``kind`` separates the answer itself from a model's intermediate
    reasoning, so a voice pipeline can speak only the answer. A finished
    tool request arrives as a single ``tool_call`` chunk.
    """

    text: str = ""
    kind: ChunkKind = "content"
    tool_call: ToolCall | None = None


@dataclass(frozen=True)
class ChatRequest:
    """Everything a provider needs to produce a completion."""

    messages: list[Message]
    model: str
    max_tokens: int = 1024
    temperature: float | None = None
    reasoning_effort: str | None = None
    tools: list[dict[str, Any]] | None = None


@runtime_checkable
class Provider(Protocol):
    """A chat backend that can stream a reply."""

    name: str
    default_model: str
    supports_tools: bool

    def resolve_model(self, requested: str | None) -> str:
        """Pick the model to use, preferring an explicit request."""
        ...

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Yield reply chunks as they arrive."""
        ...
