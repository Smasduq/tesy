"""Shared types for the chat provider layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Protocol, runtime_checkable

ChunkKind = Literal["content", "reasoning"]


class ProviderError(RuntimeError):
    """Raised when a provider cannot fulfil a request."""


@dataclass(frozen=True)
class Message:
    """A single chat message."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class StreamChunk:
    """A piece of a streamed reply.

    ``kind`` separates the answer itself from a model's intermediate
    reasoning, so a voice pipeline can speak only the answer.
    """

    text: str
    kind: ChunkKind = "content"


@dataclass(frozen=True)
class ChatRequest:
    """Everything a provider needs to produce a completion."""

    messages: list[Message]
    model: str
    max_tokens: int = 1024
    temperature: float | None = None
    reasoning_effort: str | None = None


@runtime_checkable
class Provider(Protocol):
    """A chat backend that can stream a reply."""

    name: str
    default_model: str

    def resolve_model(self, requested: str | None) -> str:
        """Pick the model to use, preferring an explicit request."""
        ...

    def stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Yield reply chunks as they arrive."""
        ...
