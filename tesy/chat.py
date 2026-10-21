"""A conversation with Tesy, shared by the text and voice front ends."""

from __future__ import annotations

from typing import Iterator

from .providers.base import ChatRequest, Message

DEFAULT_HISTORY_LIMIT = 40


class Conversation:
    """Keeps the history and asks the provider for replies."""

    def __init__(
        self,
        provider,
        model: str,
        system_prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.history_limit = history_limit
        self.history: list[Message] = [Message(role="system", content=system_prompt)]

    def request(self) -> ChatRequest:
        return ChatRequest(
            messages=self.history[-self.history_limit :],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )

    def ask(self, user_text: str) -> ChatRequest:
        """Record what was said and return the request to answer it."""
        self.history.append(Message(role="user", content=user_text))
        return self.request()

    def remember_reply(self, reply: str) -> None:
        """Record the answer so it stays part of the conversation."""
        self.history.append(Message(role="assistant", content=reply))

    def reply(self, user_text: str) -> Iterator[str]:
        """Send a message and yield the reply as it is generated."""
        pieces: list[str] = []
        for chunk in self.provider.stream(self.ask(user_text)):
            if chunk.kind != "content":
                continue
            pieces.append(chunk.text)
            yield chunk.text
        self.remember_reply("".join(pieces))

    def reset(self) -> None:
        """Forget the conversation but keep the system prompt."""
        system = self.history[0] if self.history else None
        self.history = [system] if system else []
