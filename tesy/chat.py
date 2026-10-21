"""A conversation with Tesy, shared by the text and voice front ends."""

from __future__ import annotations

from typing import Iterator

from .computer import ComputerAccess
from .providers.base import ChatRequest, Message, StreamChunk, ToolCall

DEFAULT_HISTORY_LIMIT = 40
DEFAULT_TOOL_ROUNDS = 6


class Conversation:
    """Keeps the history and asks the provider for replies.

    When ``tools`` is set, the provider can ask to read a file. The request
    is run locally and the result is fed back to the model, which then
    answers. This repeats for a few rounds at most, so a confused model
    cannot loop forever.
    """

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
        tools: ComputerAccess | None = None,
        tool_rounds: int = DEFAULT_TOOL_ROUNDS,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.history_limit = history_limit
        self.tools = tools
        self.tool_rounds = max(1, tool_rounds)
        self.history: list[Message] = [Message(role="system", content=system_prompt)]

    def request(self) -> ChatRequest:
        return ChatRequest(
            messages=self.window(),
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            tools=self.tools.schemas() if self.tools else None,
        )

    def window(self) -> list[Message]:
        """The part of the history that is sent to the model.

        The system prompt always stays, and a tool result is never sent
        without the assistant message that asked for it.
        """
        system = self.history[0] if self.history and self.history[0].role == "system" else None
        body = self.history[1:] if system else self.history[:]
        room = self.history_limit - (1 if system else 0)
        body = body[-room:] if room > 0 else []
        while body and body[0].role == "tool":
            body = body[1:]
        return ([system] if system else []) + body

    def ask(self, user_text: str) -> ChatRequest:
        """Record what was said and return the request to answer it."""
        self.history.append(Message(role="user", content=user_text))
        return self.request()

    def remember_reply(self, reply: str) -> None:
        """Record an answer that was handled outside :meth:`turn`."""
        self.history.append(Message(role="assistant", content=reply))

    def turn(self, user_text: str) -> Iterator[StreamChunk]:
        """Send a message and yield the whole answer, tool calls included.

        Content and reasoning chunks are yielded as they arrive. A finished
        tool request is run and the result is added to the history, then the
        model is asked again for the actual answer.
        """
        self.history.append(Message(role="user", content=user_text))

        for _ in range(self.tool_rounds):
            pieces: list[str] = []
            calls: list[ToolCall] = []

            for chunk in self.provider.stream(self.request()):
                if chunk.kind == "tool_call":
                    if chunk.tool_call is not None:
                        calls.append(chunk.tool_call)
                    continue
                if chunk.kind == "content":
                    pieces.append(chunk.text)
                yield chunk

            self.history.append(
                Message(role="assistant", content="".join(pieces), tool_calls=tuple(calls))
            )

            if not calls:
                return
            if self.tools is None:
                return

            for call in calls:
                result = self.tools.run(call.name, call.arguments)
                self.history.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

    def reply(self, user_text: str) -> Iterator[str]:
        """Send a message and yield only the words of the reply."""
        for chunk in self.turn(user_text):
            if chunk.kind == "content":
                yield chunk.text

    def reset(self) -> None:
        """Forget the conversation but keep the system prompt."""
        system = self.history[0] if self.history else None
        self.history = [system] if system else []
