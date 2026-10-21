"""Tests for the conversation loop and its tool round trip."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tesy.chat import Conversation  # noqa: E402
from tesy.providers.base import Message, StreamChunk, ToolCall  # noqa: E402


class ScriptedProvider:
    supports_tools = True

    def __init__(self, rounds):
        self.rounds = [list(round_) for round_ in rounds]
        self.requests = []

    def stream(self, request):
        self.requests.append(request)
        if not self.rounds:
            return
        yield from self.rounds.pop(0)


class FakeComputer:
    def __init__(self, answer="the file says hello"):
        self.answer = answer
        self.calls = []

    def schemas(self):
        return [{"type": "function", "function": {"name": "read_file"}}]

    def run(self, name, arguments):
        self.calls.append((name, arguments))
        return self.answer


def content_texts(chunks):
    return [chunk.text for chunk in chunks if chunk.kind == "content"]


class ToolLoopTests(unittest.TestCase):
    def test_runs_the_tool_and_asks_again(self):
        provider = ScriptedProvider(
            [
                [
                    StreamChunk(text="Let me look. "),
                    StreamChunk(
                        kind="tool_call",
                        tool_call=ToolCall("call_1", "read_file", '{"path": "notes.txt"}'),
                    ),
                ],
                [StreamChunk(text="It says hello.")],
            ]
        )
        computer = FakeComputer()
        conversation = Conversation(
            provider, "model", "system", tools=computer
        )

        chunks = list(conversation.turn("what is in notes.txt?"))

        self.assertEqual(content_texts(chunks), ["Let me look. ", "It says hello."])
        self.assertEqual(computer.calls, [("read_file", '{"path": "notes.txt"}')])
        self.assertEqual(len(provider.requests), 2)
        self.assertIsNotNone(provider.requests[0].tools)
        self.assertIsNotNone(provider.requests[1].tools)

    def test_history_keeps_the_tool_exchange(self):
        provider = ScriptedProvider(
            [
                [
                    StreamChunk(
                        kind="tool_call",
                        tool_call=ToolCall("call_1", "read_file", '{"path": "notes.txt"}'),
                    )
                ],
                [StreamChunk(text="hello")],
            ]
        )
        conversation = Conversation(provider, "model", "system", tools=FakeComputer())
        list(conversation.turn("read it"))

        roles = [message.role for message in conversation.history]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])

        assistant = conversation.history[2]
        self.assertEqual(assistant.tool_calls[0].name, "read_file")
        tool_result = conversation.history[3]
        self.assertEqual(tool_result.tool_call_id, "call_1")
        self.assertEqual(tool_result.name, "read_file")
        self.assertEqual(tool_result.content, "the file says hello")

    def test_reply_yields_only_words(self):
        provider = ScriptedProvider(
            [
                [
                    StreamChunk(kind="reasoning", text="thinking"),
                    StreamChunk(
                        kind="tool_call",
                        tool_call=ToolCall("call_1", "read_file", "{}"),
                    ),
                ],
                [StreamChunk(text="done")],
            ]
        )
        conversation = Conversation(provider, "model", "system", tools=FakeComputer())
        self.assertEqual(list(conversation.reply("go")), ["done"])

    def test_tool_call_without_tools_ends_the_turn(self):
        provider = ScriptedProvider(
            [
                [
                    StreamChunk(text="I would read it."),
                    StreamChunk(
                        kind="tool_call",
                        tool_call=ToolCall("call_1", "read_file", "{}"),
                    ),
                ]
            ]
        )
        conversation = Conversation(provider, "model", "system")
        chunks = list(conversation.turn("read it"))
        self.assertEqual(content_texts(chunks), ["I would read it."])
        self.assertEqual(len(provider.requests), 1)

    def test_request_has_no_tools_when_disabled(self):
        conversation = Conversation(ScriptedProvider([]), "model", "system")
        self.assertIsNone(conversation.request().tools)


class HistoryWindowTests(unittest.TestCase):
    def test_system_prompt_survives_a_short_window(self):
        conversation = Conversation(ScriptedProvider([]), "model", "system", history_limit=3)
        conversation.history.extend(
            [
                Message(role="user", content="one"),
                Message(role="assistant", content="two"),
                Message(role="user", content="three"),
                Message(role="assistant", content="four"),
            ]
        )
        window = conversation.window()
        self.assertEqual(window[0].role, "system")
        self.assertEqual([message.content for message in window], ["system", "three", "four"])

    def test_an_orphan_tool_result_is_dropped(self):
        conversation = Conversation(ScriptedProvider([]), "model", "system", history_limit=4)
        conversation.history.extend(
            [
                Message(role="tool", content="orphan", tool_call_id="call_1"),
                Message(role="user", content="hello"),
            ]
        )
        window = conversation.window()
        self.assertEqual([message.role for message in window], ["system", "user"])


if __name__ == "__main__":
    unittest.main()
