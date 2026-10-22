"""Tests for the provider layer: ``python -m unittest discover -s tests``."""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tesy.config import env_get, load_env_file, system_prompt  # noqa: E402
from tesy.prompt import DEFAULT_SYSTEM_PROMPT  # noqa: E402
from tesy.providers import PROVIDERS, get_provider  # noqa: E402
from tesy.providers import local, openrouter, snowflake  # noqa: E402
from tesy.providers.base import ChatRequest, Message, ProviderError, ToolCall  # noqa: E402
from tesy.cli import (  # noqa: E402
    build_computer,
    computer_enabled,
    resolve_provider_name,
    resolve_system_prompt,
)


def texts(chunks):
    return [chunk.text for chunk in chunks]


class OpenRouterStreamTests(unittest.TestCase):
    def test_parses_content_chunks(self):
        lines = [
            b'data: {"choices":[{"delta":{"content":"He"}}]}',
            b'data: {"choices":[{"delta":{"content":"llo"}}]}',
            b"data: [DONE]",
            b"",
        ]
        self.assertEqual(texts(openrouter.parse_stream(lines)), ["He", "llo"])

    def test_ignores_non_data_lines(self):
        lines = [
            b": keep-alive",
            b"event: message",
            b'data: {"choices":[{"delta":{"content":"x"}}]}',
        ]
        self.assertEqual(texts(openrouter.parse_stream(lines)), ["x"])

    def test_raises_on_error_event(self):
        lines = [b'data: {"error":{"message":"bad key"}}']
        with self.assertRaises(ProviderError):
            list(openrouter.parse_stream(lines))

    def test_skips_malformed_json(self):
        lines = [b"data: {oops", b'data: {"choices":[{"delta":{"content":"ok"}}]}']
        self.assertEqual(texts(openrouter.parse_stream(lines)), ["ok"])

    def test_separates_reasoning_from_content(self):
        lines = [
            b'data: {"choices":[{"delta":{"content":"","reasoning":"thinking"}}]}',
            b'data: {"choices":[{"delta":{"content":"answer"}}]}',
        ]
        chunks = list(openrouter.parse_stream(lines))
        self.assertEqual([chunk.kind for chunk in chunks], ["reasoning", "content"])
        self.assertEqual(texts(chunks), ["thinking", "answer"])

    def test_assembles_a_streamed_tool_call(self):
        lines = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            b'"type":"function","function":{"name":"read_file","arguments":""}}]}}]}',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            b'"function":{"arguments":"{\\"path\\":"}}]}}]}',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            b'"function":{"arguments":" \\"notes.txt\\"}"}}]}}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
            b"data: [DONE]",
        ]
        chunks = list(openrouter.parse_stream(lines))
        tool_chunks = [chunk for chunk in chunks if chunk.kind == "tool_call"]
        self.assertEqual(len(tool_chunks), 1)
        call = tool_chunks[0].tool_call
        self.assertEqual(call.id, "call_1")
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.arguments, '{"path": "notes.txt"}')

    def test_keeps_parallel_tool_calls_apart(self):
        lines = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a",'
            b'"function":{"name":"read_file","arguments":"{}"}}]}}]}',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b",'
            b'"function":{"name":"list_directory","arguments":"{}"}}]}}]}',
            b"data: [DONE]",
        ]
        calls = [
            chunk.tool_call
            for chunk in openrouter.parse_stream(lines)
            if chunk.kind == "tool_call"
        ]
        self.assertEqual([call.name for call in calls], ["read_file", "list_directory"])
        self.assertEqual([call.id for call in calls], ["a", "b"])

    def test_default_model_is_free(self):
        self.assertTrue(openrouter.DEFAULT_MODEL.endswith(":free"))


class SnowflakePayloadTests(unittest.TestCase):
    def test_system_prompt_is_top_level(self):
        request = ChatRequest(
            messages=[
                Message(role="system", content="Be kind."),
                Message(role="user", content="Hello"),
            ],
            model="claude-sonnet-4-5",
        )
        payload = snowflake.build_payload(request)
        self.assertEqual(payload["system"], "Be kind.")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Hello"}])

    def test_joins_multiple_system_messages(self):
        request = ChatRequest(
            messages=[
                Message(role="system", content="One"),
                Message(role="system", content="Two"),
                Message(role="user", content="Hi"),
            ],
            model="claude-sonnet-4-5",
        )
        self.assertEqual(snowflake.build_payload(request)["system"], "One\n\nTwo")

    def test_omits_system_when_absent(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            model="claude-sonnet-4-5",
        )
        self.assertNotIn("system", snowflake.build_payload(request))

    def test_includes_temperature_only_when_set(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            model="claude-sonnet-4-5",
        )
        self.assertNotIn("temperature", snowflake.build_payload(request))
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            model="claude-sonnet-4-5",
            temperature=0.4,
        )
        self.assertEqual(snowflake.build_payload(request)["temperature"], 0.4)

    def test_converts_tools_to_anthropic_schema(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            model="claude-sonnet-4-5",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        )
        tool = snowflake.build_payload(request)["tools"][0]
        self.assertEqual(tool["name"], "read_file")
        self.assertEqual(tool["description"], "Read a file")
        self.assertEqual(tool["input_schema"], request.tools[0]["function"]["parameters"])
        self.assertNotIn("function", tool)

    def test_omits_tools_when_absent(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            model="claude-sonnet-4-5",
        )
        self.assertNotIn("tools", snowflake.build_payload(request))

    def test_assistant_tool_call_becomes_tool_use_block(self):
        request = ChatRequest(
            messages=[
                Message(
                    role="assistant",
                    content="Looking.",
                    tool_calls=(
                        ToolCall(id="call_1", name="read_file", arguments='{"path":"a.txt"}'),
                    ),
                )
            ],
            model="claude-sonnet-4-5",
        )
        message = snowflake.build_payload(request)["messages"][0]
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"][0], {"type": "text", "text": "Looking."})
        self.assertEqual(
            message["content"][1],
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "read_file",
                "input": {"path": "a.txt"},
            },
        )

    def test_tool_result_becomes_user_tool_result_block(self):
        request = ChatRequest(
            messages=[
                Message(role="user", content="my notes"),
                Message(
                    role="tool",
                    content="a.txt (1 line)",
                    tool_call_id="call_1",
                    name="read_file",
                ),
            ],
            model="claude-sonnet-4-5",
        )
        message = snowflake.build_payload(request)["messages"][1]
        self.assertEqual(message["role"], "user")
        self.assertEqual(
            message["content"][0],
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "a.txt (1 line)",
            },
        )


class SnowflakeStreamTests(unittest.TestCase):
    def test_parses_content_block_delta(self):
        lines = [
            b'data: {"type":"content_block_delta","delta":{"text":"Hi"}}',
            b"data: [DONE]",
        ]
        self.assertEqual(texts(snowflake.parse_stream(lines)), ["Hi"])

    def test_ignores_other_event_types(self):
        lines = [b'data: {"type":"message_start","delta":{"text":"nope"}}']
        self.assertEqual(texts(snowflake.parse_stream(lines)), [])

    def test_raises_on_error_event(self):
        lines = [b'data: {"error":{"message":"no access"}}']
        with self.assertRaises(ProviderError):
            list(snowflake.parse_stream(lines))

    def test_collects_streamed_tool_call(self):
        lines = [
            b'data: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file"}}',
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":"}}',
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"input_json_delta","partial_json":" \\"a.txt\\"}"}}',
            b'data: {"type":"content_block_stop","index":0}',
        ]
        calls = [
            chunk.tool_call
            for chunk in snowflake.parse_stream(lines)
            if chunk.kind == "tool_call"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].id, "toolu_1")
        self.assertEqual(calls[0].name, "read_file")
        self.assertEqual(json.loads(calls[0].arguments), {"path": "a.txt"})

    def test_ignores_text_delta_of_other_type(self):
        lines = [
            b'data: {"type":"content_block_delta","index":0,'
            b'"delta":{"type":"input_json_delta","partial_json":"{}"}}'
        ]
        self.assertEqual(texts(snowflake.parse_stream(lines)), [])


class LocalProviderTests(unittest.TestCase):
    def test_is_registered(self):
        self.assertIn("local", PROVIDERS)

    def test_reads_base_url_from_environment(self):
        with tempfile_env({"OLLAMA_BASE_URL": "http://127.0.0.1:1234", "OLLAMA_MODEL": "my-tesy"}):
            settings = local.LocalSettings.from_env()
            self.assertEqual(settings.base_url, "http://127.0.0.1:1234")
            self.assertEqual(settings.model, "my-tesy")

    def test_default_base_url(self):
        with tempfile_env({"OLLAMA_BASE_URL": "", "OLLAMA_MODEL": ""}):
            settings = local.LocalSettings.from_env()
            self.assertEqual(settings.base_url, local.DEFAULT_BASE_URL)
            self.assertEqual(settings.model, local.DEFAULT_MODEL)

    def test_model_resolution_prefers_explicit_request(self):
        provider = local.LocalProvider(local.LocalSettings())
        self.assertEqual(provider.resolve_model("custom"), "custom")


class ProviderRegistryTests(unittest.TestCase):
    def test_all_providers_are_registered(self):
        self.assertIn("openrouter", PROVIDERS)
        self.assertIn("snowflake", PROVIDERS)
        self.assertIn("local", PROVIDERS)

    def test_unknown_provider_raises(self):
        with self.assertRaises(ProviderError):
            get_provider("nope")

    def test_openrouter_requires_api_key(self):
        previous = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            with self.assertRaises(ProviderError) as context:
                get_provider("openrouter")
            self.assertIn("OPENROUTER_API_KEY", str(context.exception))
        finally:
            if previous is not None:
                os.environ["OPENROUTER_API_KEY"] = previous

    def test_model_resolution_prefers_explicit_request(self):
        provider = openrouter.OpenRouterProvider(
            openrouter.OpenRouterSettings(api_key="test")
        )
        self.assertEqual(provider.resolve_model("vendor/some-model"), "vendor/some-model")

    def test_model_resolution_falls_back_to_environment(self):
        previous = os.environ.get("OPENROUTER_MODEL")
        os.environ["OPENROUTER_MODEL"] = "vendor/env-model"
        try:
            provider = openrouter.OpenRouterProvider(
                openrouter.OpenRouterSettings(api_key="test")
            )
            self.assertEqual(provider.resolve_model(None), "vendor/env-model")
        finally:
            if previous is None:
                os.environ.pop("OPENROUTER_MODEL", None)
            else:
                os.environ["OPENROUTER_MODEL"] = previous

    def test_model_resolution_falls_back_to_default(self):
        previous = os.environ.pop("OPENROUTER_MODEL", None)
        try:
            provider = openrouter.OpenRouterProvider(
                openrouter.OpenRouterSettings(api_key="test")
            )
            self.assertEqual(provider.resolve_model(None), openrouter.DEFAULT_MODEL)
        finally:
            if previous is not None:
                os.environ["OPENROUTER_MODEL"] = previous


class ComputerConfigTests(unittest.TestCase):
    class ToolProvider:
        supports_tools = True

    class PlainProvider:
        supports_tools = False

    def test_enabled_by_default(self):
        with tempfile_env({"TESY_COMPUTER": ""}):
            self.assertTrue(computer_enabled())

    def test_disabled_by_flag(self):
        self.assertFalse(computer_enabled(disabled=True))

    def test_disabled_by_environment(self):
        for value in ("off", "0", "false", "no", "disabled"):
            with tempfile_env({"TESY_COMPUTER": value}):
                self.assertFalse(computer_enabled(), value)

    def test_only_for_tool_capable_providers(self):
        with tempfile_env({"TESY_COMPUTER": ""}):
            self.assertIsNotNone(build_computer(self.ToolProvider()))
            self.assertIsNone(build_computer(self.PlainProvider()))
            self.assertIsNone(build_computer(self.ToolProvider(), disabled=True))


class ConfigTests(unittest.TestCase):
    def test_env_file_does_not_override_existing_values(self):
        with tempfile_env({"TESY_TEST_KEY": "from-environment"}):
            with tempfile_file("TESY_TEST_KEY=from-file\nOTHER_KEY=other\n") as path:
                load_env_file(path)
                self.assertEqual(os.environ["TESY_TEST_KEY"], "from-environment")
                self.assertEqual(os.environ["OTHER_KEY"], "other")
                os.environ.pop("OTHER_KEY", None)

    def test_env_get_skips_empty_values(self):
        with tempfile_env({"EMPTY_ONE": "", "FULL_ONE": "value"}):
            self.assertEqual(env_get("EMPTY_ONE", "FULL_ONE"), "value")
            self.assertIsNone(env_get("EMPTY_ONE", "MISSING_ONE"))

    def test_provider_name_resolution_order(self):
        with tempfile_env({"TESY_PROVIDER": "snowflake"}):
            self.assertEqual(resolve_provider_name(None), "snowflake")
            self.assertEqual(resolve_provider_name("openrouter"), "openrouter")
        with tempfile_env({"TESY_PROVIDER": ""}):
            self.assertEqual(resolve_provider_name(None), "openrouter")

    def test_system_prompt_resolution(self):
        with tempfile_env({"TESY_SYSTEM": "be brief"}):
            self.assertEqual(resolve_system_prompt(None), "be brief")
            self.assertEqual(resolve_system_prompt("override"), "override")

    def test_system_prompt_reads_persona_file_after_marker(self):
        with tempfile_file("notes about the file\n---\nbe warm\n") as path:
            with tempfile_env({"TESY_SYSTEM_FILE": path, "TESY_SYSTEM": ""}):
                self.assertEqual(system_prompt(), "be warm")

    def test_system_prompt_uses_whole_file_when_no_marker(self):
        with tempfile_file("be warm\n") as path:
            with tempfile_env({"TESY_SYSTEM_FILE": path, "TESY_SYSTEM": ""}):
                self.assertEqual(system_prompt(), "be warm")

    def test_system_prompt_falls_back_to_default(self):
        with tempfile_env({"TESY_SYSTEM_FILE": "/nonexistent/persona.md", "TESY_SYSTEM": ""}):
            self.assertEqual(system_prompt(), DEFAULT_SYSTEM_PROMPT)


class MessageTests(unittest.TestCase):
    def test_to_dict(self):
        self.assertEqual(
            Message(role="user", content="hi").to_dict(),
            {"role": "user", "content": "hi"},
        )

    def test_tool_call_dict(self):
        message = Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall("call_1", "read_file", '{"path": "x"}'),),
        )
        self.assertEqual(
            message.to_dict()["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "x"}'},
                }
            ],
        )

    def test_tool_result_dict(self):
        message = Message(
            role="tool", content="hello", tool_call_id="call_1", name="read_file"
        )
        self.assertEqual(
            message.to_dict(),
            {
                "role": "tool",
                "content": "hello",
                "tool_call_id": "call_1",
                "name": "read_file",
            },
        )


def tempfile_env(values: dict[str, str]):
    class _Context:
        def __enter__(self):
            self.previous = {key: os.environ.get(key) for key in values}
            os.environ.update(values)

        def __exit__(self, *args):
            for key, value in self.previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            return False

    return _Context()


def tempfile_file(content: str):
    import tempfile

    class _Context:
        def __enter__(self):
            handle = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
            handle.write(content)
            handle.close()
            self.path = handle.name
            return self.path

        def __exit__(self, *args):
            os.unlink(self.path)
            return False

    return _Context()


if __name__ == "__main__":
    unittest.main()
