"""Command line interface for Tesy."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from .chat import Conversation
from .config import env_get, load_env_file, system_prompt
from .providers import DEFAULT_PROVIDER, PROVIDERS, get_provider
from .providers.base import ProviderError
from .providers.local import list_models as list_local_models
from .providers.openrouter import REASONING_EFFORTS, list_models

DEFAULT_MAX_TOKENS = 1024
HISTORY_LIMIT = 40


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tesy",
        description="Chat with Tesy. Talks to OpenRouter by default, "
        "or to Snowflake Cortex with --provider snowflake.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to send (or set TESY_PROMPT). Omit with --interactive.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=sorted(PROVIDERS),
        help=f"Chat backend to use (default: TESY_PROVIDER or {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use (default: the provider's default model)",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="System prompt (default: TESY_SYSTEM, persona.md, or Tesy's built-in)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens per reply (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (provider default when omitted)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=REASONING_EFFORTS,
        help="How much a reasoning model should think before answering "
        "(use 'none' to keep reasoning models from eating the token budget)",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Print the model's reasoning to stderr while it thinks",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Keep the conversation going instead of sending a single prompt",
    )
    parser.add_argument(
        "--talk",
        action="store_true",
        help="Talk to Tesy out loud: listen, reply, speak",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="With --talk, transcribe and reply in text but stay quiet",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak the reply out loud",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List the models available through OpenRouter and exit",
    )
    parser.add_argument(
        "--free",
        action="store_true",
        help="With --list-models, show only the free models",
    )
    parser.add_argument(
        "--list-local-models",
        action="store_true",
        help="List the models available on the local Ollama server and exit",
    )
    return parser


def resolve_provider_name(requested: str | None) -> str:
    return requested or env_get("TESY_PROVIDER") or DEFAULT_PROVIDER


def resolve_system_prompt(requested: str | None) -> str:
    return system_prompt(requested)


def print_reply(conversation: Conversation, user_text: str, show_reasoning: bool = False) -> None:
    """Stream a reply to the terminal."""
    pieces: list[str] = []
    for chunk in conversation.provider.stream(conversation.ask(user_text)):
        if chunk.kind == "reasoning":
            if show_reasoning:
                sys.stderr.write(chunk.text)
                sys.stderr.flush()
            continue
        pieces.append(chunk.text)
        sys.stdout.write(chunk.text)
        sys.stdout.flush()
    conversation.remember_reply("".join(pieces))
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_interactive(conversation: Conversation, show_reasoning: bool = False) -> int:
    print(f"Tesy is listening ({conversation.model}). Ctrl-D or 'exit' to stop.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            print()
            return 0

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            return 0

        print("Tesy:", end=" ", flush=True)
        print_reply(conversation, user_input, show_reasoning)


def print_models(models: Iterable[dict], only_free: bool = False) -> None:
    for model in models:
        identifier = model.get("id", "?")
        if only_free and not identifier.endswith(":free"):
            continue
        pricing = model.get("pricing") or {}
        prompt_price = pricing.get("prompt") or "0"
        context = model.get("context_length") or 0
        print(f"{identifier:<55} ctx={context:<9} in=${prompt_price}/token")


def main(argv: list[str] | None = None) -> int:
    try:
        load_env_file()
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        try:
            print_models(list_models(), only_free=args.free)
        except ProviderError as error:
            print(error, file=sys.stderr)
            return 1
        return 0

    if args.list_local_models:
        try:
            for name in list_local_models():
                print(name)
        except ProviderError as error:
            print(error, file=sys.stderr)
            return 1
        return 0

    try:
        provider = get_provider(resolve_provider_name(args.provider))
    except ProviderError as error:
        print(error, file=sys.stderr)
        return 1

    model = provider.resolve_model(args.model)
    conversation = Conversation(
        provider,
        model=model,
        system_prompt=resolve_system_prompt(args.system),
        max_tokens=args.max_tokens or getattr(provider, "default_max_tokens", DEFAULT_MAX_TOKENS),
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        history_limit=HISTORY_LIMIT,
    )

    try:
        if args.talk:
            from .voice.config import resolve_voice_config
            from .voice.loop import VoiceLoop

            voice_config = resolve_voice_config()
            loop = VoiceLoop(conversation, voice_config, speak=not args.no_tts)
            return loop.run()

        if args.interactive:
            return run_interactive(conversation, args.show_reasoning)

        prompt = args.prompt or env_get("TESY_PROMPT")
        if not prompt:
            try:
                prompt = input("You: ").strip()
            except EOFError:
                parser.error("provide a prompt, --interactive or --talk")
        if not prompt:
            parser.error("prompt cannot be empty")

        if args.speak:
            from .voice.config import resolve_voice_config
            from .voice.loop import speak_stream
            from .voice.tts import PiperSpeaker

            speaker = PiperSpeaker(resolve_voice_config())
            speak_stream(conversation.reply(prompt), speaker)
            return 0

        print_reply(conversation, prompt, args.show_reasoning)
        return 0
    except ProviderError as error:
        print(error, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
