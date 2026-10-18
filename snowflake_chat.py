#!/usr/bin/env python3
"""Send a prompt to Snowflake Cortex and stream the response."""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "claude-sonnet-4-5"
API_PATH = "/api/v2/cortex/v1/messages"


def load_env_file() -> None:
    env_file = Path(".env")
    if not env_file.is_file():
        return

    for line_number, line in enumerate(env_file.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key.isascii() or not key.isidentifier():
            raise ValueError(f"Invalid entry in .env on line {line_number}")

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def stream_response(account_url: str, token: str, model: str, prompt: str) -> None:
    account_url = account_url.strip()
    if "://" not in account_url:
        account_url = f"https://{account_url}"
    url = f"{account_url.rstrip('/')}{API_PATH}"
    payload = json.dumps(
        {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
    ).encode("utf-8")
    request = Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue

                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break

                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(f"Snowflake API error: {event['error']}")

                delta = event.get("delta", {})
                text = delta.get("text")
                if event.get("type") == "content_block_delta" and isinstance(text, str):
                    print(text, end="", flush=True)
            print()
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Snowflake returned HTTP {error.code}: {details}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Could not connect to Snowflake: {error.reason}") from error


def main() -> int:
    try:
        load_env_file()
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to send to Snowflake Cortex (or set SNOWFLAKE_PROMPT)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SNOWFLAKE_MODEL", DEFAULT_MODEL),
        help="Model to use (default: SNOWFLAKE_MODEL or claude-sonnet-4-5)",
    )
    args = parser.parse_args()

    account_url = os.environ.get("SNOWFLAKE_ACCOUNT_URL")
    if not account_url:
        account = os.environ.get("account") or os.environ.get("ACCOUNT")
        if account:
            account_url = (
                account
                if "://" in account
                else f"https://{account}.snowflakecomputing.com"
            )
    token = (
        os.environ.get("SNOWFLAKE_PAT")
        or os.environ.get("api_key")
        or os.environ.get("API_KEY")
    )
    if not account_url or not token:
        parser.error(
            "set SNOWFLAKE_ACCOUNT_URL (or account) and SNOWFLAKE_PAT "
            "(or api_key) in the environment or .env file"
        )

    prompt = args.prompt or os.environ.get("SNOWFLAKE_PROMPT")
    if not prompt:
        try:
            prompt = input("Prompt: ").strip()
        except EOFError:
            parser.error(
                "provide a prompt or set the SNOWFLAKE_PROMPT environment variable"
            )
    if not prompt:
        parser.error("prompt cannot be empty")

    try:
        stream_response(account_url, token, args.model, prompt)
    except (RuntimeError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
