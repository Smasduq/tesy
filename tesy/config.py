"""Environment and configuration handling."""

from __future__ import annotations

import os
from pathlib import Path

from .prompt import DEFAULT_SYSTEM_PROMPT

ENV_FILE = ".env"
PERSONA_FILE = "persona.md"


def load_env_file(path: str | Path | None = None) -> None:
    """Load ``KEY=VALUE`` pairs from a dotenv-style file into the environment.

    Values already present in the environment win, so exported secrets are
    never overwritten by the file. Without an explicit path the file is looked
    up in the working directory first, then next to the project root.
    """
    candidates = [Path(path)] if path is not None else [
        Path(ENV_FILE),
        Path(__file__).resolve().parent.parent / ENV_FILE,
    ]

    for candidate in candidates:
        if candidate.is_file():
            _read_env_file(candidate)
            return


def _read_env_file(env_file: Path) -> None:
    for line_number, line in enumerate(env_file.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key.isascii() or not key.isidentifier():
            raise ValueError(f"Invalid entry in {env_file} on line {line_number}")

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def env_get(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return default


def system_prompt(explicit: str | None = None) -> str:
    """Resolve Tesy's system prompt.

    Order: the explicit flag, ``TESY_SYSTEM``, the ``TESY_SYSTEM_FILE``
    (default ``persona.md``), then the built-in personality.
    """
    if explicit:
        return explicit

    from_environment = env_get("TESY_SYSTEM")
    if from_environment:
        return from_environment

    path = Path(env_get("TESY_SYSTEM_FILE", default=PERSONA_FILE))
    if path.is_file():
        try:
            text = path.read_text().strip()
        except OSError:
            text = ""
        if text:
            return _prompt_part(text)

    return DEFAULT_SYSTEM_PROMPT


def _prompt_part(text: str) -> str:
    """Use everything after a ``---`` separator, or the whole text."""
    marker = "\n---"
    index = text.find(marker)
    if index == -1:
        return text
    return text[index + len(marker) :].strip() or text
