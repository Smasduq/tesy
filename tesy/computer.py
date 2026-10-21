"""Read-only access to files on this computer.

Tesy can look at files under the user's home directory but never write,
move or delete anything. Credential stores are refused outright: SSH keys,
cloud credentials, browser profiles, password managers and anything named
like a secret. The tools are only offered to providers that support tool
calling, and every lookup is printed so the user can see what was read.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

MAX_READ_BYTES = 128 * 1024
MAX_READ_LINES = 400
MAX_LIST_ENTRIES = 200
MAX_SEARCH_RESULTS = 100
MAX_SEARCH_FILES = 5000
MAX_SEARCH_BYTES = 2 * 1024 * 1024
MAX_LINE_CHARS = 400

# Directory names that hold credentials or private data. Matched against
# every path component, at any depth.
DENY_DIRS = {
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".docker",
    ".kube",
    ".mozilla",
    ".thunderbird",
    ".password-store",
    ".terraform.d",
    ".pki",
    ".android",
    ".npm",
    ".cargo",
    ".gradle",
    ".m2",
    ".streamlit",
    ".wakatime",
    ".steam",
    ".codex",
    ".claude",
    ".gemini",
    ".copilot",
    ".hermes",
    ".kilo",
    ".pi",
    "keyrings",
    "gcloud",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "bravesoftware",
    "microsoft-edge",
    "vivaldi",
    "opera",
    "firefox",
    "gh",
    "signal",
    "slack",
    "discord",
    "element",
    "telegram-desktop",
}

# File names that are secrets wherever they live.
DENY_FILES = {
    ".env",
    ".netrc",
    ".pgpass",
    ".my.cnf",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
    ".authinfo",
    ".htpasswd",
    ".envrc",
    ".vault-token",
    ".bash_history",
    ".zsh_history",
    ".python_history",
    ".node_repl_history",
    "credentials",
    "credentials.json",
    "credentials.toml",
    "credentials.yml",
    "credentials.yaml",
    "secrets.json",
    "secrets.toml",
    "secrets.yaml",
    "secrets.yml",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "cookies",
    "login data",
    "logins.json",
    "key4.db",
    "key3.db",
    "local state",
    "web data",
    "wallet.dat",
}

DENY_SUFFIXES = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    ".keystore",
    ".kdbx",
    ".ppk",
    ".asc",
)

# Folders that are never interesting to text search.
NOISE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".cache",
}


class ToolError(Exception):
    """A request that cannot be carried out, explained to the model."""


def _human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _clip(text: str, limit: int = MAX_LINE_CHARS) -> str:
    text = text.rstrip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _is_denied(path: Path, root: Path) -> bool:
    """Is this path a credential store or another off-limits file?"""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True

    name = path.name.casefold()
    if name in DENY_FILES or name.startswith(".env."):
        return True
    if name.endswith(DENY_SUFFIXES):
        return True

    return any(part.casefold() in DENY_DIRS for part in relative.parts)


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass
class ComputerAccess:
    """The read-only view of the home directory handed to the model."""

    root: Path = field(default_factory=Path.home)
    on_action: Callable[[str], None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(os.path.expanduser(str(self.root))).resolve()

    # -- tools ------------------------------------------------------------
    def schemas(self) -> list[dict[str, Any]]:
        """The tool definitions sent to the model."""
        home = str(self.root)
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        f"Read a text file under {home}. Paths may be relative "
                        "to the home directory or start with ~. Returns numbered "
                        "lines. Read-only: this cannot change files. Credential "
                        "files (SSH keys, .env, browser data) are refused."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File to read, e.g. 'Documents/notes.txt'.",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "First line to return, 1-based.",
                            },
                            "max_lines": {
                                "type": "integer",
                                "description": "How many lines to return (default 200, max 400).",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": (
                        f"List the files and folders in a directory under {home}. "
                        "Use it to find out what exists before reading."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory to list, default '~'.",
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": (
                        f"Search for a literal phrase inside text files under {home}. "
                        "Returns matching file names, line numbers and the lines "
                        "themselves. Skips binary files, huge files and build folders."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Text to look for, case-insensitive.",
                            },
                            "path": {
                                "type": "string",
                                "description": "File or directory to search, default '~'.",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "How many matching lines to return (default 30, max 100).",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def run(self, name: str, arguments: str) -> str:
        """Run a tool by name and return text for the model.

        Errors come back as text rather than exceptions so the model can
        correct itself instead of the conversation dying.
        """
        try:
            payload = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return "Error: the tool arguments were not valid JSON."
        if not isinstance(payload, dict):
            return "Error: the tool arguments must be a JSON object."

        try:
            if name == "read_file":
                return self.read_file(**payload)
            if name == "list_directory":
                return self.list_directory(**payload)
            if name == "search_files":
                return self.search_files(**payload)
            return f"Error: there is no tool called {name!r}."
        except ToolError as error:
            return f"Error: {error}"
        except TypeError as error:
            return f"Error: wrong arguments for {name}: {error}"
        except (OSError, ValueError, RuntimeError) as error:
            return f"Error: {error}"

    # -- implementation ---------------------------------------------------
    def read_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> str:
        target = self._resolve(path)
        if target.is_dir():
            raise ToolError(f"{self.display(target)} is a directory; use list_directory.")
        if not target.exists():
            raise ToolError(f"{self.display(target)} does not exist.")
        if not target.is_file():
            raise ToolError(f"{self.display(target)} is not a regular file.")

        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            raise ToolError(
                f"{self.display(target)} is {_human_size(size)}, larger than the "
                f"{_human_size(MAX_READ_BYTES)} limit. Use search_files to find "
                "the part you need."
            )

        raw = target.read_bytes()
        if b"\x00" in raw[:4096]:
            raise ToolError(f"{self.display(target)} looks like a binary file.")

        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if not lines:
            return f"{self.display(target)} is empty."

        try:
            first = int(start_line)
        except (TypeError, ValueError):
            first = 1
        try:
            count = int(max_lines)
        except (TypeError, ValueError):
            count = 200
        first = max(1, first)
        count = max(1, min(count, MAX_READ_LINES))

        window = lines[first - 1 : first - 1 + count]
        if not window:
            raise ToolError(
                f"{self.display(target)} only has {len(lines)} lines; "
                f"start_line {first} is past the end."
            )

        last = first + len(window) - 1
        self._notify(f"read {self.display(target)} (lines {first}-{last})")
        header = f"{self.display(target)} (lines {first}-{last} of {len(lines)})"
        body = "\n".join(f"{number}: {_clip(line)}" for number, line in enumerate(window, start=first))
        return f"{header}\n{body}"

    def list_directory(self, path: str = "~") -> str:
        target = self._resolve(path or "~")
        if not target.exists():
            raise ToolError(f"{self.display(target)} does not exist.")
        if not target.is_dir():
            raise ToolError(f"{self.display(target)} is a file; use read_file.")

        try:
            entries = sorted(
                target.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.casefold())
            )
        except OSError as error:
            raise ToolError(f"cannot list {self.display(target)}: {error}") from error

        lines: list[str] = []
        skipped = 0
        for entry in entries:
            if _is_denied(entry, self.root):
                skipped += 1
                continue
            if len(lines) >= MAX_LIST_ENTRIES:
                skipped += 1
                continue
            try:
                if entry.is_symlink() and not _is_inside(entry.resolve(), self.root):
                    lines.append(f"{entry.name} (link outside home, not followed)")
                elif entry.is_dir():
                    lines.append(f"{entry.name}/")
                else:
                    lines.append(f"{entry.name}  ({_human_size(entry.stat().st_size)})")
            except OSError:
                lines.append(f"{entry.name} (unreadable)")

        self._notify(f"listed {self.display(target)}")
        header = f"{self.display(target)} ({len(lines)} entries)"
        if skipped:
            header += f", {skipped} hidden or too many to show"
        return header + "\n" + ("\n".join(lines) if lines else "(empty)")

    def search_files(self, query: str, path: str = "~", max_results: int = 30) -> str:
        needle = (query or "").strip()
        if not needle:
            raise ToolError("query cannot be empty.")
        if len(needle) > 200:
            raise ToolError("query is too long; search for a shorter phrase.")

        try:
            limit = int(max_results)
        except (TypeError, ValueError):
            limit = 30
        limit = max(1, min(limit, MAX_SEARCH_RESULTS))

        target = self._resolve(path or "~")
        if not target.exists():
            raise ToolError(f"{self.display(target)} does not exist.")

        matches: list[str] = []
        files_scanned = 0
        truncated = False
        folded = needle.casefold()

        def consider(file_path: Path) -> bool:
            """Search one file; return False when there is no room left."""
            nonlocal files_scanned, truncated
            if _is_denied(file_path, self.root):
                return True
            try:
                resolved = file_path.resolve()
                if not _is_inside(resolved, self.root):
                    return True
                if not resolved.is_file():
                    return True
                size = resolved.stat().st_size
            except OSError:
                return True
            if size > MAX_SEARCH_BYTES:
                return True
            if files_scanned >= MAX_SEARCH_FILES:
                truncated = True
                return False
            files_scanned += 1
            try:
                raw = resolved.read_bytes()
            except OSError:
                return True
            if b"\x00" in raw[:4096]:
                return True
            text = raw.decode("utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                if folded in line.casefold():
                    matches.append(f"{self.display(resolved)}:{number}: {_clip(line.strip())}")
                    if len(matches) >= limit:
                        truncated = True
                        return False
            return True

        if target.is_file():
            consider(target)
        else:
            for directory, dirnames, filenames in os.walk(target, followlinks=False):
                folder = Path(directory)
                dirnames[:] = [
                    name
                    for name in dirnames
                    if name.casefold() not in NOISE_DIRS
                    and not _is_denied(folder / name, self.root)
                ]
                for filename in filenames:
                    if not consider(folder / filename):
                        break
                if len(matches) >= limit or files_scanned >= MAX_SEARCH_FILES:
                    truncated = True
                    break

        self._notify(f"searched {self.display(target)} for {needle!r}")
        if not matches:
            return f"No matches for {needle!r} in {self.display(target)} ({files_scanned} files scanned)."
        header = f"{len(matches)} match(es) for {needle!r} in {self.display(target)}"
        if truncated:
            header += " (stopped at the limit)"
        return header + "\n" + "\n".join(matches)

    # -- paths ------------------------------------------------------------
    def _resolve(self, raw: str) -> Path:
        text = (raw or "~").strip()
        if not text:
            text = "~"
        if text == "~" or text.startswith("~/"):
            candidate = self.root / text[2:]
        else:
            expanded = Path(os.path.expanduser(text))
            candidate = expanded if expanded.is_absolute() else self.root / expanded
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError):
            resolved = candidate.absolute()
        if not _is_inside(resolved, self.root):
            raise ToolError(
                f"{text} is outside {self.display(self.root)}; only the home "
                "directory can be read."
            )
        if _is_denied(resolved, self.root):
            raise ToolError(
                f"{self.display(resolved)} looks like a credential store and "
                "will not be opened."
            )
        return resolved

    def display(self, path: Path) -> str:
        """A short path for messages, relative to home where possible."""
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return str(path)
        return "~" if not relative.parts else f"~/{relative.as_posix()}"

    def _notify(self, message: str) -> None:
        if self.on_action is not None:
            self.on_action(message)
