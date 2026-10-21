"""Transcribe speech with whisper.cpp."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .config import VoiceConfig
from .recorder import Utterance

# whisper.cpp likes to invent a sentence when it hears nothing at all.
HALLUCINATIONS = {
    "you",
    "thank you.",
    "thanks for watching!",
    "please subscribe to my channel.",
    "[blank_audio]",
    "(silence)",
    "bye.",
    ".",
}


class TranscriptionError(RuntimeError):
    """Raised when whisper.cpp cannot run."""


class WhisperTranscriber:
    """Runs the local whisper.cpp binary over a recorded utterance."""

    def __init__(self, config: VoiceConfig) -> None:
        self.config = config

    def available(self) -> bool:
        return self.config.whisper_binary is not None and self.config.stt_model.is_file()

    def _environment(self) -> dict[str, str]:
        """The environment for whisper, with its own libraries on the path."""
        environment = dict(os.environ)
        library_dir = str(self.config.voice_dir / "bin")
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            f"{library_dir}:{existing}" if existing else library_dir
        )
        return environment

    def transcribe(self, utterance: Utterance) -> str:
        """Return the spoken text, or an empty string if nothing was said."""
        binary = self.config.whisper_binary
        if binary is None:
            raise TranscriptionError(
                "whisper.cpp is not installed. Run: ./scripts/install_voice.sh"
            )
        if not self.config.stt_model.is_file():
            raise TranscriptionError(
                f"the speech model {self.config.stt_model} is missing. "
                "Run: ./scripts/install_voice.sh"
            )

        with tempfile.TemporaryDirectory(prefix="tesy-") as folder:
            path = Path(folder) / "utterance.wav"
            path.write_bytes(utterance.to_wav())

            command = [
                str(binary),
                "-m",
                str(self.config.stt_model),
                "-f",
                str(path),
                "-l",
                self.config.language,
                "-t",
                str(self.config.threads),
                "-nt",
                "-np",
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=self._environment(),
                )
            except subprocess.TimeoutExpired as error:
                raise TranscriptionError("whisper.cpp timed out") from error
            except OSError as error:
                raise TranscriptionError(f"could not run whisper.cpp: {error}") from error

        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            message = detail[-1] if detail else f"exit code {result.returncode}"
            raise TranscriptionError(f"whisper.cpp failed: {message}")

        return clean_transcript(result.stdout)


def clean_transcript(text: str) -> str:
    """Strip whisper's decorations and the phrases it invents out of silence."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        lines.append(line)

    cleaned = " ".join(lines).strip()
    if cleaned.lower() in HALLUCINATIONS:
        return ""
    return cleaned
