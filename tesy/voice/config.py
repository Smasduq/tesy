"""Settings for the local voice stack."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import env_get

DEFAULT_VOICE_DIR = Path.home() / ".local" / "share" / "tesy"
DEFAULT_VOICE_MODEL = "en_US-amy-medium"
DEFAULT_STT_MODEL = "ggml-tiny.en.bin"
SAMPLE_RATE = 16000
FRAME_MS = 20


@dataclass(frozen=True)
class VoiceConfig:
    """Everything the voice stack needs to know."""

    voice_dir: Path
    mic_source: str | None
    speaker: str | None
    tts_model: Path
    stt_model: Path
    language: str
    silence_ms: int
    onset_ms: int
    min_speech_ms: int
    max_utterance_s: float
    threshold: float | None
    high_pass_hz: float
    threads: int

    @property
    def whisper_binary(self) -> Path | None:
        """The whisper.cpp executable, whichever name it was installed under."""
        for name in ("whisper-cli", "main"):
            candidate = self.voice_dir / "bin" / name
            if candidate.is_file():
                return candidate
        return None

    @property
    def piper_binary(self) -> Path | None:
        candidate = self.voice_dir / "venv" / "bin" / "piper"
        return candidate if candidate.is_file() else None

    @property
    def piper_python(self) -> Path | None:
        candidate = self.voice_dir / "venv" / "bin" / "python"
        return candidate if candidate.is_file() else None

    @property
    def piper_worker(self) -> Path:
        """The helper that keeps one voice loaded in memory."""
        override = env_get("TESY_PIPER_WORKER")
        if override:
            return Path(override)
        return self.voice_dir / "piper_worker.py"


def _int_env(name: str, default: int) -> int:
    value = env_get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float | None) -> float | None:
    value = env_get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def list_sources() -> list[str]:
    """Names of the audio capture PipeWire exposes."""
    return _pactl("sources")


def list_sinks() -> list[str]:
    """Names of the audio output PipeWire exposes."""
    return _pactl("sinks")


def _pactl(kind: str) -> list[str]:
    if not shutil.which("pactl"):
        return []
    try:
        result = subprocess.run(
            ["pactl", "list", kind, "short"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [line.split("\t")[1] for line in result.stdout.splitlines() if "\t" in line]


def default_mic_source() -> str | None:
    """The source PipeWire would use by default."""
    sources = [name for name in list_sources() if ".monitor" not in name]
    for source in sources:
        if source.startswith("bluez_input"):
            return source
    return sources[0] if sources else None


def default_speaker() -> str | None:
    """Prefer a wired output over Bluetooth.

    A Bluetooth headset that is being recorded from usually drops its output
    to the narrowband headset profile, which sounds bad. Playing through the
    laptop's own speaker avoids that entirely.
    """
    sinks = list_sinks()
    for sink in sinks:
        if sink.startswith("alsa_output"):
            return sink
    return sinks[0] if sinks else None


def _card_for_source(source: str) -> str | None:
    """The PipeWire card a Bluetooth source belongs to."""
    if not source.startswith("bluez_input."):
        return None
    device = source[len("bluez_input.") :]
    return f"bluez_card.{device.replace(':', '_')}"


def _cards() -> list[dict]:
    """The audio cards PipeWire knows about, with their profiles."""
    if not shutil.which("pactl"):
        return []
    try:
        result = subprocess.run(
            ["pactl", "list", "cards"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return []

    cards: list[dict] = []
    current: dict | None = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Name:"):
            current = {"name": stripped.split(":", 1)[1].strip(), "active": None, "mic": []}
            cards.append(current)
            continue
        if current is None:
            continue
        if stripped.startswith("Active Profile:"):
            current["active"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("headset") and "(sinks: 1, sources: 1" in stripped:
            current["mic"].append(stripped.split(":", 1)[0].strip())
    return cards


def _switch_profile(card: str, profile: str) -> bool:
    try:
        result = subprocess.run(
            ["pactl", "set-card-profile", card, profile],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def enable_bluetooth_microphone() -> str | None:
    """Turn on the microphone of a connected Bluetooth headset.

    A Bluetooth card sitting on the high fidelity profile can only play
    music. The microphone lives on the headset profile, so switch over when
    needed. Returns the profile that was selected, if any.
    """
    for card in _cards():
        if not card["name"].startswith("bluez_card"):
            continue
        active = card["active"]
        if active and active.startswith("headset"):
            continue
        for profile in card["mic"]:
            if _switch_profile(card["name"], profile):
                return profile
    return None


def ensure_microphone_profile(source: str | None) -> str | None:
    """Make sure the chosen source really has a microphone behind it."""
    if not source:
        return None
    card = _card_for_source(source)
    if card is None:
        return None

    for entry in _cards():
        if entry["name"] != card:
            continue
        active = entry["active"]
        if active and active.startswith("headset"):
            return None
        for profile in entry["mic"]:
            if _switch_profile(card, profile):
                return profile
    return None


def resolve_voice_config() -> VoiceConfig:
    """Build the voice configuration from the environment."""
    voice_dir = Path(env_get("TESY_VOICE_DIR", default=str(DEFAULT_VOICE_DIR)))
    voices_dir = voice_dir / "voices"
    models_dir = voice_dir / "models"

    tts_model = Path(
        env_get("TESY_TTS_MODEL", default=str(voices_dir / f"{DEFAULT_VOICE_MODEL}.onnx"))
    )
    stt_model = Path(
        env_get("TESY_STT_MODEL", default=str(models_dir / DEFAULT_STT_MODEL))
    )

    mic_source = env_get("TESY_MIC_SOURCE")
    if mic_source:
        ensure_microphone_profile(mic_source)
    else:
        # A headset on the music profile has no microphone, so ask for one
        # before deciding where to listen.
        enable_bluetooth_microphone()
        mic_source = default_mic_source()
    speaker = env_get("TESY_SPEAKER") or default_speaker()

    return VoiceConfig(
        voice_dir=voice_dir,
        mic_source=mic_source,
        speaker=speaker,
        tts_model=tts_model,
        stt_model=stt_model,
        language=env_get("TESY_STT_LANGUAGE", default="en"),
        silence_ms=_int_env("TESY_VAD_SILENCE_MS", 700),
        onset_ms=_int_env("TESY_VAD_ONSET_MS", 120),
        min_speech_ms=_int_env("TESY_VAD_MIN_SPEECH_MS", 250),
        max_utterance_s=float(_float_env("TESY_MAX_UTTERANCE_S", 20.0)),
        threshold=_float_env("TESY_VAD_THRESHOLD", None),
        high_pass_hz=float(_float_env("TESY_HIGH_PASS_HZ", 100.0)),
        threads=_int_env("TESY_STT_THREADS", 4),
    )
