"""Local voice support for Tesy.

The stack is deliberately plain: PipeWire for capture and playback, whisper.cpp
for transcription, piper for speech. No GPU, no paid API, no Python packages
outside the small piper virtualenv.
"""

from .config import VoiceConfig, resolve_voice_config
from .recorder import Recorder, Utterance
from .sentences import split_sentences
from .stt import WhisperTranscriber
from .tts import PiperSpeaker

__all__ = [
    "PiperSpeaker",
    "Recorder",
    "Utterance",
    "VoiceConfig",
    "WhisperTranscriber",
    "resolve_voice_config",
    "split_sentences",
]
