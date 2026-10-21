"""The half-duplex voice conversation loop.

Tesy listens, transcribes, replies, and speaks. While she speaks she is not
listening, which keeps the microphone from picking up her own voice and
makes the whole thing much easier to follow.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Iterator

from ..chat import Conversation
from ..config import env_get
from .audio import clipped_fraction
from .config import VoiceConfig
from .recorder import EndOfSource, Recorder
from .sentences import SentenceBuffer
from .stt import WhisperTranscriber
from .tts import PiperSpeaker


def speak_stream(chunks: Iterator[str], speaker: PiperSpeaker, show_text: bool = True) -> None:
    """Print a reply as it arrives and speak it sentence by sentence.

    Synthesis and playback run in their own threads, so the next sentence is
    being prepared while the previous one is still being spoken.
    """
    sentences: Queue[str | None] = Queue()
    audio: Queue[bytes | None] = Queue()
    errors: list[str] = []

    def synthesize() -> None:
        while True:
            sentence = sentences.get()
            if sentence is None:
                audio.put(None)
                return
            try:
                audio.put(speaker.synthesize(sentence))
            except Exception as error:  # keep the conversation alive
                errors.append(str(error))
                audio.put(b"")

    def playback() -> None:
        while True:
            chunk = audio.get()
            if chunk is None:
                return
            if not chunk:
                continue
            try:
                speaker.play(chunk)
            except Exception as error:
                errors.append(str(error))

    threads = [
        threading.Thread(target=synthesize, daemon=True),
        threading.Thread(target=playback, daemon=True),
    ]
    for thread in threads:
        thread.start()

    buffer = SentenceBuffer()
    for chunk in chunks:
        if show_text:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        for sentence in buffer.feed(chunk):
            sentences.put(sentence)

    for sentence in buffer.flush():
        sentences.put(sentence)
    sentences.put(None)

    if show_text:
        sys.stdout.write("\n")
        sys.stdout.flush()

    for thread in threads:
        thread.join(timeout=300)

    if errors:
        print(f"speech error: {errors[0]}", file=sys.stderr)


class VoiceLoop:
    """Runs the listen, think and speak cycle."""

    def __init__(
        self,
        conversation: Conversation,
        config: VoiceConfig,
        *,
        speak: bool = True,
        show_text: bool = True,
    ) -> None:
        self.conversation = conversation
        self.config = config
        self.speak = speak
        self.show_text = show_text
        self.recorder = Recorder(config)
        self.transcriber = WhisperTranscriber(config)
        self.speaker = PiperSpeaker(config)

    def run(self) -> int:
        print(
            f"Tesy is listening ({self.conversation.model}). "
            "Speak, or press Ctrl-D to stop."
        )
        print(f"  microphone: {self.config.mic_source}")
        print(f"  speaker:    {self.config.speaker}")
        print(f"  voice:      {self.config.tts_model.name}")

        if self.speak:
            # Load the voice while the first question is still being asked.
            threading.Thread(target=self.speaker.warmup, daemon=True).start()

        while True:
            try:
                utterance = self.recorder.record()
            except EndOfSource:
                print()
                return 0
            except KeyboardInterrupt:
                print()
                return 0
            except Exception as error:
                print(f"microphone error: {error}", file=sys.stderr)
                return 1

            if utterance is None:
                continue

            debug_directory = env_get("TESY_DEBUG_AUDIO")
            if debug_directory:
                self._save_utterance(utterance, debug_directory)

            try:
                heard = self.transcriber.transcribe(utterance)
            except Exception as error:
                print(f"transcription error: {error}", file=sys.stderr)
                continue

            if not heard:
                continue

            print(f"\nYou: {heard}")
            sys.stdout.write("Tesy: ")
            sys.stdout.flush()

            if self.speak:
                speak_stream(
                    self.conversation.reply(heard),
                    self.speaker,
                    show_text=self.show_text,
                )
            else:
                for chunk in self.conversation.reply(heard):
                    if chunk:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()

    def _save_utterance(self, utterance, directory: str) -> None:
        """Keep the captured audio so it can be inspected later."""
        folder = Path(directory).expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = folder / f"utterance-{stamp}.wav"
            path.write_bytes(utterance.to_wav())
        except OSError as error:
            print(f"could not save audio: {error}", file=sys.stderr)
            return

        clipped = clipped_fraction(utterance.samples)
        note = f", {clipped:.0%} clipped" if clipped > 0 else ""
        print(f"  saved {path} ({utterance.duration_s:.1f}s{note})")
