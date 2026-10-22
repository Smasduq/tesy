"""The voice conversation loop.

Tesy listens, transcribes, replies, and speaks. She keeps listening while she
speaks, so talking over her cuts her off and starts a new turn. That is the
only way a spoken conversation feels like a conversation rather than a
answering machine.
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
from .audio import level_of
from .config import VoiceConfig
from .recorder import FRAME_BYTES, EndOfSource, Recorder
from .sentences import SentenceBuffer
from .stt import WhisperTranscriber
from .tts import InterruptedSpeech, PiperSpeaker


def _still_speaking(sentences: Queue, audio: Queue, speaker: PiperSpeaker) -> bool:
    """Is there speech still queued, being made, or coming out of the speaker?

    The queues are checked as well as the player because a sentence can be
    waiting to be spoken when the reply itself has already been sent.
    """
    return not sentences.empty() or not audio.empty() or speaker.is_playing


class _PrefixStream:
    """Hands a captured phrase back to the recorder before live audio.

    Words said while Tesy was talking were already read off the microphone.
    Feeding them in first means the phrase is transcribed as one utterance
    instead of losing whatever was said before she stopped.
    """

    def __init__(self, prefix: bytes, stream: object | None) -> None:
        self._prefix = bytes(prefix)
        self._stream = stream

    def read(self, size: int) -> bytes:
        if self._prefix:
            chunk = self._prefix[:size]
            self._prefix = self._prefix[size:]
            return chunk
        if self._stream is None:
            return b""
        return self._stream.read(size)


class _Interrupter:
    """Watches the microphone while Tesy speaks and shouts when cut in.

    The capture is opened before the reply is spoken so no words are lost to
    the stream starting up. Frames are measured, not transcribed: a handful
    of frames clearly above the room's noise floor is enough to stop the
    voice, and only then is the rest of the phrase transcribed properly.
    """

    CALIBRATION_FRAMES = 10
    REQUIRED_FRAMES = 4           # 80ms of speech, long enough to ignore a click
    MIN_FRAMES_BEFORE_STOP = 6    # do not cut in on Tesy's own first syllables
    NOISE_FACTOR = 3.0
    MIN_THRESHOLD = 0.02

    def __init__(self, recorder: Recorder, config: VoiceConfig) -> None:
        self.recorder = recorder
        self.config = config
        self.process = None
        self.stream = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._levels: list[float] = []
        self._lock = threading.Lock()
        self._captured = b""
        self.heard_speech = False

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> bool:
        """Start listening. Returns False if the microphone will not open."""
        if self.recorder.is_file_source:
            # Replaying a clip: the recorder already owns the reader, and
            # there is no live microphone to watch.
            return False
        try:
            self.process, self.stream = self.recorder.open_stream()
        except (OSError, RuntimeError):
            return False
        return True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def stop(self) -> bytes:
        """Stop watching, close the capture, and return what was heard.

        The captured audio starts at the first frame that counted as speech,
        so whatever the user said while Tesy was talking comes back as the
        opening of their next sentence.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        with self._lock:
            captured = self._captured
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        self.stream = None
        return captured

    # -- the watcher -------------------------------------------------------
    def _threshold(self) -> float:
        if self.config.threshold is not None:
            return self.config.threshold
        if len(self._levels) < self.CALIBRATION_FRAMES:
            return self.MIN_THRESHOLD
        ordered = sorted(self._levels[-50:])
        floor = ordered[len(ordered) // 2]
        return max(self.MIN_THRESHOLD, floor * self.NOISE_FACTOR)

    def _watch(self) -> None:
        run = 0
        frames = 0
        stream = self.stream
        try:
            while not self._stop.is_set():
                frame = stream.read(FRAME_BYTES)
                if not frame:
                    return
                frames += 1
                level = level_of(frame)
                threshold = self._threshold()
                with self._lock:
                    if self.heard_speech:
                        self._captured += frame
                    elif level <= threshold:
                        # Track the room, but never learn from something that
                        # is already loud: speech in the median would drag the
                        # threshold up above the voice it is meant to catch.
                        self._levels.append(level)

                if level > threshold:
                    run += 1
                else:
                    if self.heard_speech:
                        # The user has stopped for a moment; the phrase so
                        # far is theirs, so hand the rest of the turn back
                        # to the recorder.
                        return
                    run = 0
                    continue

                if not self.heard_speech:
                    if frames < self.MIN_FRAMES_BEFORE_STOP:
                        continue
                    if run >= self.REQUIRED_FRAMES:
                        with self._lock:
                            self.heard_speech = True
                            self._captured = b""
                        # Keep reading to keep the words that follow, but
                        # stop the moment the user pauses.
                        continue
        except Exception:
            return


def speak_stream(
    chunks: Iterator[str],
    speaker: PiperSpeaker,
    show_text: bool = True,
    interrupter: _Interrupter | None = None,
) -> bool:
    """Print a reply as it arrives and speak it sentence by sentence.

    Synthesis and playback run in their own threads, so the next sentence is
    being prepared while the previous one is still being spoken. When an
    ``interrupter`` is watching the microphone, the moment it reports speech
    the voice stops and the remaining reply is abandoned. Returns True when
    the reply was cut short.
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
            except InterruptedSpeech:
                return
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
            except InterruptedSpeech:
                return
            except Exception as error:
                errors.append(str(error))
                return

    threads = [
        threading.Thread(target=synthesize, daemon=True),
        threading.Thread(target=playback, daemon=True),
    ]
    for thread in threads:
        thread.start()

    buffer = SentenceBuffer()
    interrupted = False
    try:
        for chunk in chunks:
            if show_text:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            for sentence in buffer.feed(chunk):
                sentences.put(sentence)
            if interrupter is not None and interrupter.heard_speech:
                interrupted = True
                break

        if not interrupted:
            for sentence in buffer.flush():
                sentences.put(sentence)

        if not interrupted and interrupter is not None:
            # The reply is over, but the last sentence may still be
            # synthesising or playing. Keep watching until it has actually
            # finished, so the tail of the reply can be talked over too.
            while not interrupter.heard_speech and _still_speaking(
                sentences, audio, speaker
            ):
                time.sleep(0.05)
            if interrupter.heard_speech:
                interrupted = True
    finally:
        if interrupted:
            speaker.interrupt()
        sentences.put(None)
        for thread in threads:
            thread.join(timeout=5)
        if show_text:
            sys.stdout.write("\n")
            sys.stdout.flush()

    if errors:
        print(f"speech error: {errors[0]}", file=sys.stderr)
    return interrupted


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
        self._pending: bytes = b""

    def run(self) -> int:
        print(
            f"Tesy is listening ({self.conversation.model}). "
            "Speak, or press Ctrl-D to stop."
        )
        print(f"  microphone: {self.config.mic_source}")
        print(f"  speaker:    {self.config.speaker}")
        print(f"  voice:      {self.config.tts_model.name}")
        if self.speak and self.config.barge_in:
            print("  interrupt:  talk over Tesy to cut her off")
        if self.conversation.tools is not None:
            print(f"  files:      read-only under {self.conversation.tools.root}")

        if self.speak:
            # Load the voice while the first question is still being asked.
            threading.Thread(target=self.speaker.warmup, daemon=True).start()

        try:
            return self._loop()
        finally:
            self.speaker.close()

    def _loop(self) -> int:
        while True:
            try:
                utterance = self._listen()
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

            if not self.speak:
                for chunk in self.conversation.reply(heard):
                    if chunk:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()
                continue

            interrupter = None
            if self.config.barge_in:
                interrupter = _Interrupter(self.recorder, self.config)
                if not interrupter.open():
                    interrupter = None

            if interrupter is not None:
                interrupter.start()

            try:
                speak_stream(
                    self.conversation.reply(heard),
                    self.speaker,
                    show_text=self.show_text,
                    interrupter=interrupter,
                )
            finally:
                if interrupter is not None:
                    self._pending = interrupter.stop()

    def _listen(self):
        """Record one turn, folded together with anything said mid-reply.

        When the user cut Tesy off, the words already captured are replayed
        through the recorder and listening continues on the same microphone
        stream, so the beginning of the sentence is not lost.
        """
        if not self._pending:
            return self.recorder.record()

        prefix = self._pending
        self._pending = b""
        process = None
        stream = None
        if not self.recorder.is_file_source:
            try:
                process, stream = self.recorder.open_stream()
            except (OSError, RuntimeError):
                process = stream = None
        try:
            return self.recorder.record(
                stream=_PrefixStream(prefix, stream),
                process=process,
            )
        finally:
            if process is not None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass

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
