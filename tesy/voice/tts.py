"""Speak text with piper, playing it through PipeWire.

Loading a piper voice takes several seconds, so the speaker keeps one voice
loaded in a small helper process and reuses it for every sentence. If the
helper is not available the plain ``piper`` command is used instead.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading

from .config import VoiceConfig
from .sentences import clean_for_speech

PIPER_RATE = 22050
FLUSH_TIMEOUT_S = 2.0


class SpeechError(RuntimeError):
    """Raised when piper cannot run."""


class InterruptedSpeech(RuntimeError):
    """Raised inside the speech pipeline once the user has cut in."""


class PiperSpeaker:
    """Turns text into sound using the local piper model."""

    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._process: subprocess.Popen | None = None
        self._log = None
        self._rate = PIPER_RATE
        self._lock = threading.RLock()
        self._player: subprocess.Popen | None = None
        self._stop = threading.Event()

    @property
    def is_playing(self) -> bool:
        """Is a sentence coming out of the speaker right now?"""
        with self._lock:
            player = self._player
        return player is not None and player.poll() is None

    # -- the worker --------------------------------------------------------
    def _start_worker(self) -> subprocess.Popen | None:
        """Start the helper that holds the voice in memory."""
        python = self.config.piper_python
        worker = self.config.piper_worker
        if python is None or not worker.is_file():
            return None

        self.config.voice_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.config.voice_dir / "piper_worker.log"
        self._log = open(log_path, "ab")
        process = subprocess.Popen(
            [str(python), str(worker), str(self.config.tts_model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            start_new_session=True,
        )

        header = process.stdout.readline()
        if not header:
            process.kill()
            raise SpeechError(f"the piper worker did not start, see {log_path}")
        try:
            ready = json.loads(header.decode("utf-8"))
        except ValueError as error:
            process.kill()
            raise SpeechError(f"the piper worker sent a bad reply: {error}") from error

        self._rate = ready.get("rate", PIPER_RATE)
        return process

    def _worker(self) -> subprocess.Popen | None:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._process = self._start_worker()
        return self._process

    def _stop_worker(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.write(b'{"cmd": "quit"}\n')
                process.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def close(self) -> None:
        """Shut the helper down."""
        self._stop_worker()
        if self._log is not None:
            self._log.close()
            self._log = None

    # -- speaking ----------------------------------------------------------
    def available(self) -> bool:
        return self.config.piper_binary is not None and self.config.tts_model.is_file()

    def warmup(self) -> None:
        """Start the helper so the voice is ready for the first reply.

        Loading the model takes several seconds, which is a lot to wait for
        while someone is waiting for an answer.
        """
        try:
            self._worker()
        except SpeechError:
            pass

    def synthesize(self, text: str) -> bytes:
        """Return raw signed 16 bit PCM for ``text``."""
        self._check_stop()
        text = clean_for_speech(text)
        if not text:
            return b""
        worker = self._worker()
        if worker is None:
            return self._synthesize_with_command(text)
        try:
            return self._ask_worker(worker, text)
        except InterruptedSpeech:
            raise
        except SpeechError:
            # The helper may have died between sentences, give it one retry.
            self._stop_worker()
            worker = self._worker()
            if worker is None:
                return self._synthesize_with_command(text)
            return self._ask_worker(worker, text)

    def _ask_worker(self, worker: subprocess.Popen, text: str) -> bytes:
        request = json.dumps({"text": text}).encode("utf-8") + b"\n"
        assert worker.stdin is not None and worker.stdout is not None
        try:
            worker.stdin.write(request)
            worker.stdin.flush()
        except (OSError, ValueError) as error:
            raise SpeechError(f"the piper worker stopped: {error}") from error

        header = worker.stdout.readline()
        if not header:
            raise SpeechError("the piper worker stopped unexpectedly")
        try:
            reply = json.loads(header.decode("utf-8"))
        except ValueError as error:
            raise SpeechError(f"the piper worker sent a bad reply: {error}") from error

        if not reply.get("ok"):
            raise SpeechError(reply.get("error", "piper failed"))

        size = int(reply.get("bytes", 0))
        audio = worker.stdout.read(size)
        if len(audio) != size:
            raise SpeechError("the piper worker sent a short reply")
        return audio

    def _synthesize_with_command(self, text: str) -> bytes:
        """Fall back to running piper for each sentence."""
        binary = self.config.piper_binary
        if binary is None:
            raise SpeechError("piper is not installed. Run: ./scripts/install_voice.sh")
        if not self.config.tts_model.is_file():
            raise SpeechError(
                f"the voice model {self.config.tts_model} is missing. "
                "Run: ./scripts/install_voice.sh"
            )

        command = [
            str(binary),
            "-m",
            str(self.config.tts_model),
            "--output-raw",
        ]
        try:
            result = subprocess.run(
                command,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=120,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as error:
            raise SpeechError("piper timed out") from error
        except OSError as error:
            raise SpeechError(f"could not run piper: {error}") from error

        self._check_stop()

        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise SpeechError(f"piper failed: {detail or f'exit code {result.returncode}'}")

        self._rate = PIPER_RATE
        return result.stdout

    def speak(self, text: str) -> None:
        """Synthesise ``text`` and play it."""
        audio = self.synthesize(text)
        if not audio:
            return
        self.play(audio)

    def play(self, audio: bytes) -> None:
        """Play raw signed 16 bit PCM through PipeWire.

        The player runs as a child process and its progress is watched, so
        :meth:`interrupt` can cut a sentence short the instant the user
        starts talking. It runs in its own process group, which keeps
        Ctrl-C in the terminal away from it.
        """
        if not shutil.which("pw-play"):
            raise SpeechError("pw-play is not installed (PipeWire)")

        command = [
            "pw-play",
            "--raw",
            "--rate",
            str(self._rate),
            "--channels",
            "1",
            "--format",
            "s16",
        ]
        if self.config.speaker:
            command.append(f"--target={self.config.speaker}")
        command.append("-")

        try:
            player = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            raise SpeechError(f"could not play audio: {error}") from error

        with self._lock:
            self._stop.clear()
            self._player = player
        try:
            try:
                player.communicate(audio, timeout=300)
            except subprocess.TimeoutExpired as error:
                player.kill()
                player.communicate()
                raise SpeechError("playback timed out") from error
        finally:
            with self._lock:
                self._player = None

    def interrupt(self) -> None:
        """Stop the sentence being spoken, if any, and forget the rest."""
        with self._lock:
            self._stop.set()
            player = self._player
        if player is not None and player.poll() is None:
            try:
                player.terminate()
            except OSError:
                pass

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise InterruptedSpeech()

    def close(self) -> None:
        """Shut the helper down."""
        self._stop_worker()
        if self._log is not None:
            self._log.close()
            self._log = None
