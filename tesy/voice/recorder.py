"""Record an utterance from the microphone and find where it starts and ends."""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from .audio import HighPass, level_of, to_wav
from .config import FRAME_MS, SAMPLE_RATE, VoiceConfig

FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
WARMUP_MS = 300
CALIBRATION_MS = 400
PRE_ROLL_MS = 400
MIN_THRESHOLD = 0.004
NOISE_FACTOR = 3.0
NOISE_WINDOW = 50  # frames, about one second
NOISE_MIN_SAMPLES = 10


@dataclass
class Utterance:
    """One captured phrase."""

    samples: bytes
    sample_rate: int = SAMPLE_RATE
    duration_s: float = 0.0

    def to_wav(self) -> bytes:
        return to_wav(self.samples, self.sample_rate)


class VoiceActivityDetector:
    """Turns a stream of loudness values into speech and silence decisions.

    The threshold follows the room. Background noise is tracked while
    nothing is being said, and speech has to be a good deal louder than that
    before it counts. A room that gets louder is picked up automatically, so
    the detector does not need to be told about it.
    """

    def __init__(
        self,
        *,
        threshold: float | None = None,
        silence_ms: int = 700,
        onset_ms: int = 120,
        min_speech_ms: int = 250,
        noise_factor: float = NOISE_FACTOR,
    ) -> None:
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.onset_ms = onset_ms
        self.min_speech_ms = min_speech_ms
        self.noise_factor = noise_factor

        self.speaking = False
        self._noise_levels: list[float] = []
        self._onset_run_ms = 0
        self._silence_run_ms = 0
        self._speech_ms = 0

    # -- the noise floor ----------------------------------------------------
    @property
    def effective_threshold(self) -> float:
        """The level speech has to beat to count as speech."""
        if self.threshold is not None:
            return self.threshold
        if len(self._noise_levels) < NOISE_MIN_SAMPLES:
            return MIN_THRESHOLD
        ordered = sorted(self._noise_levels)
        floor = ordered[len(ordered) // 2]
        return max(MIN_THRESHOLD, floor * self.noise_factor)

    def calibrate(self, levels: list[float]) -> None:
        """Seed the noise floor with a sample of background noise."""
        if self.threshold is not None or not levels:
            return
        self._noise_levels = list(levels[-NOISE_WINDOW:])

    def _remember_noise(self, level: float) -> None:
        self._noise_levels.append(level)
        if len(self._noise_levels) > NOISE_WINDOW:
            self._noise_levels.pop(0)

    # -- the state machine --------------------------------------------------
    def feed(self, level: float, frame_ms: int = FRAME_MS) -> str:
        """Feed one frame. Returns ``start``, ``end`` or an empty string."""
        threshold = self.effective_threshold
        loud = level > threshold

        if not self.speaking:
            if loud:
                self._onset_run_ms += frame_ms
                if self._onset_run_ms >= self.onset_ms:
                    self.speaking = True
                    self._onset_run_ms = 0
                    self._silence_run_ms = 0
                    self._speech_ms = 0
                    return "start"
            else:
                self._onset_run_ms = 0
                self._remember_noise(level)
            return ""

        self._speech_ms += frame_ms
        if loud:
            self._silence_run_ms = 0
        else:
            self._silence_run_ms += frame_ms
            if (
                self._silence_run_ms >= self.silence_ms
                and self._speech_ms >= self.min_speech_ms
            ):
                self.speaking = False
                self._onset_run_ms = 0
                self._silence_run_ms = 0
                self._speech_ms = 0
                return "end"
        return ""


class EndOfSource(RuntimeError):
    """Raised when a source has no more audio to give."""


class Recorder:
    """Captures speech from a PipeWire source."""

    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._reader: _WavReader | None = None

    @property
    def is_file_source(self) -> bool:
        """Is the source a WAV file instead of a device?"""
        return bool(self.config.mic_source) and self.config.mic_source.startswith("file:")

    def _command(self) -> list[str]:
        """Build the capture command.

        ``pw-cat`` is PipeWire's own tool and starts almost immediately.
        ``parec`` is the PulseAudio compatible one and takes a couple of
        seconds to open its stream, so it is only the fallback.
        """
        if shutil.which("pw-cat"):
            command = [
                "pw-cat",
                "--record",
                "--rate",
                str(SAMPLE_RATE),
                "--channels",
                "1",
                "--format",
                "s16",
            ]
            if self.config.mic_source:
                command.append(f"--target={self.config.mic_source}")
            command.append("-")
            return command

        command = [
            "parec",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
            "--format=s16le",
        ]
        if self.config.mic_source:
            command.append(f"--device={self.config.mic_source}")
        return command

    def _open_source(self) -> tuple[object | None, object | None]:
        """Return the process and the stream to read frames from."""
        if self.is_file_source:
            if self._reader is None:
                path = Path(str(self.config.mic_source)[len("file:") :])
                self._reader = _WavReader(path)
            return None, self._reader
        process = subprocess.Popen(
            self._command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        return process, process.stdout

    def record(self, timeout_s: float | None = None) -> Utterance | None:
        """Wait for someone to speak, then capture until they stop.

        Returns ``None`` if the microphone is unavailable or nothing was said.
        """
        if not shutil.which("pw-cat") and not shutil.which("parec"):
            raise RuntimeError("neither pw-cat nor parec is installed (PipeWire)")

        limit_s = timeout_s if timeout_s is not None else self.config.max_utterance_s
        process, stream = self._open_source()

        detector = VoiceActivityDetector(
            threshold=self.config.threshold,
            silence_ms=self.config.silence_ms,
            onset_ms=self.config.onset_ms,
            min_speech_ms=self.config.min_speech_ms,
        )

        calibration: list[float] = []
        calibration_frames = max(1, CALIBRATION_MS // FRAME_MS)
        warmup_frames = max(1, WARMUP_MS // FRAME_MS)
        pre_roll_frames = max(1, PRE_ROLL_MS // FRAME_MS)
        high_pass = HighPass(SAMPLE_RATE, self.config.high_pass_hz)

        history: list[bytes] = []
        captured: list[bytes] = []
        started = False
        frames_read = 0
        max_frames = int(limit_s * 1000 / FRAME_MS)

        def read_frame() -> bytes | None:
            """Read exactly one frame of audio, or None at the end."""
            nonlocal frames_read
            if frames_read >= max_frames:
                return None
            assert stream is not None
            parts = []
            remaining = FRAME_BYTES
            while remaining > 0:
                chunk = stream.read(remaining)
                if not chunk:
                    break
                parts.append(chunk)
                remaining -= len(chunk)
            if not parts:
                return None
            frames_read += 1
            return b"".join(parts)

        try:
            # 1. let the stream settle, then listen to the room so the
            #    threshold fits it
            for _ in range(warmup_frames):
                if read_frame() is None:
                    return None
            for _ in range(calibration_frames):
                frame = read_frame()
                if frame is None:
                    return None
                frame = high_pass.process(frame)
                calibration.append(level_of(frame))
                history.append(frame)

            detector.calibrate(calibration)

            # 2. wait for speech, keeping a little of the run up
            while not started:
                frame = read_frame()
                if frame is None:
                    return None
                frame = high_pass.process(frame)
                history.append(frame)
                if len(history) > pre_roll_frames:
                    history.pop(0)
                if detector.feed(level_of(frame)) == "start":
                    captured = list(history)
                    started = True

            # 3. capture until the speaker stops
            while True:
                frame = read_frame()
                if frame is None:
                    break
                frame = high_pass.process(frame)
                captured.append(frame)
                if detector.feed(level_of(frame)) == "end":
                    break
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        if started and captured:
            return self._utterance(captured)
        return None

    def _utterance(self, frames: list[bytes]) -> Utterance:
        samples = b"".join(frames)
        duration = len(samples) / 2 / SAMPLE_RATE
        return Utterance(samples=samples, sample_rate=SAMPLE_RATE, duration_s=duration)


class _WavReader:
    """Reads frames from a WAV file, so the pipeline can be tested offline."""

    def __init__(self, path: Path) -> None:
        self._path = path
        with wave.open(str(path), "rb") as handle:
            if handle.getframerate() != SAMPLE_RATE:
                raise ValueError(
                    f"{path} is {handle.getframerate()}Hz, it needs to be {SAMPLE_RATE}Hz"
                )
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ValueError(f"{path} needs to be one channel, 16 bit audio")
            self._data = handle.readframes(handle.getnframes())
        self._position = 0

    def read(self, size: int) -> bytes:
        if self._position >= len(self._data):
            raise EndOfSource(f"{self._path} has no more audio")
        chunk = self._data[self._position : self._position + size]
        self._position += len(chunk)
        return chunk
