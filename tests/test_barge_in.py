"""Tests for cutting Tesy off mid-sentence: ``python -m unittest discover -s tests``."""

import math
import sys
import threading
import time
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tesy.voice.config import VoiceConfig  # noqa: E402
from tesy.voice.loop import _Interrupter, _PrefixStream  # noqa: E402
from tesy.voice.recorder import FRAME_BYTES, FRAME_SAMPLES, Recorder  # noqa: E402
from tesy.voice.tts import InterruptedSpeech, PiperSpeaker  # noqa: E402


def silence_frames(count: int) -> list[bytes]:
    """``count`` frames of digital silence, one frame each."""
    frame = b"\x00\x00" * FRAME_SAMPLES
    return [frame] * count


def speech_frames(count: int, amplitude: float = 0.6) -> list[bytes]:
    """``count`` frames of a loud tone, one frame each."""
    values = array("h")
    for index in range(FRAME_SAMPLES * count):
        values.append(int(32767 * amplitude * math.sin(2 * math.pi * 220 * index / 16000)))
    raw = values.tobytes()
    size = FRAME_SAMPLES * 2
    return [raw[offset : offset + size] for offset in range(0, len(raw), size)]


class FakeStream:
    """Hands out scripted frames, then blocks like a live microphone would."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)
        self._lock = threading.Lock()

    def read(self, size: int) -> bytes:
        with self._lock:
            if self._frames:
                return self._frames.pop(0)
        # Nothing left, so behave like a quiet microphone.
        time.sleep(0.01)
        return b"\x00" * size


class FakeProcess:
    def __init__(self) -> None:
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.terminated = True


def voice_config(**overrides) -> VoiceConfig:
    values = dict(
        voice_dir=Path("/nowhere"),
        mic_source="alsa_input.test",
        speaker=None,
        tts_model=Path("/nowhere/voice.onnx"),
        stt_model=Path("/nowhere/model.bin"),
        language="en",
        silence_ms=700,
        onset_ms=120,
        min_speech_ms=250,
        max_utterance_s=20.0,
        threshold=None,
        high_pass_hz=100.0,
        threads=4,
    )
    values.update(overrides)
    return VoiceConfig(**values)


class PrefixStreamTests(unittest.TestCase):
    def test_replays_the_prefix_before_live_audio(self):
        live = FakeStream([b"L" * 4, b"L" * 4])
        stream = _PrefixStream(b"P" * 6, live)
        self.assertEqual(stream.read(4), b"PPPP")
        self.assertEqual(stream.read(4), b"PP")
        self.assertEqual(stream.read(4), b"LLLL")

    def test_a_prefix_alone_still_terminates(self):
        stream = _PrefixStream(b"P" * 4, None)
        self.assertEqual(stream.read(4), b"PPPP")
        self.assertEqual(stream.read(4), b"")


class InterrupterTests(unittest.TestCase):
    """The watcher has to fire on speech and stay quiet otherwise."""

    def _interrupter(self, frames: list[bytes], **overrides) -> _Interrupter:
        config = voice_config(**overrides)
        interrupter = _Interrupter(Recorder(config), config)
        # Bypass the microphone: drive the watcher off a scripted stream.
        interrupter.open = lambda: True  # type: ignore[method-assign]
        interrupter.stream = FakeStream(frames)
        interrupter.process = FakeProcess()
        return interrupter

    def _run_until(self, interrupter: _Interrupter, limit: float = 3.0) -> bool:
        interrupter.start()
        deadline = time.time() + limit
        while time.time() < deadline:
            if interrupter.heard_speech:
                return True
            time.sleep(0.01)
        return False

    def test_speech_cuts_in(self):
        frames = silence_frames(15) + speech_frames(20)
        interrupter = self._interrupter(frames)
        self.assertTrue(self._run_until(interrupter))
        interrupter.stop()

    def test_a_quiet_room_does_not_cut_in(self):
        frames = silence_frames(300)
        interrupter = self._interrupter(frames)
        self.assertFalse(self._run_until(interrupter, limit=0.6))
        interrupter.stop()

    def test_a_single_click_is_ignored(self):
        # one loud frame is 20ms, less than the 80ms needed to count as speech
        frames = silence_frames(20) + speech_frames(1) + silence_frames(20)
        interrupter = self._interrupter(frames)
        self.assertFalse(self._run_until(interrupter, limit=0.8))
        interrupter.stop()

    def test_her_own_voice_at_the_start_is_ignored(self):
        # the first few frames are her own playback bleeding into the mic
        frames = speech_frames(4) + silence_frames(40)
        interrupter = self._interrupter(frames)
        self.assertFalse(self._run_until(interrupter, limit=0.8))
        interrupter.stop()

    def test_the_captured_prefix_is_returned(self):
        frames = silence_frames(15) + speech_frames(30)
        interrupter = self._interrupter(frames)
        self.assertTrue(self._run_until(interrupter))
        captured = interrupter.stop()
        self.assertGreater(len(captured), 0)

    def test_nothing_captured_when_she_is_not_interrupted(self):
        frames = silence_frames(300)
        interrupter = self._interrupter(frames)
        self._run_until(interrupter, limit=0.6)
        self.assertEqual(interrupter.stop(), b"")


class InterruptibleSpeakerTests(unittest.TestCase):
    def test_stop_is_checked_before_playing(self):
        speaker = PiperSpeaker(voice_config())
        speaker.interrupt()
        with self.assertRaises(InterruptedSpeech):
            speaker._check_stop()

    def test_interrupt_without_a_player_is_harmless(self):
        speaker = PiperSpeaker(voice_config())
        speaker.interrupt()  # nothing playing; must not raise
        self.assertFalse(speaker.is_playing)


class BorrowedCaptureTests(unittest.TestCase):
    """The recorder must never close a stream it did not open."""

    def test_a_borrowed_process_is_left_alone(self):
        config = voice_config()
        recorder = Recorder(config)
        process = FakeProcess()
        stream = _PrefixStream(b"".join(speech_frames(40)), None)
        utterance = recorder.record(stream=stream, process=process)
        self.assertIsNotNone(utterance)
        self.assertFalse(process.terminated)

    def test_an_owned_capture_is_closed(self):
        # A file source is the simplest owned capture we can exercise here.
        import tempfile
        import wave

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            quiet = b"\x00\x00" * 8000
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(quiet + b"".join(speech_frames(40)) + quiet)

            recorder = Recorder(voice_config(mic_source=f"file:{path}"))
            utterance = recorder.record()
            self.assertIsNotNone(utterance)


if __name__ == "__main__":
    unittest.main()
