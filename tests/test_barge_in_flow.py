"""End to end tests for cutting off a spoken reply."""

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tesy.voice.loop import speak_stream  # noqa: E402
from tesy.voice.tts import InterruptedSpeech  # noqa: E402


class FakeSpeaker:
    """Records what was asked to be spoken, without making any sound."""

    def __init__(self, per_sentence: float = 0.05) -> None:
        self.spoken: list[str] = []
        self.per_sentence = per_sentence
        self._lock = threading.Lock()
        self._playing = False
        self.interrupted = False

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def synthesize(self, text: str) -> bytes:
        return text.encode("utf-8")

    def play(self, audio: bytes) -> None:
        if self.interrupted:
            raise InterruptedSpeech()
        with self._lock:
            self._playing = True
            self.spoken.append(audio.decode("utf-8"))
        try:
            time.sleep(self.per_sentence)
        finally:
            with self._lock:
                self._playing = False

    def interrupt(self) -> None:
        self.interrupted = True


class FakeInterrupter:
    def __init__(self) -> None:
        self.heard_speech = False


def reply(*sentences: str):
    """A streamed reply, one token at a time."""
    for sentence in sentences:
        for word in sentence.split(" "):
            yield word + " "
        yield ""


class SpeakStreamTests(unittest.TestCase):
    def test_speaks_every_sentence_when_not_interrupted(self):
        speaker = FakeSpeaker()
        speak_stream(reply("One.", "Two.", "Three."), speaker, show_text=False)
        self.assertEqual(len(speaker.spoken), 3)
        self.assertFalse(speaker.interrupted)

    def test_a_late_interrupter_returns_false(self):
        speaker = FakeSpeaker()
        interrupter = FakeInterrupter()
        # Never fires: heard_speech stays False.
        interrupted = speak_stream(
            reply("One."), speaker, show_text=False, interrupter=interrupter
        )
        self.assertFalse(interrupted)

    def test_interrupting_stops_playback(self):
        speaker = FakeSpeaker(per_sentence=0.4)
        interrupter = FakeInterrupter()

        # Fire the interrupt shortly after speaking starts.
        def fire() -> None:
            time.sleep(0.15)
            interrupter.heard_speech = True

        threading.Thread(target=fire, daemon=True).start()
        interrupted = speak_stream(
            reply("One.", "Two.", "Three."), speaker, show_text=False, interrupter=interrupter
        )
        self.assertTrue(interrupted)
        self.assertTrue(speaker.interrupted)
        # The later sentences must not have been spoken.
        self.assertLess(len(speaker.spoken), 3)


if __name__ == "__main__":
    unittest.main()
