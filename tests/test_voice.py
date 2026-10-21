"""Tests for the voice layer: ``python -m unittest discover -s tests``."""

import math
import os
import struct
import sys
import tempfile
import unittest
import wave
from array import array
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tesy.voice import audio, sentences  # noqa: E402
from tesy.voice.config import (  # noqa: E402
    VoiceConfig,
    _best_mic_profile,
    resolve_voice_config,
)
from tesy.voice.recorder import (  # noqa: E402
    EndOfSource,
    Recorder,
    VoiceActivityDetector,
    _WavReader,
)
from tesy.voice.stt import clean_transcript  # noqa: E402
from tesy.voice.tts import PiperSpeaker  # noqa: E402


def tone(samples: int, amplitude: float, sample_rate: int = 16000) -> bytes:
    """A sine wave, as signed 16 bit little endian PCM."""
    values = array("h")
    for index in range(samples):
        value = amplitude * math.sin(2 * math.pi * 220 * index / sample_rate)
        values.append(int(value * 32767))
    return values.tobytes()


def env(values: dict[str, str]):
    """Set environment variables for the duration of a test."""

    class _Context:
        def __enter__(self):
            self.previous = {key: os.environ.get(key) for key in values}
            os.environ.update(values)

        def __exit__(self, *args):
            for key, value in self.previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            return False

    return _Context()


class LevelTests(unittest.TestCase):
    def test_silence_is_zero(self):
        self.assertEqual(audio.level_of(b"\x00\x00" * 320), 0.0)
    def test_empty_is_zero(self):
        self.assertEqual(audio.level_of(b""), 0.0)

    def test_full_scale_is_about_one(self):
        samples = struct.pack("<320h", *([32767] * 320))
        self.assertAlmostEqual(audio.level_of(samples), 1.0, places=3)

    def test_quiet_tone_is_quieter_than_loud_tone(self):
        quiet = audio.level_of(tone(320, 0.05))
        loud = audio.level_of(tone(320, 0.8))
        self.assertLess(quiet, loud)
        self.assertGreater(quiet, 0.0)

    def test_odd_byte_is_dropped(self):
        # two bytes make one sample, the stray third byte is ignored
        expected = abs(struct.unpack("<h", b"\x01\x02")[0]) / 32768
        self.assertAlmostEqual(audio.level_of(b"\x01\x02\x03"), expected, places=5)


class ClippingTests(unittest.TestCase):
    def test_clean_audio_is_not_clipped(self):
        self.assertEqual(audio.clipped_fraction(tone(320, 0.5)), 0.0)

    def test_full_scale_audio_is_clipped(self):
        samples = struct.pack("<320h", *([32767] * 320))
        self.assertEqual(audio.clipped_fraction(samples), 1.0)

    def test_estimates_the_share_of_clipped_samples(self):
        samples = struct.pack("<10h", *([0, 0, 0, 0, 0, 0, 0, 0, 32767, 32767]))
        self.assertAlmostEqual(audio.clipped_fraction(samples), 0.2)


class HighPassTests(unittest.TestCase):
    def test_removes_low_frequency_rumble(self):
        rumble = tone(16000, 0.5, sample_rate=16000)
        # a 30Hz rumble instead of 220Hz
        values = array("h")
        for index in range(16000):
            values.append(int(32767 * 0.5 * math.sin(2 * math.pi * 30 * index / 16000)))
        rumble = values.tobytes()

        filtered = audio.HighPass(16000, 100.0).process(rumble)
        before = audio.level_of(rumble)
        after = audio.level_of(filtered)
        self.assertLess(after, before * 0.5)

    def test_keeps_speech_frequencies(self):
        filtered = audio.HighPass(16000, 100.0).process(tone(16000, 0.5))
        self.assertGreater(audio.level_of(filtered), audio.level_of(tone(16000, 0.5)) * 0.7)

    def test_keeps_its_state_between_chunks(self):
        filter = audio.HighPass(16000, 100.0)
        whole = filter.process(tone(3200, 0.5))
        pieces = b"".join(filter.process(tone(320, 0.5)) for _ in range(10))
        self.assertEqual(len(whole), len(pieces))

    def test_empty_input_is_harmless(self):
        self.assertEqual(audio.HighPass().process(b""), b"")


class WavTests(unittest.TestCase):
    def test_wav_has_a_valid_header(self):
        data = audio.to_wav(tone(320, 0.5), 16000)
        self.assertTrue(data.startswith(b"RIFF"))
        self.assertIn(b"WAVE", data[:16])
        self.assertEqual(len(data), 44 + 640)

    def test_write_wav_creates_a_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.wav"
            audio.write_wav(str(path), tone(320, 0.5), 16000)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 44)


class SentenceTests(unittest.TestCase):
    def test_splits_simple_sentences(self):
        self.assertEqual(
            sentences.split_sentences("Hello there. How are you?"),
            ["Hello there.", "How are you?"],
        )

    def test_keeps_abbreviations_intact(self):
        self.assertEqual(
            sentences.split_sentences("Dr. Smith called. He was late."),
            ["Dr. Smith called.", "He was late."],
        )

    def test_keeps_ellipsis_intact(self):
        self.assertEqual(
            sentences.split_sentences("Wait... that is odd. Really."),
            ["Wait... that is odd.", "Really."],
        )

    def test_handles_decimal_numbers(self):
        self.assertEqual(
            sentences.split_sentences("It costs 3.50 dollars. Cheap."),
            ["It costs 3.50 dollars.", "Cheap."],
        )

    def test_keeps_closing_quote_with_sentence(self):
        self.assertEqual(
            sentences.split_sentences('She said "yes." Then she left.'),
            ['She said "yes."', "Then she left."],
        )

    def test_keeps_a_trailing_partial(self):
        self.assertEqual(
            sentences.split_sentences("Done. And then"),
            ["Done.", "And then"],
        )


class SentenceBufferTests(unittest.TestCase):
    def test_yields_sentences_as_they_complete(self):
        buffer = sentences.SentenceBuffer()
        self.assertEqual(buffer.feed("Hello"), [])
        self.assertEqual(buffer.feed(" there. How"), ["Hello there."])
        self.assertEqual(buffer.feed(" are you?"), ["How are you?"])

    def test_flush_returns_the_remainder(self):
        buffer = sentences.SentenceBuffer()
        buffer.feed("Almost done")
        self.assertEqual(buffer.flush(), ["Almost done"])
        self.assertEqual(buffer.flush(), [])


class DetectorTests(unittest.TestCase):
    def _detector(self, **kwargs):
        return VoiceActivityDetector(threshold=0.05, **kwargs)

    def _feed_until(self, detector, level, wanted, limit=50):
        """Feed frames until the detector reports ``wanted``."""
        for _ in range(limit):
            event = detector.feed(level)
            if event == wanted:
                return event
        return ""

    def test_detects_the_start_of_speech(self):
        detector = self._detector(onset_ms=40)
        self.assertEqual(detector.feed(0.01), "")
        self.assertEqual(detector.feed(0.01), "")
        # two loud frames add up to the 40ms onset window
        self.assertEqual(detector.feed(0.5), "")
        self.assertEqual(detector.feed(0.5), "start")

    def test_detects_the_end_after_a_pause(self):
        detector = self._detector(onset_ms=40, silence_ms=100, min_speech_ms=40)
        self.assertEqual(self._feed_until(detector, 0.5, "start"), "start")
        self.assertEqual(self._feed_until(detector, 0.01, "end"), "end")

    def test_ignores_a_click_that_is_too_short(self):
        detector = self._detector(onset_ms=100)
        detector.feed(0.5)
        detector.feed(0.01)
        self.assertEqual(detector.feed(0.01), "")

    def test_calibrates_from_background_noise(self):
        detector = VoiceActivityDetector()
        detector.calibrate([0.001] * 30)
        self.assertGreater(detector.effective_threshold, 0.001)
        self.assertEqual(detector.feed(0.002), "")

    def test_an_explicit_threshold_is_kept(self):
        detector = VoiceActivityDetector(threshold=0.2)
        detector.calibrate([0.001] * 30)
        self.assertEqual(detector.effective_threshold, 0.2)

    def test_the_threshold_follows_a_noisy_room(self):
        detector = VoiceActivityDetector()
        detector.calibrate([0.01] * 30)
        quiet = detector.effective_threshold
        detector.calibrate([0.05] * 30)
        self.assertGreater(detector.effective_threshold, quiet)

    def test_calibrating_on_speech_does_not_deafen_the_detector(self):
        # someone was already talking during the calibration window
        detector = VoiceActivityDetector(onset_ms=40)
        detector.calibrate([0.2] * 40)
        self.assertLess(detector.effective_threshold, 0.2)
        self.assertEqual(self._feed_until(detector, 0.2, "start"), "start")


class TranscriptCleaningTests(unittest.TestCase):
    def test_removes_blank_audio_markers(self):
        self.assertEqual(clean_transcript("[BLANK_AUDIO]\nhello"), "hello")

    def test_drops_invented_silence_phrases(self):
        self.assertEqual(clean_transcript("Thank you."), "")

    def test_keeps_real_speech(self):
        self.assertEqual(clean_transcript(" what is the weather "), "what is the weather")

    def test_drops_invented_sound_descriptions(self):
        self.assertEqual(clean_transcript("(indistinct chatter)"), "")
        self.assertEqual(clean_transcript("[music] (upbeat music)"), "")
        self.assertEqual(clean_transcript("[BLANK_AUDIO]"), "")

    def test_keeps_speech_around_a_sound_description(self):
        self.assertEqual(clean_transcript("(sighs) not again"), "not again")
        self.assertEqual(clean_transcript("hello [laughs] there"), "hello there")

    def test_collapses_extra_spaces(self):
        self.assertEqual(clean_transcript("  hello   there  "), "hello there")


class WavReaderTests(unittest.TestCase):
    def _write_wav(self, folder: str, samples: bytes, rate: int = 16000) -> Path:
        path = Path(folder) / "audio.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(samples)
        return path

    def test_reads_frames_in_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_wav(folder, tone(1280, 0.5))
            reader = _WavReader(path)
            first = reader.read(640)
            second = reader.read(640)
            self.assertEqual(len(first), 640)
            self.assertNotEqual(first, second)

    def test_raises_when_the_file_runs_out(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_wav(folder, tone(320, 0.5))
            reader = _WavReader(path)
            reader.read(640)
            with self.assertRaises(EndOfSource):
                reader.read(640)

    def test_refuses_the_wrong_sample_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_wav(folder, tone(320, 0.5, sample_rate=22050), rate=22050)
            with self.assertRaises(ValueError):
                _WavReader(path)


class FileSourceTests(unittest.TestCase):
    def _config(self, path: Path) -> VoiceConfig:
        return VoiceConfig(
            voice_dir=Path("/nowhere"),
            mic_source=f"file:{path}",
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

    def test_recognises_a_file_source(self):
        config = self._config(Path("/tmp/whatever.wav"))
        self.assertTrue(Recorder(config).is_file_source)
        device = replace(config, mic_source="alsa_input.test")
        self.assertFalse(Recorder(device).is_file_source)

    def test_captures_speech_from_a_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            quiet = array("h", [0] * 16000)
            loud = array("h")
            for index in range(32000):
                loud.append(
                    int(20000 * math.sin(2 * math.pi * 220 * index / 16000))
                )
            samples = quiet.tobytes() + loud.tobytes() + quiet.tobytes()
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(samples)

            recorder = Recorder(self._config(path))
            utterance = recorder.record()
            self.assertIsNotNone(utterance)
            # the phrase is 2s; add the 400ms pre-roll and 700ms of silence
            self.assertGreater(utterance.duration_s, 2.5)
            self.assertLess(utterance.duration_s, 3.5)

    def test_speech_from_the_first_frame_is_heard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(tone(32000, 0.5))

            recorder = Recorder(self._config(path))
            utterance = recorder.record()
            self.assertIsNotNone(utterance)
            self.assertGreater(utterance.duration_s, 1.8)

    def test_a_clip_that_ends_mid_sentence_still_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            quiet = array("h", [0] * 24000).tobytes()
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(quiet + tone(16000, 0.5))

            recorder = Recorder(self._config(path))
            utterance = recorder.record()
            self.assertIsNotNone(utterance)
            self.assertGreater(utterance.duration_s, 1.0)
            with self.assertRaises(EndOfSource):
                recorder.record()

    def test_the_file_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(tone(320, 0.5))

            recorder = Recorder(self._config(path))
            with self.assertRaises(EndOfSource):
                recorder.record()
                recorder.record()


class SpeakerTests(unittest.TestCase):
    def _config(self) -> VoiceConfig:
        return VoiceConfig(
            voice_dir=Path("/nowhere"),
            mic_source=None,
            speaker="alsa_output.test",
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

    def test_reports_when_piper_is_missing(self):
        speaker = PiperSpeaker(self._config())
        self.assertFalse(speaker.available())

    def test_falls_back_to_the_command(self):
        speaker = PiperSpeaker(self._config())
        # no helper and no piper binary, so it cannot work
        with self.assertRaises(Exception):
            speaker.synthesize("hello")


class VoiceConfigTests(unittest.TestCase):
    def test_wideband_headset_profile_comes_first(self):
        profiles = ["headset-head-unit-cvsd", "headset-head-unit"]
        self.assertEqual(
            _best_mic_profile(profiles),
            ["headset-head-unit", "headset-head-unit-cvsd"],
        )

    def test_other_profiles_keep_their_order(self):
        self.assertEqual(_best_mic_profile([]), [])

    def test_defaults(self):
        with env({"TESY_VOICE_DIR": "/tmp/tesy-test", "TESY_TTS_MODEL": "", "TESY_STT_MODEL": ""}):
            config = resolve_voice_config()
        self.assertEqual(str(config.tts_model), "/tmp/tesy-test/voices/en_US-amy-medium.onnx")
        self.assertEqual(str(config.stt_model), "/tmp/tesy-test/models/ggml-base.en.bin")

    def test_environment_overrides(self):
        with env(
            {
                "TESY_VOICE_DIR": "/tmp/tesy-test",
                "TESY_TTS_MODEL": "/tmp/voice.onnx",
                "TESY_STT_MODEL": "/tmp/model.bin",
                "TESY_VAD_SILENCE_MS": "500",
                "TESY_MAX_UTTERANCE_S": "9.5",
                "TESY_HIGH_PASS_HZ": "80",
            }
        ):
            config = resolve_voice_config()
        self.assertEqual(str(config.tts_model), "/tmp/voice.onnx")
        self.assertEqual(str(config.stt_model), "/tmp/model.bin")
        self.assertEqual(config.silence_ms, 500)
        self.assertEqual(config.max_utterance_s, 9.5)
        self.assertEqual(config.high_pass_hz, 80)

    def test_missing_binaries_are_reported_as_none(self):
        config = VoiceConfig(
            voice_dir=Path("/nowhere"),
            mic_source=None,
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
        self.assertIsNone(config.whisper_binary)
        self.assertIsNone(config.piper_binary)


if __name__ == "__main__":
    unittest.main()
