"""Small audio helpers built on the standard library."""

from __future__ import annotations

import io
import math
import wave
from array import array

SAMPLE_WIDTH = 2  # signed 16 bit
MAX_AMPLITUDE = 32768.0


class HighPass:
    """A one pole high pass filter that takes the rumble out of a microphone.

    Laptop microphones pick up desk vibration, fans and handling noise, and
    most of that energy sits below 100Hz. Left in, it drowns out speech,
    confuses the voice detector and gives whisper.cpp a bad time. Filtering it
    out fixes all three at once.
    """

    def __init__(self, sample_rate: int = 16000, cutoff: float = 100.0) -> None:
        rc = 1.0 / (2.0 * math.pi * cutoff)
        self._alpha = rc / (rc + 1.0 / sample_rate)
        self._previous_sample = 0.0
        self._previous_output = 0.0

    def process(self, samples: bytes) -> bytes:
        """Filter a chunk of signed 16 bit PCM, keeping the filter state."""
        values = array("h")
        values.frombytes(samples[: len(samples) - (len(samples) % SAMPLE_WIDTH)])
        if not values:
            return b""

        alpha = self._alpha
        previous_sample = self._previous_sample
        previous_output = self._previous_output
        output = array("h")

        for value in values:
            sample = value / MAX_AMPLITUDE
            filtered = alpha * (previous_output + sample - previous_sample)
            previous_sample = sample
            previous_output = filtered
            scaled = int(max(-1.0, min(1.0, filtered)) * 32767)
            output.append(scaled)

        self._previous_sample = previous_sample
        self._previous_output = previous_output
        return output.tobytes()


def level_of(samples: bytes) -> float:
    """Root mean square of a signed 16 bit little endian frame, 0.0 to 1.0."""
    if not samples:
        return 0.0
    values = array("h")
    values.frombytes(samples[: len(samples) - (len(samples) % SAMPLE_WIDTH)])
    if not values:
        return 0.0
    total = 0
    for value in values:
        total += value * value
    return math.sqrt(total / len(values)) / MAX_AMPLITUDE


def clipped_fraction(samples: bytes) -> float:
    """How much of a frame is pinned at full scale, 0.0 to 1.0.

    A microphone whose gain is too high flattens the loudest parts of speech.
    Whisper cannot recover what was clipped, so this is worth surfacing.
    """
    if not samples:
        return 0.0
    values = array("h")
    values.frombytes(samples[: len(samples) - (len(samples) % SAMPLE_WIDTH)])
    if not values:
        return 0.0
    clipped = sum(1 for value in values if value <= -32767 or value >= 32767)
    return clipped / len(values)


def to_wav(samples: bytes, sample_rate: int) -> bytes:
    """Wrap raw signed 16 bit PCM in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(samples)
    return buffer.getvalue()


def write_wav(path: str, samples: bytes, sample_rate: int) -> None:
    """Write raw signed 16 bit PCM to a WAV file."""
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(samples)
