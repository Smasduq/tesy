#!/usr/bin/env python3
"""Record a fixed microphone clip and show exactly what Whisper hears.

This separates the three things that can make a transcript wrong:

1. the microphone (quiet, clipped, noisy, wrong device)
2. the voice detector cutting words off
3. the speech model mishearing clean audio

Run it, say the sentence it shows, and read the result. ``--file`` replays a
saved 16 kHz mono WAV instead of using the microphone.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tesy.config import load_env_file  # noqa: E402
from tesy.voice.audio import HighPass, clipped_fraction, level_of, to_wav  # noqa: E402
from tesy.voice.config import (  # noqa: E402
    FRAME_MS,
    SAMPLE_RATE,
    VoiceConfig,
    resolve_voice_config,
)
from tesy.voice.recorder import (  # noqa: E402
    CALIBRATION_MS,
    FRAME_BYTES,
    WARMUP_MS,
    EndOfSource,
    Recorder,
    Utterance,
    VoiceActivityDetector,
)
from tesy.voice.stt import TranscriptionError, WhisperTranscriber  # noqa: E402

SENTENCE = "The quick brown fox jumps over the lazy dog."


def capture(config: VoiceConfig, seconds: float) -> bytes:
    """Record raw 16 kHz mono audio for a fixed number of seconds."""
    wanted = int(SAMPLE_RATE * 2 * seconds)
    command = [
        "pw-cat",
        "--record",
        "--raw",
        "--rate",
        str(SAMPLE_RATE),
        "--channels",
        "1",
        "--format",
        "s16",
        "--sample-count",
        str(wanted // 2),
    ]
    if config.mic_source:
        command.append(f"--target={config.mic_source}")
    command.append("-")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    chunks: list[bytes] = []
    received = 0
    try:
        assert process.stdout is not None
        while received < wanted:
            chunk = process.stdout.read(min(65536, wanted - received))
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    return b"".join(chunks)[:wanted]


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{path} is {handle.getframerate()} Hz; it needs to be {SAMPLE_RATE} Hz"
            )
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path} needs one channel of 16 bit audio")
        return handle.readframes(handle.getnframes())


def analyse(config: VoiceConfig, samples: bytes, clip: Path) -> str:
    """Print levels and what the real recorder keeps from this clip."""
    if not samples:
        return "No audio was captured."

    high_pass = HighPass(SAMPLE_RATE, config.high_pass_hz)
    frames = [
        high_pass.process(samples[index : index + FRAME_BYTES])
        for index in range(0, len(samples) - FRAME_BYTES + 1, FRAME_BYTES)
    ]
    levels = [level_of(frame) for frame in frames]
    if not levels:
        return "The clip is too short to analyse."

    ordered = sorted(levels)
    at = lambda fraction: ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]
    duration = len(samples) / 2 / SAMPLE_RATE
    lines = [
        f"duration:       {duration:.2f} s ({len(frames)} frames)",
        f"level (RMS):    p10={at(0.10):.4f}  p50={at(0.50):.4f}  "
        f"p90={at(0.90):.4f}  p99={at(0.99):.4f}  max={ordered[-1]:.4f}",
        f"clipped:        {clipped_fraction(samples):.2%} of samples",
    ]

    warmup_frames = max(1, WARMUP_MS // FRAME_MS)
    calibration_frames = max(1, CALIBRATION_MS // FRAME_MS)
    detector = VoiceActivityDetector(
        threshold=config.threshold,
        silence_ms=config.silence_ms,
        onset_ms=config.onset_ms,
        min_speech_ms=config.min_speech_ms,
    )
    detector.calibrate(levels[warmup_frames : warmup_frames + calibration_frames])
    lines.append(f"VAD threshold:  {detector.effective_threshold:.4f}")

    # Replay the clip through the real recorder so the numbers include the
    # pre-roll and calibration replay instead of a second guess at them.
    recorder = Recorder(replace(config, mic_source=f"file:{clip}"))
    try:
        utterance = recorder.record()
    except EndOfSource:
        utterance = None

    if utterance is None:
        lines.append("recorder:       no speech found")
        return "\n".join(lines)

    lines.append(
        f"recorder:       kept {utterance.duration_s:.2f} s of {duration:.2f} s "
        f"(silence {config.silence_ms} ms, onset {config.onset_ms} ms)"
    )
    lines.append(segment_text(config, utterance.samples, "recorder"))
    return "\n".join(lines)


def segment_text(config: VoiceConfig, samples: bytes, label: str, model: Path | None = None) -> str:
    """Transcribe one slice of audio and describe how long it took."""
    chosen = model or config.stt_model
    transcriber = WhisperTranscriber(replace(config, stt_model=chosen))
    utterance = Utterance(
        samples=samples,
        sample_rate=SAMPLE_RATE,
        duration_s=len(samples) / 2 / SAMPLE_RATE,
    )
    started = time.perf_counter()
    try:
        text = transcriber.transcribe(utterance)
    except TranscriptionError as error:
        return f"{label} ({chosen.name}): ERROR {error}"
    elapsed = time.perf_counter() - started
    heard = text if text else "(nothing)"
    return f"{label} ({chosen.name}, {elapsed:.1f}s): {heard}"


def find_models(config: VoiceConfig, requested: list[str] | None) -> list[Path]:
    """Turn model names from the command line into paths next to the default."""
    if not requested:
        return [config.stt_model]
    directory = config.stt_model.parent
    models = []
    for name in requested:
        path = Path(name)
        if not path.is_absolute():
            filename = name if name.endswith(".bin") else f"ggml-{name}.bin"
            path = directory / filename
        models.append(path)
    return models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record a microphone clip and show what Whisper hears."
    )
    parser.add_argument("--seconds", type=float, default=8.0, help="clip length (default 8)")
    parser.add_argument("--file", type=Path, help="analyse an existing 16 kHz mono WAV")
    parser.add_argument("--mic", help="record from this PipeWire source instead of the default")
    parser.add_argument(
        "--models",
        nargs="+",
        help="transcribe with these models too, e.g. base.en small.en-q5_1",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="where to save the recording (default: <voice dir>/stt-check.wav)",
    )
    args = parser.parse_args(argv)

    try:
        load_env_file()
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    config = resolve_voice_config()
    if args.mic:
        config = replace(config, mic_source=args.mic)

    if args.file:
        samples = read_wav(args.file)
        source = str(args.file)
    else:
        print(f"microphone: {config.mic_source}")
        print(f"model:      {config.stt_model.name}")
        print(f'\nSay this sentence:\n  "{SENTENCE}"\n')
        for number in (3, 2, 1):
            print(f"  {number}...", flush=True)
            time.sleep(1)
        print(f"  recording {args.seconds:.0f}s, speak now.", flush=True)
        samples = capture(config, args.seconds)
        source = "microphone"

    save = args.save or (config.voice_dir / "stt-check.wav")
    try:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_bytes(to_wav(samples, SAMPLE_RATE))
    except OSError as error:
        print(f"could not save the clip: {error}", file=sys.stderr)
        save = Path(tempfile.mkdtemp(prefix="tesy-stt-")) / "clip.wav"
        save.write_bytes(to_wav(samples, SAMPLE_RATE))

    print(f"\nsource: {source}\n")
    print(analyse(config, samples, save))
    print(f"\nsaved: {save}")

    print("\nwhole clip:")
    for model in find_models(config, args.models):
        if not model.exists():
            print(f"  whisper ({model.name}): model not found at {model}")
            continue
        print("  " + segment_text(config, samples, "whisper", model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
