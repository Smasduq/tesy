"""Keep one piper voice in memory and synthesise sentences on demand.

Loading a piper model takes several seconds. Doing that for every sentence
makes a voice chatbot feel broken, so this helper stays alive and does the
loading once.

It speaks a tiny line based protocol on stdin and stdout:

    request   one JSON object per line: {"text": "hello"}
    response  one JSON header line:     {"ok": true, "bytes": 1234, "rate": 22050}
              followed by exactly "bytes" bytes of raw signed 16 bit PCM

Run it with the virtualenv that has piper installed:

    ~/.local/share/tesy/venv/bin/python piper_worker.py voice.onnx
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: piper_worker.py MODEL", file=sys.stderr)
        return 2

    from piper import PiperVoice

    output = sys.stdout.buffer
    voice = PiperVoice.load(sys.argv[1])
    rate = voice.config.sample_rate

    output.write(json.dumps({"ready": True, "rate": rate}).encode("utf-8") + b"\n")
    output.flush()

    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            request = {}

        if request.get("cmd") == "quit":
            break

        text = request.get("text", "")
        audio = b""
        try:
            audio = b"".join(
                chunk.audio_int16_bytes for chunk in voice.synthesize(text)
            )
            header = {"ok": True, "bytes": len(audio), "rate": rate}
        except Exception as error:  # keep serving after a bad request
            header = {"ok": False, "error": str(error)}

        output.write(json.dumps(header).encode("utf-8") + b"\n")
        output.write(audio)
        output.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
