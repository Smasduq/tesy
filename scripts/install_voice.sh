#!/usr/bin/env bash
# Install Tesy's local voice stack: piper (text to speech) and whisper.cpp (speech to text).
#
# Everything lands in ~/.local/share/tesy (override with TESY_VOICE_DIR).
# Nothing here needs a GPU, and nothing calls a paid API.
#
#   ./scripts/install_voice.sh          # install everything
#   ./scripts/install_voice.sh --check  # report what is present and what is missing
#
set -euo pipefail

PREFIX="${TESY_VOICE_DIR:-$HOME/.local/share/tesy}"
VENV="$PREFIX/venv"
VOICES="$PREFIX/voices"
MODELS="$PREFIX/models"
BIN="$PREFIX/bin"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PIPER_WHEEL_VERSION="1.8.0"
WHISPER_TAG="b5454"
WHISPER_ASSET="whisper-bin-ubuntu-x64.tar.gz"
VOICE_ID="en_US-amy-medium"
VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"

say() { printf '\033[1;36m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$1"; }
die() { printf '\033[1;31mxx\033[0m %s\n' "$1" >&2; exit 1; }

# Downloads sometimes drop halfway. Give every fetch a few tries.
fetch() {
    local url="$1"
    local output="$2"
    local attempt=1
    while [ "$attempt" -le 4 ]; do
        if curl -fL --progress-bar -C - -o "$output" "$url"; then
            return 0
        fi
        attempt=$((attempt + 1))
        warn "download failed, retrying ($attempt of 4)"
        sleep 3
    done
    return 1
}

check() {
    local missing=0
    say "checking $PREFIX"
    if [ -x "$VENV/bin/piper" ]; then
        echo "    piper      ok ($VENV/bin/piper)"
    else
        echo "    piper      MISSING"; missing=1
    fi
    if [ -f "$VOICES/$VOICE_ID.onnx" ]; then
        echo "    voice      ok ($VOICE_ID)"
    else
        echo "    voice      MISSING ($VOICE_ID)"; missing=1
    fi
    if [ -x "$BIN/whisper-cli" ] || [ -x "$BIN/main" ]; then
        echo "    whisper    ok ($BIN)"
    else
        echo "    whisper    MISSING ($BIN)"; missing=1
    fi
    if [ -f "$PREFIX/piper_worker.py" ]; then
        echo "    worker     ok ($PREFIX/piper_worker.py)"
    else
        echo "    worker     MISSING ($PREFIX/piper_worker.py)"; missing=1
    fi
    for model in tiny.en base.en; do
        if [ -f "$MODELS/ggml-$model.bin" ]; then
            echo "    stt model  ok (ggml-$model.bin)"
        else
            echo "    stt model  missing (ggml-$model.bin)"; missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        warn "something is missing, run: $0"
        return 1
    fi
    say "everything is in place"
    return 0
}

if [ "${1:-}" = "--check" ]; then
    check
    exit $?
fi

mkdir -p "$PREFIX" "$VOICES" "$MODELS" "$BIN"

# --- python environment ------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
    say "creating a virtualenv at $VENV"
    python3 -m venv "$VENV" || die "could not create the virtualenv (is python3-venv installed?)"
fi

say "installing piper (text to speech)"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "piper-tts==${PIPER_WHEEL_VERSION}" || die "could not install piper-tts"
"$VENV/bin/piper" --help >/dev/null 2>&1 || warn "the piper CLI did not answer, continuing anyway"

say "installing the speech helper"
cp "$HERE/piper_worker.py" "$PREFIX/piper_worker.py" || die "could not install the speech helper"

# --- voice ------------------------------------------------------------------
if [ ! -f "$VOICES/$VOICE_ID.onnx" ]; then
    say "downloading the voice: $VOICE_ID"
    fetch "$VOICE_URL/$VOICE_ID.onnx" "$VOICES/$VOICE_ID.onnx.part" \
        && mv "$VOICES/$VOICE_ID.onnx.part" "$VOICES/$VOICE_ID.onnx" \
        || die "voice download failed"
    fetch "$VOICE_URL/$VOICE_ID.onnx.json" "$VOICES/$VOICE_ID.onnx.json" || die "voice config download failed"
fi

# --- speech to text ---------------------------------------------------------
if [ ! -f "$BIN/libwhisper.so.1" ]; then
    say "downloading whisper.cpp ($WHISPER_TAG)"
    tmp="$(mktemp -d)"
    fetch "https://github.com/ggml-org/whisper.cpp/releases/download/$WHISPER_TAG/$WHISPER_ASSET" "$tmp/whisper.tar.gz" \
        || die "whisper.cpp download failed"
    tar xzf "$tmp/whisper.tar.gz" -C "$tmp"
    find "$tmp" -type f \( -name 'whisper-cli' -o -name 'main' \) -exec cp {} "$BIN/" \;
    # -L turns the library symlinks into real files so they never dangle
    find "$tmp" -name '*.so*' -exec cp -L {} "$BIN/" \;
    chmod +x "$BIN"/whisper-cli "$BIN"/main 2>/dev/null || true
    rm -rf "$tmp"
    [ -x "$BIN/whisper-cli" ] || [ -x "$BIN/main" ] || die "no whisper binary found in the release archive"
fi

for model in tiny.en base.en; do
    target="$MODELS/ggml-$model.bin"
    [ -f "$target" ] && continue
    say "downloading the speech model: ggml-$model.bin"
    fetch "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$model.bin" "$target.part" \
        && mv "$target.part" "$target" \
        || warn "could not download ggml-$model.bin"
done

check
