# Tesy

Your co-worker and friend. Never lets you get lonely.

Tesy chats over a pluggable backend, so the model provider can be swapped
without touching the app. It defaults to **OpenRouter** on a **free model**,
and can switch to a **locally fine-tuned model** or **Snowflake Cortex** with
a single flag.

## Quick start

```bash
# The API key already lives in .env (gitignored).
python main.py "hey, what are we working on today?"
python -m tesy --interactive          # keep the conversation going
```

The `.env` file in the repo root is loaded automatically. Any variable already
present in your shell wins over the file.

## Providers

| Provider   | Flag                   | Credentials                                  |
| ---------- | ---------------------- | -------------------------------------------- |
| OpenRouter | `--provider openrouter` | `OPENROUTER_API_KEY`                         |
| Local      | `--provider local`      | Ollama running locally (`OLLAMA_MODEL`)       |
| Snowflake  | `--provider snowflake` | `SNOWFLAKE_ACCOUNT_URL` + `SNOWFLAKE_PAT`     |

OpenRouter is the default. Switching is a one-liner:

```bash
python main.py --provider local "hello"      # your fine-tuned model
python main.py --provider snowflake "hello"  # Snowflake Cortex
export TESY_PROVIDER=local                   # make it stick for a shell
```

## Free models

The default model is `minimax/minimax-m3.1-flash:free` - no credits, no
spending. Your key allows 50 free-model requests per day.

```bash
python main.py --list-models --free    # every free model on OpenRouter
python main.py --model nvidia/nemotron-3.5-lightning:free "hello"
```

A few free models are gated to approved agent harnesses and answer `403` if
called directly - the list above shows what is listed, not what is callable.

### Reasoning models

Nemotron, DeepSeek and friends stream their thinking before the answer. Two
flags keep them in check:

```bash
python main.py --reasoning-effort none --model nvidia/nemotron-3.5-lightning:free "hi"
python main.py --show-reasoning "hi"    # print the thinking to stderr
```

Without `--reasoning-effort none` a reasoning model can burn your whole
token budget before answering.

## Customising Tesy

`persona.md` is the system prompt. Edit the part after the `---` separator and
Tesy's whole manner changes on the next run - no code, no cost, no training.

```bash
python main.py --system "you are terse"      # one-off
TESY_SYSTEM="..." python main.py "hello"      # environment
```

## Files

Tesy can look at files under your home directory. Three read-only tools are
offered to the model:

- `read_file` - a text file, with line numbers and an optional line range
- `list_directory` - what is in a folder
- `search_files` - find a phrase inside text files

Nothing can be written, moved or deleted. Credential stores are refused
outright: `.ssh`, `.gnupg`, `.aws`, `.azure`, `.docker`, `.kube`, browser
profiles, password managers, `.env`, `*.pem`, `*.key`, `id_*`, `.netrc` and
similar. Symlinks that leave the home directory are not followed, binaries are
skipped, and reads are capped at 128 KB. Every lookup is printed so you can
see it happen:

```
  computer: read ~/Documents/notes.txt (lines 1-40)
```

The content Tesy reads is sent to the model. With the default OpenRouter
backend that means it leaves this machine, so do not point Tesy at files you
would not share with that provider. Turn the whole feature off with:

```bash
python main.py --no-computer "hello"
TESY_COMPUTER=off python main.py --talk
```

File tools are only offered to providers that support tool calling, which is
OpenRouter today.

## Fine-tuning

OpenRouter cannot fine-tune models - there is no fine-tuning API. The free
route is a free Colab GPU, Unsloth, and Ollama, which Tesy already talks to.
Full walkthrough in **[docs/FINE_TUNING.md](docs/FINE_TUNING.md)**.

```bash
ollama create tesy ./your-finetuned.Q4_K_M.gguf
python main.py --provider local "hello"
python main.py --list-local-models
```

## Voice

Tesy can hold a spoken conversation: your microphone, on-device transcription,
the same model as the text path, and a local voice speaking back. Everything
runs offline except the language model, so the free tier of OpenRouter is the
only cost and it is zero.

```bash
scripts/install_voice.sh --check     # what is already installed
scripts/install_voice.sh             # fetch and set up the voice stack
python main.py --talk                # press Enter or Ctrl-D to stop
python main.py "say that again" --speak   # one reply, spoken
```

`--talk` prints what it heard and then streams the answer to the voice
sentence by sentence, so speech starts before the model has finished writing.
Markdown is stripped before it reaches the voice: `*emphasis*` is spoken as
the word, links as their label, and URLs are skipped instead of spelled out.

The installer is a one-time download of about 150 MB and needs no root. It
puts everything under `~/.local/share/tesy/`:

- **Transcription** - [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
  with `ggml-base.en.bin`, which holds up much better on a real microphone
  than the tiny model. Set `TESY_STT_MODEL` to the `ggml-tiny.en.bin` path if
  you would rather have the lower latency.
- **Speech** - [piper](https://github.com/rhasspy/piper) with the
  `en_US-amy-medium` voice, run by a persistent worker so the model loads once
  instead of on every sentence.
- **Capture and playback** - PipeWire's `pw-cat` and `pw-play`.

### Settings

| Variable                | Default                          | What it does                                |
| ----------------------- | -------------------------------- | ------------------------------------------- |
| `TESY_MIC_SOURCE`       | first input device               | PipeWire source name, or `file:clip.wav`     |
| `TESY_SPEAKER`          | first output device              | PipeWire sink name                           |
| `TESY_VOICE_DIR`        | `~/.local/share/tesy`            | Where the installer put everything           |
| `TESY_STT_MODEL`        | `.../models/ggml-base.en.bin`    | Whisper model                                |
| `TESY_TTS_MODEL`        | `.../voices/en_US-amy-medium.onnx` | Piper voice                                |
| `TESY_STT_LANGUAGE`     | `en`                             | Whisper language                             |
| `TESY_HIGH_PASS_HZ`     | `100`                            | Cuts laptop rumble before speech detection   |
| `TESY_VAD_SILENCE_MS`   | `700`                            | Pause long enough to end a turn              |
| `TESY_VAD_ONSET_MS`     | `120`                            | Speech needed before a turn starts           |
| `TESY_VAD_MIN_SPEECH_MS`| `250`                            | Shorter noises are ignored                   |
| `TESY_MAX_UTTERANCE_S`  | `20`                             | Safety cap on a single turn                  |
| `TESY_VAD_THRESHOLD`    | adaptive                         | Fixed level instead of noise tracking        |
| `TESY_STT_THREADS`      | `4`                              | Whisper threads                              |
| `TESY_DEBUG_AUDIO`      | off                              | Directory to save each captured turn as WAV  |

The microphone is only listened to while the voice is silent, so the half-duplex
loop never transcribes Tesy's own speech.

### Bluetooth headsets

A headset on the A2DP music profile has no usable microphone. Tesy asks
PipeWire to switch the card to a headset profile on start. It asks for mSBC
(16 kHz) first, because the older CVSD codec is only 8 kHz and makes
transcription noticeably worse. Output is narrowband while `--talk` is
running. Unset `TESY_MIC_SOURCE` to let it pick the default device, or point
it at the laptop microphone and keep music quality:

```bash
TESY_MIC_SOURCE=alsa_input.pci-0000_00_1f.3.analog-stereo python main.py --talk
```

If a headset only offers CVSD, Tesy still uses it rather than giving up, but
that is the 8 kHz path and words will be missed more often. Pointing
`TESY_MIC_SOURCE` at the laptop microphone is usually more accurate there.

### When transcription is wrong

Set a debug directory and each turn is saved as a WAV with its level and
clipping shown:

```bash
TESY_DEBUG_AUDIO=~/tesy-audio python main.py --talk
```

That file is exactly what whisper was given, so it separates a microphone
problem (quiet, clipped, distorted) from a model problem. The usual causes,
in order: the wrong microphone, a headset on CVSD, speaking before the
stream settles, and the tiny model on a noisy room.

For a controlled test, `scripts/stt_check.py` records eight seconds with no
voice detection, prints the level and clipping, shows the part of the clip the
detector would keep, and transcribes both the whole clip and that part:

```bash
python scripts/stt_check.py
```

It asks you to say "The quick brown fox jumps over the lazy dog." If the whole
clip is right but the detector's part is cut, raise `TESY_VAD_SILENCE_MS` (1000
is a good start). If the whole clip is wrong, try the other microphone:

```bash
python scripts/stt_check.py --mic alsa_input.pci-0000_00_1f.3.analog-stereo
```

Compare the saved WAVs by ear as well as by transcript. To see whether the
model is the limit, transcribe the same clip with more than one model:

```bash
python scripts/stt_check.py --file ~/.local/share/tesy/stt-check.wav \
    --models base.en small.en-q5_1
```

`ggml-small.en-q5_1.bin` is more accurate than base on accents and noise, but
on a CPU without a GPU it takes about 12 seconds for a five second clip, so it
is opt-in rather than the default. Download it into `models/` and set
`TESY_STT_MODEL` to use it.

If a Bluetooth headset is the microphone, remember that its headset profile is
narrowband. The laptop's built-in microphone is often clearer for dictation.

### Trying it without a microphone

`file:` sources replay a 16 kHz mono WAV through the whole pipeline, which is
handy for testing and for demos:

```bash
TESY_MIC_SOURCE=file:/tmp/opencode/mixed.wav python main.py --talk
```

## Options

```
--provider {openrouter,local,snowflake}  Chat backend (default: TESY_PROVIDER or openrouter)
--model MODEL                            Model to use (default: the provider's default model)
--system TEXT                            System prompt (default: TESY_SYSTEM, persona.md, or built-in)
--max-tokens N                           Maximum tokens per reply (default: 1024)
--temperature FLOAT                      Sampling temperature (provider default when omitted)
--reasoning-effort {none,minimal,low,medium,high}
                                         How much a reasoning model should think first
--show-reasoning                          Print the model's reasoning to stderr
-i, --interactive                        Keep the conversation going
--talk                                   Talk to Tesy with your voice
--speak                                  Speak the reply out loud
--no-tts                                 With --talk, listen and answer in text only
--no-computer                            Do not let Tesy read files under your home
--list-models                            List the models OpenRouter offers
--free                                   With --list-models, show only the free ones
--list-local-models                      List the models on the local Ollama server
```

Model selection follows the same order everywhere: the `--model` flag, then
the provider-specific environment variable (`OPENROUTER_MODEL` /
`OLLAMA_MODEL` / `SNOWFLAKE_MODEL`), then the provider default.

## Adding a provider

A backend is one module in `tesy/providers/` that exposes a class with a
`name`, a `default_model`, `resolve_model()` and a streaming `stream()`
method. Register it in `PROVIDERS` in `tesy/providers/__init__.py` and it
becomes available as `--provider <name>` everywhere else in the app.

## Tests

```bash
python -m unittest discover -s tests
```

## Roadmap

Speech in and out is local and free. The next step is a fine-tuned model
(see above) so Tesy sounds like itself rather than a general assistant, and a
daily request counter for the 50-request free tier.
