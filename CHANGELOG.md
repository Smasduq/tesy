# Changelog

All notable changes to Tesy are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `pyproject.toml`, so Tesy installs with `pip install .` and exposes a `tesy`
  command.
- GitHub Actions workflow that runs the test suite on Python 3.10-3.13.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`,
  `CHANGELOG.md`, and issue/PR templates.
- MIT licence.

## [0.2.0] - 2026-10-22

### Added

- Voice conversation mode: local whisper.cpp transcription, the chat model,
  and local piper speech, with sentence-by-sentence streaming.
- Barge-in, so speaking while Tesy talks cuts her off and starts a new turn.
- Adaptive noise tracking, high-pass filtering, and voice activity detection
  tuned for real microphones.
- Bluetooth headset support that prefers the wideband mSBC profile.
- PipeWire echo-cancellation instructions and device selection settings.
- `scripts/install_voice.sh`, a no-root installer for whisper.cpp, piper, and
  the base English speech model.
- `scripts/stt_check.py` for diagnosing microphone and transcription problems.
- Optional debug capture of each spoken turn as a WAV file.
- Read-only file tools for the home directory: `read_file`,
  `list_directory`, and `search_files`, with protected credential paths.
- File tools for the Snowflake Cortex provider.
- Persona editing through `persona.md`, `TESY_SYSTEM`, and `--system`.
- A Captain-aware default persona that matches the user's tone.
- Markdown and URL stripping before text reaches the voice.
- `--show-reasoning` and `--reasoning-effort` for reasoning models.

### Changed

- The default OpenRouter model is the free `minimax/minimax-m3.1-flash:free`.
- The speech detector calibrates while listening instead of guessing a level.
- The base English whisper model replaced the tiny model as the default.

### Fixed

- Snowflake Cortex payload format.
- Speech that starts before calibration is no longer lost.
- `pw-cat` no longer adds a header to captured audio.
- Whisper noise annotations are filtered out of transcripts.

## [0.1.0] - 2026-10-18

### Added

- First release: OpenRouter and local Ollama providers, streaming chat,
  interactive mode, `.env` loading, and the `persona.md` system prompt.
- Snowflake Cortex provider.

[Unreleased]: https://github.com/Smasduq/tesy/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Smasduq/tesy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Smasduq/tesy/releases/tag/v0.1.0
