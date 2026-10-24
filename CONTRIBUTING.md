# Contributing to Tesy

Thanks for wanting to make Tesy better. This is a small, dependency-light
Python project, and most changes stay that way.

## Before you start

- Open an issue for anything larger than a bug fix or a small improvement, so
  we can agree on the shape of the change before you write it.
- Keep pull requests focused. One change per PR is much easier to review and
  revert than five bundled together.
- Be kind in issues and reviews. Disagreement is fine; rudeness is not.

## Setting up

Tesy needs Python 3.11 or newer. There are no third-party Python packages for
the text path, so a virtual environment and an editable install are enough:

```bash
git clone https://github.com/Smasduq/tesy.git
cd tesy
python -m venv .venv
source .venv/bin/activate
```

Copy `.env.example` to `.env` and fill in the credentials for the provider you
want to use. Never commit `.env` or any other file containing real keys.

```bash
cp .env.example .env
python main.py "hello"
```

Voice is optional and has its own installer:

```bash
scripts/install_voice.sh --check
scripts/install_voice.sh
```

## Tests

Run the test suite before opening a pull request:

```bash
python -m unittest discover -s tests
```

Add a test for any behaviour you change. Tests should not call a real model
provider, touch the network, or need a microphone; stub the provider and use
temporary directories instead.

If you touch the voice pipeline, also try it by hand. The tests cannot hear:

```bash
python main.py --talk
python scripts/stt_check.py
```

## Style

- Follow the style of the file you are editing. The codebase prefers plain
  functions, small modules, and standard-library solutions over frameworks.
- Keep the public CLI documented in the README in sync with the flags you add
  or change in `tesy/cli.py`.
- User-facing text is spoken aloud by default. Keep it short, natural, and free
  of formatting that sounds wrong when read out.
- Prefer clear names and comments that explain *why*, not *what*.

## Adding a provider

A backend is one module in `tesy/providers/` exposing a class with:

- `name` - the value used by `--provider`
- `default_model` - the model used when none is configured
- `resolve_model()` - pick the effective model from flags and environment
- `stream()` - yield the reply as it arrives

Register the class in `PROVIDERS` in `tesy/providers/__init__.py`, add a test in
`tests/test_providers.py`, and document the provider and its environment
variables in the README.

## Security

Do not open a public issue for a vulnerability. Follow [SECURITY.md](SECURITY.md)
instead.

## Licence

By contributing, you agree that your contribution is licensed under the
[MIT License](LICENSE) that covers this project.
