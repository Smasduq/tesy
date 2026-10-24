# Security Policy

## Supported versions

Tesy is developed on `main`. Security fixes land there and are released from
there; older commits and forks are not supported.

## Reporting a vulnerability

Please do not open a public issue, pull request, or discussion for a security
problem.

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/Smasduq/tesy/security/advisories/new)
or by email to [sadiqulmasduq2010@gmail.com](mailto:sadiqulmasduq2010@gmail.com).

Include as much of the following as you can:

- what the issue is and what an attacker could do with it
- the affected file, command, or provider
- steps to reproduce, or a small proof of concept
- the version or commit you tested

You will get an acknowledgement within a few days. Please give us a reasonable
window to fix the issue before disclosing it publicly, and we will credit you
in the fix unless you ask us not to.

## Scope

Tesy runs locally and talks to a model provider using credentials you supply.
Useful reports include, but are not limited to:

- reads that escape the file-tool allowlist or its protected-path list
- command, path, or prompt injection that causes Tesy to read files it should
  refuse
- credential or `.env` leakage in logs, errors, or provider requests
- symlink or path-traversal tricks in the file tools or the voice installers
- unsafe handling of downloaded voice models or scripts
- vulnerabilities in the provider clients or the local voice worker

## Not in scope

- The model saying something wrong, unkind, or unhelpful. That is a prompt or
  persona issue, not a vulnerability.
- Data you deliberately send to a provider. Files read by Tesy are sent to the
  configured backend, so only point it at files you are willing to share.
- Vulnerabilities in third-party software such as Ollama, OpenRouter,
  Snowflake Cortex, whisper.cpp, piper, or PipeWire. Report those upstream.
- Denial of service against your own machine with your own key.
- Anything that requires an attacker to already have local shell access or
  your `.env` file.

## Handling secrets

- Keep keys in `.env` or your shell environment. `.env` is gitignored.
- Use a key with the smallest permissions that work, and rotate it if it is
  ever exposed.
- The file tools refuse known credential stores, but that is defence in depth,
  not a guarantee. Run Tesy only on files you trust.
