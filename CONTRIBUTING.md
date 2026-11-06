# Contributing

## Issues

Open an issue on GitHub with the macOS version, the Python version, and the exact command that failed. Include the error output as text.

Do not attach personal photos, a Photos library, database files, API keys, transcripts from `logs/`, or any other personal data. Reduce a problem to a description or a synthetic example instead.

## Pull requests

1. Fork the repository and create a branch.
2. Keep changes focused. Explain what changed and why in the pull request description.
3. Run the tests before opening the pull request:

```bash
env -u ANTHROPIC_API_KEY -u GEMINI_API_KEY .venv/bin/python -m unittest discover -s tests
```

4. Do not commit generated data: `.venv/`, `index/*.sqlite*`, `index/previews/`, `logs/`, `out/`, `__pycache__/`, or `tools/vision_helper`.

## Code changes

- The library is never modified. Any change that opens a path inside the Photos library at runtime must stop and use a copy or the preview cache instead.
- The question step must not send photos. The caption step must stay opt-in behind `--send-previews`.
- Add or update tests for behavior changes. Tests must run without a real Photos library and without API keys.
