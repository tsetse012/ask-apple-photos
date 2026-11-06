# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-06

### Added

- Local index of a macOS Photos library (`index/cards.sqlite`) built from Photos metadata, scene labels, place data, face counts, and text in photos.
- Local enrichment with Apple Vision classifiers and Korean OCR, with no network access.
- Natural-language question answering over the index with Claude using three tools: `search_cards`, `stats`, and `list_vocab`.
- Optional caption step that sends one preview JPEG per request to Google Gemini. It is disabled unless `--send-previews` is passed.
- Preview cache and a local-only web UI bound to `127.0.0.1`.
- Cumulative and per-question cost limits, plus call logs in `logs/llm.jsonl` and transcripts in `logs/transcripts/`.
- Test suite that runs without a Photos library or API keys.
