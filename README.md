[한국어](README.ko.md)

# ask-apple-photos

A command-line tool that finds photos in a macOS Photos library from a natural-language question. It can also run a local web UI on the same machine.

Examples:

```bash
.venv/bin/python ask.py "저번달쯤 강원도 카페에서 찍은 사진"
.venv/bin/python ask.py "양식"
```

The tool copies information the Photos app already has (capture date and time, place, scene labels, place types, face count, text in photos) into a local table at `index/cards.sqlite`. It gives that table to Claude as tools, and Claude searches it to answer the question. Photos themselves never leave the machine during the question step. To search by photo content, you can opt in to a caption step that sends previews to Google Gemini and stores a one-line description. That step is off by default.

**Questions and labels are tuned for Korean.** The index vocabulary (place names, scene labels, cuisine types) is mostly Korean, and the example questions are Korean. English questions work, but they are matched against Korean terms.

## What works / what does not

- **macOS only.** A Photos library (`.photoslibrary`) must exist.
- **Windows and Linux are not supported.** The tool reads Photos app internal files and uses Apple Vision.
- **A plain image folder is not supported.** It only reads a library imported into the Photos app.

## Requirements

| Item | Details |
|---|---|
| macOS | A machine with a Photos library |
| Python 3.12 | The `python3.12` command must be available |
| Xcode command line tools | The local enrichment step compiles a small Vision helper with `swiftc`. Run `xcode-select --install` |
| Anthropic API key | Used only for questions. Environment variable `ANTHROPIC_API_KEY` |
| Google Gemini API key | Used only when captions are enabled. Environment variable `GEMINI_API_KEY` |

### Permissions

If the terminal cannot read the Photos library, indexing can stop partway. In that case, add the terminal app (for example Terminal or iTerm) to **System Settings → Privacy & Security → Full Disk Access**.

Background execution such as `launchd` can **stop without an error** when this permission is missing. When setting up automation, run the tool once to completion in a terminal first, and put a short timeout on the background run.

## Quick start

Run these in order from a copy of the repository.

```bash
git clone <repository URL> ask-apple-photos
cd ask-apple-photos

python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1) Build the index — read the Photos library and create index/cards.sqlite
.venv/bin/python index/build_index.py

# 2) Local enrichment (free, no network) — public Vision classifiers and OCR
.venv/bin/python index/enrich_vision.py

# 3) Ask
export ANTHROPIC_API_KEY=...
.venv/bin/python ask.py "저번달쯤 강원도 카페에서 찍은 사진"

# 4) Optional: captions — send previews to Google and store scene descriptions
export GEMINI_API_KEY=...
.venv/bin/python index/enrich_caption.py                 # explains what would be sent, exit code 2
.venv/bin/python index/enrich_caption.py --send-previews # actually sends

# 5) Optional: local web UI
export PHOTO_ASK_WEB_PASSWORD="$(openssl rand -hex 16)"
echo "$PHOTO_ASK_WEB_PASSWORD"
.venv/bin/uvicorn --factory web.app:create_app --host 127.0.0.1 --port 4100 --no-proxy-headers
# Browser: http://127.0.0.1:4100  (user photo, password printed above)
```

`--no-proxy-headers` stops uvicorn from replacing the client address with `X-Forwarded-For`, so the per-IP rate limits cannot be bypassed with a forged header.

If the library is in a different location, pass `--library "/path/Photos Library.photoslibrary"` or set the `PHOTO_ASK_LIBRARY` environment variable.

Measured on 6,093 photos and videos (macOS 27.0.1, one Mac): the index took 21–24 seconds, Vision classification 62–73 seconds, and OCR 37–51 seconds (ranges from three runs under the same conditions). Later runs update only changed photos.

Cards for photos deleted from the Photos app are removed, but if a single run would remove more than 5% of existing cards, the run keeps them and only warns, because the copy may have been incomplete. Pass `--allow-mass-delete` to allow it.

## Commands

### Index

```bash
.venv/bin/python index/build_index.py [--library PATH] [--index FILE] [--rebuild]
                                      [--allow-mass-delete] [--no-preview-cache]
```

- The library is **never modified.** The tool copies the internal database to a temporary directory and opens the copy.
- Hidden photos are excluded from search by default.
- The run also creates Photos scene confidence data (`label_scores`) and a preview cache for the web UI (`index/previews/`).
- Some Photos internal files (`database/search/psi.sqlite`, `leo.sqlite`, the scene taxonomy) may be missing or named differently depending on the macOS version. The tool warns and continues indexing.

### Local enrichment (free, no network)

```bash
.venv/bin/python index/enrich_vision.py [--step classify|ocr|all] [--force] [--limit N] [--report]
.venv/bin/python index/sync_previews.py [--index FILE] [--dir DIR] [--include-hidden] [--dry-run] [--copy]
```

- `enrich_vision.py` runs Apple's public Vision classifiers and Korean OCR through a small Swift helper. The helper is compiled with `swiftc` on first use; if `swiftc` is missing, the script explains and exits.
- No photos leave the machine in this step.
- In a sandbox (an execution environment with restricted file access), OCR may start failing after the first photo. Run it outside the sandbox.
- `sync_previews.py` creates preview copies for the web UI. Run it in a terminal that can read the library.

### Ask

```bash
.venv/bin/python ask.py "question" [--interactive] [--limit N] [--open] [--sheet]
                                   [--show-paths] [--model NAME] [--effort low|medium|high] [--index FILE]
```

- When there are more than 50 candidates, the model asks one clarifying question. Pass `--interactive` to answer it in place.
- `--open` opens preview **copies** with `open`; it never opens the originals in the library.
- `--sheet` writes a contact sheet at `out/<query id>/index.html`.
- The result table identifies photos by the first 8 characters of the UUID. Pass `--show-paths` to include original paths.
- Call records go to `logs/llm.jsonl`, and full conversations go to `logs/transcripts/<query id>.json`.

### Captions (optional; photos leave the machine)

```bash
.venv/bin/python index/enrich_caption.py                 # explanation only, no network, exit code 2
.venv/bin/python index/enrich_caption.py --report        # statistics only, no network
.venv/bin/python index/enrich_caption.py --reparse       # rebuild fields from stored raw text, no network
.venv/bin/python index/enrich_caption.py --send-previews # actually sends
#   Additional options: --model --budget --concurrency --limit --retry-failed
#                       --skip-hidden/--no-skip-hidden  --skip-screenshots/--no-skip-screenshots
```

- Without `--send-previews`, **no photos are sent.** The script prints what would be sent and exits with code 2.
- Hidden photos are not sent by default.
- If the run is interrupted, running it again skips photos that already have captions (resume).
- To stop a run, create the file `logs/caption.stop`.
- The question step still works when there are zero captions. The summary and tools say captions are missing and search by scene labels, Vision labels, and text in photos instead.

### Web UI (local machine only)

```bash
export PHOTO_ASK_WEB_PASSWORD="$(openssl rand -hex 16)"
echo "$PHOTO_ASK_WEB_PASSWORD"
.venv/bin/uvicorn --factory web.app:create_app --host 127.0.0.1 --port 4100 --no-proxy-headers
```

- `--no-proxy-headers` stops uvicorn from replacing the client address with `X-Forwarded-For`, so the per-IP rate limits cannot be bypassed with a forged header.
- With this command the server listens on `127.0.0.1` only. Do not expose it on another address.
- The user is `photo` and the password is `PHOTO_ASK_WEB_PASSWORD`. The server refuses to start if the password is missing or shorter than 16 characters.
- Rate limiting uses the client address visible on the socket. It does not trust `X-Real-Ip` or `X-Forwarded-For`.
- The server does not open the Photos library directly; it shows only the `index/previews/` cache.
- Sessions live in memory and expire 30 minutes after last use.

## What leaves your Mac

| Step | Destination | Content | Photos |
|---|---|---|---|
| Index / local enrichment | none | — | 0 |
| Question | Anthropic | question text, library summary (counts, vocabulary, date range), tool results (counts, dates, places, labels, one-line caption scenes) | **0** |
| Caption (only with `--send-previews`) | Google Gemini | one preview re-saved as a pixels-only JPEG, plus a fixed prompt | 1 per request |

- The question step does not send photos (`photo_ask/agent.py:3`). Tool results do not include full text from photos or file paths.
- The caption step must be **enabled explicitly**. With a free-tier Gemini key, inputs may be used to improve the product. Check whether sending photos is acceptable, and prefer a paid-tier key.
- Since v0.1.1, the caption step decodes each preview and sends a new JPEG containing only its pixels (`photo_ask/jpeg_meta.py`). GPS coordinates, capture time, device model, and other file metadata are not carried over. A photo is not sent if this fails. What is visible in the photo itself, such as faces, signs, and documents, is still sent.
- Re-saving slightly reduces image quality and removes the color profile, so wide-gamut photos can look slightly different to the model.
- **v0.1.0 sent previews with this metadata.** If you ran captions with v0.1.0, any coordinates, capture times, and device models stored in those previews went to Google.
- API keys are read from environment variables only. There is no feature that writes them to a file.

## Cost

Price tables and limits are in the code (`photo_ask/config.py:28`, `photo_ask/config.py:45`). Costs are estimates calculated from those price tables. Before a new call, the tool compares the cost already recorded in `logs/llm.jsonl` with the limit; a call already in flight, or a retry of a failed request, can push the total slightly over the limit.

| Environment variable | Default | Meaning |
|---|---|---|
| `PHOTO_ASK_BUDGET_USD` | 3.0 | Cumulative limit for questions (USD) |
| `PHOTO_ASK_QUERY_BUDGET_USD` | 0.25 | Limit for one question (USD) |
| `PHOTO_ASK_CAPTION_BUDGET_USD` | 6.0 | Cumulative limit for captions (USD) |

Measured on 2026-11-06 (macOS 27.0.1, one Mac, 6,093 photos and videos; the same 4 questions run three times — 12 runs — and 3 captions run three times):

- Index 21–24 seconds, Vision classification 62–73 seconds, OCR 37–51 seconds.
- One question: 5–13 seconds, $0.011–$0.051.
- Gemini caption: 3 photos $0.00236, about $0.0008 per photo.

These ranges come from a small sample of three runs under the same conditions, and other runs can fall outside them. Costs are estimates calculated from the price table and have not been reconciled with a bill. The amount depends on the model and prices used.

## Environment variables

| Name | Default | Used by |
|---|---|---|
| `ANTHROPIC_API_KEY` | none (required) | Question |
| `GEMINI_API_KEY` | none (required when captions are on) | Caption |
| `PHOTO_ASK_LIBRARY` | `~/Pictures/Photos Library.photoslibrary` | Index |
| `PHOTO_ASK_TZ` | `Asia/Seoul` | Question dates and times |
| `PHOTO_ASK_INDEX` | `index/cards.sqlite` | Everything |
| `PHOTO_ASK_LOG_DIR` | `logs/` | Everything |
| `PHOTO_ASK_PREVIEW_CACHE` | `index/previews/` | Web UI |
| `PHOTO_ASK_MODEL` | `claude-sonnet-5-5` | Question |
| `PHOTO_ASK_EFFORT` | `low` | Question |
| `PHOTO_ASK_BUDGET_USD` | `3.0` | Question |
| `PHOTO_ASK_QUERY_BUDGET_USD` | `0.25` | Question |
| `PHOTO_ASK_MAX_TURNS` | `10` | Question |
| `PHOTO_ASK_CAPTION_MODEL` | `gemini-3.1-flash-lite` | Caption |
| `PHOTO_ASK_CAPTION_BUDGET_USD` | `6.0` | Caption |
| `PHOTO_ASK_WEB_PASSWORD` | none (required, minimum 16 characters) | Web UI |

## Limits

- **Retrieval accuracy has not been measured.** There is no number for what fraction of correct photos are returned.
- The question agent is an LLM, so asking the same question again can return a different number of candidates (in our tests, running the same question twice did produce differences).
- Questions, labels, and captions are Korean-based.
- The tool reads Photos internal files (`Photos.sqlite`, `database/search/leo.sqlite`, the scene taxonomy), so a macOS update may break it.
- There is no face-name indexing. Only the **count** of faces is stored.
- OCR runs at Photos preview resolution, so small text can be missed.
- The web UI shows preview copies, not original quality.
- Editing a photo does not automatically refresh a caption that was already generated.
- Accuracy has not been compared with other tools. Only macOS 27.0.1 was tested; other macOS versions have not been tested.

## Design notes

`docs/DESIGN.md` (Korean) explains the three-stage structure, why the tools are designed this way, and which problems came up and how they were solved.

## Tests

Tests do not open a real library and do not need personal photos or API keys. Most use fake cards and fake clients; `tests/test_build_enrich.py:200` also reads the macOS scene taxonomy when that file is present, and is skipped otherwise.

```bash
env -u ANTHROPIC_API_KEY -u GEMINI_API_KEY .venv/bin/python -m unittest discover -s tests
```

## License

MIT. See `LICENSE`.
