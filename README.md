# NCERT chapter → podcast (Phase 1)

Paste a chapter, get a two-host podcast: `episode.mp3`, `episode.vtt` / `episode.srt` captions, and a `manifest.json` with per-line timestamps for the interactive player (Phase 2).

The model never uses its own memory of NCERT books. Every stage only sees the pasted text, and a verify pass checks each script line against that text before any audio is made.

## Setup

```bash
pip install -r requirements.txt        # just `requests`
# ffmpeg must be installed: sudo apt install ffmpeg  /  brew install ffmpeg
cp .env.example .env                   # then fill in keys, model, voices
```

Keys stay in `.env` on the server. Never put them in frontend code or push `.env` to GitHub.

## Try it without spending credit

```bash
python run.py render --script sample_script.json --out out/sample --mock-tts
```

This stitches a sample script using beeps instead of real voices (higher beep = host A), so you can check caption timing in any player that loads `.vtt` files.

## Real run

```bash
python run.py preview  --input chapter.txt                 # check section detection, no API calls
python run.py generate --input chapter.txt --out out/ch8 --script-only   # LLM only; read script.json first
python run.py render   --script out/ch8/script.json --out out/ch8        # then spend TTS credit
```

Or all at once: `python run.py generate --input chapter.txt --out out/ch8 --minutes 10`.

## Providers

- **TTS:** Bodhan Indic-Speak, `POST https://api.bodhan.ai/v1/audio/speech` with model `indic-speak`, a required `voice` (e.g. `Kavya`, `Arun`; any voice reads any language) and the language in `instructions` as a JSON string (`{"lang": "en"}`). Returns WAV. Needs a Bodhan key from the Bodhan dashboard.
- **LLM:** any OpenAI-compatible chat endpoint (we use OpenRouter). Bodhan has no general chat model, so the script writer can't be a Bodhan model.

## Files

| File | Job |
|---|---|
| `clean.py` | Strips page numbers, running headers, reprint footers; splits into sections |
| `prompts.py` | All prompts (extract, plan, script, verify, fix) with shared grounding rules |
| `script_builder.py` | Runs the LLM stages, validates JSON, fixes flagged lines |
| `render.py` | Per-line TTS with caching, stitching, loudness normalisation, captions, manifest |
| `providers.py` | LLM + Bodhan TTS clients, plus an offline mock TTS |
| `run.py` | Command line |

If lines are still flagged after two fix rounds, they are listed under `unresolved_issues` in `script.json` and `manifest.json` has `"needs_review": true`. Have a teacher look before publishing.

Attribution: the Indic-Speak open-model license asks apps built on it to credit "Built with Indic-Speak from Bodhan AI / AI4Bharat". Worth adding to your player's footer.
