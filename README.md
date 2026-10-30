# NCERT chapter → podcast (Phase 1)

Paste a chapter, get a two-host podcast: `episode.mp3`, `episode.vtt` / `episode.srt` captions, and a `manifest.json` with per-line timestamps for the interactive player (Phase 2).

The model never uses its own memory of NCERT books. Every stage only sees the pasted text, and a verify pass checks each script line against that text before any audio is made.

Pipeline: extract facts (each with a verbatim quote from the chapter) → plan segments (each with a why/how question and an optional misconception to refute) → write → edit the whole episode for flow and length → verify (free rule checks plus an LLM judge) → minimal fixes → fall back to the chapter's own words → make `spoken_text` digit-free for TTS → synthesise, trim, level, stitch, caption.

## Setup

```bash
pip install -r requirements.txt        # just `requests`
# ffmpeg must be installed: sudo apt install ffmpeg  /  brew install ffmpeg
cp .env.example .env                   # then fill in keys, model, voices
```

Keys stay in `.env` on the server. Never put them in frontend code or push `.env` to GitHub.

## Try it without spending credit

```bash
python run.py render --script examples/sample_script.json --out out/sample --mock-tts
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

| Path | Job |
|---|---|
| `run.py` | Command line |
| `script_builder.py` | Runs the LLM stages, validates JSON, rule checks, fixes and fallbacks |
| `prompts.py` | All prompts (extract, plan, script, edit, verify, fix, normalise) with shared grounding rules |
| `render.py` | Per-line TTS with caching, silence trim and levelling, duration QC with retakes, turn gaps, loudness, captions, manifest |
| `utils/clean.py` | Strips page numbers, running headers, reprint footers; splits into sections |
| `utils/config.py` | Reads settings from `.env` or the environment |
| `utils/providers.py` | LLM + Bodhan TTS clients, plus an offline mock TTS |
| `examples/sample_script.json` | Small script for the offline mock render |
| `docs/NOTES.md` | What each research idea changed in the pipeline, with results on our test chapter |
| `docs/research_review.md` | The literature review behind the design |
| `tests/` | 42 tests, no network needed |

If lines are still flagged after two fix rounds, they are listed under `unresolved_issues` in `script.json` and `manifest.json` has `"needs_review": true`. Have a teacher look before publishing.

Attribution: the Indic-Speak open-model license asks apps built on it to credit "Built with Indic-Speak from Bodhan AI / AI4Bharat". Worth adding to your player's footer.

## Research basis

`docs/research_review.md` reviews the literature behind these choices. Implemented: verbatim fact quotes (Attribute First), "report what the author says" framing (Zhou et al.), exact-match number checks and a link/phone filter, per-kind verification with refuted misconceptions (Muller et al., Driscoll et al.), minimal-edit fixes (RARR) with a quote fallback (Dialog Inpainting), reasoning only where judging, spoken-text normalisation (PolyNorm), a voice allow-list, clip levelling, natural turn gaps, duration QC with retakes, and two-line captions timed by spoken length.

Not yet, each needs a decision: a local HHEM/MiniCheck checker (needs PyTorch), ASR listen-back with Bodhan Indic-Transcribe (needs its key), word-level forced alignment (PyTorch), prompt-prefix caching (LLM cost is already a few cents per episode), and quizzes / Ask the hosts (out of scope for now; `manifest.json` already carries `kind`, `fact_ids` and fact quotes for them).
