"""Phase 1 command line.

  python run.py preview  --input chapter.txt
  python run.py generate --input chapter.txt --out out/ch8 [--minutes 10] [--mock-tts] [--script-only]
  python run.py render   --script out/ch8/script.json --out out/ch8 [--mock-tts]
"""
import argparse
import json
import sys
from pathlib import Path

from render import Renderer, RenderError
from script_builder import ScriptError, build_script
from utils.clean import prepare_sections
from utils.config import load_settings
from utils.providers import ProviderError, build_llm, build_tts


def _read_sections(path: str) -> list:
    raw = Path(path).read_text(encoding="utf-8")
    sections = prepare_sections(raw)
    if not sections:
        sys.exit("The input file has no usable text.")
    return sections


def cmd_preview(args) -> None:
    sections = _read_sections(args.input)
    total = sum(s["words"] for s in sections)
    print(f"Found {len(sections)} section(s), {total:,} words:")
    for s in sections:
        print(f"  {s['id']:>8}  {s['words']:>5} words  {s['title']}")


def _render(script: dict, out: Path, settings, mock: bool) -> None:
    tts, voices = build_tts(settings, mock=mock)
    cache = Path(".cache") / ("tts-mock" if mock else "tts")
    renderer = Renderer(tts, voices, settings.tts_language, cache,
                        max_chars=settings.tts_max_chars, workers=settings.tts_workers)
    manifest = renderer.render(script, out)
    minutes, seconds = divmod(int(manifest["duration"]), 60)
    print(f"Done: {out / 'episode.mp3'} ({minutes}:{seconds:02d}), captions and manifest.json written.")
    if manifest["needs_review"]:
        print("Note: some lines are flagged in script.json -> unresolved_issues. Review before publishing.")


def cmd_generate(args) -> None:
    settings = load_settings()
    sections = _read_sections(args.input)
    print(f"Read {len(sections)} section(s), {sum(s['words'] for s in sections):,} words.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    script = build_script(
        sections, build_llm(settings), minutes=args.minutes, audience=args.audience,
        language=settings.script_language, host_a=settings.host_a_name, host_b=settings.host_b_name,
    )
    script["source_sections"] = sections  # kept so every episode can be traced back to its input
    (out / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Script saved: {out / 'script.json'}")
    if not args.script_only:
        _render(script, out, settings, args.mock_tts)


def cmd_render(args) -> None:
    settings = load_settings()
    script = json.loads(Path(args.script).read_text(encoding="utf-8"))
    _render(script, Path(args.out), settings, args.mock_tts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn a pasted textbook chapter into a two-host podcast.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preview", help="show how the pasted text was split into sections (no API calls)")
    p.add_argument("--input", required=True)
    p.set_defaults(func=cmd_preview)

    g = sub.add_parser("generate", help="chapter text -> script -> audio + captions")
    g.add_argument("--input", required=True, help="text file containing the pasted chapter")
    g.add_argument("--out", required=True)
    g.add_argument("--minutes", type=int, default=10)
    g.add_argument("--audience", default="Indian school students")
    g.add_argument("--mock-tts", action="store_true", help="use offline beeps instead of real TTS")
    g.add_argument("--script-only", action="store_true", help="stop after writing script.json")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("render", help="existing script.json -> audio + captions")
    r.add_argument("--script", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--mock-tts", action="store_true")
    r.set_defaults(func=cmd_render)

    args = parser.parse_args()
    try:
        args.func(args)
    except (ProviderError, ScriptError, RenderError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
