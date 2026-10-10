"""Turns cleaned sections into a verified two-host script (script.json).

Stages: extract facts (with verbatim quotes) -> plan segments -> write each segment ->
edit the whole episode for flow and length -> verify against the source (free rule
checks plus an LLM judge) -> fix flagged lines -> fall back to the chapter's own words
-> make spoken_text TTS-safe. Verify runs after editing, so every line that reaches the
audio has been checked. Anything still flagged is kept in script["unresolved_issues"]
so a teacher can review it before publishing.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor

import prompts

WORDS_PER_MINUTE = 150
# Highest priority first: when one host's two lines merge, the merged line keeps the kind
# that gets checked hardest, so a fact never hides inside an unchecked misconception.
KINDS = ("refutation", "fact", "misconception", "recap", "analogy", "question", "banter")
MAX_FIX_ROUNDS = 2

NUMBER = re.compile(r"\d+(?:[.,:/]\d+)*")
CONTACT = re.compile(r"https?://|www\.|\b[\w.+-]+@[\w-]+\.\w|(?:\+91[\s-]?)?\b\d{10}\b", re.I)
# Anything a TTS voice may misread: digits in any script, sub/superscripts, maths and
# currency symbols, brackets, Greek letters, or the same word twice in a row.
SPOKEN_PROBLEM = re.compile(
    r"[\d\u00b2\u00b3\u00b9\u2070-\u209f+=×÷→←↑↓^/\\%₹$&@#*_~|<>\[\]{}()\u0370-\u03ff]|\b(\w+)\s+\1\b", re.I)


class ScriptError(RuntimeError):
    pass


def parse_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(cleaned[start:end + 1])


def ask_json(llm, system: str, user: str, key: str, temperature: float = 0.3, max_tokens: int = 4000,
             reasoning: bool = False) -> dict:
    """Calls the LLM and insists on a JSON object containing `key` (retries twice)."""
    prompt = user
    for _ in range(3):
        reply = llm.complete(system, prompt, temperature=temperature, max_tokens=max_tokens, reasoning=reasoning)
        try:
            data = parse_json(reply)
            if isinstance(data, dict) and isinstance(data.get(key), list):
                return data
        except ValueError:
            pass
        prompt = f'{user}\n\nYour previous reply was not a valid JSON object with a "{key}" list. Reply with the JSON object only.'
    stage = system.splitlines()[0]
    raise ScriptError(f"{stage}: the LLM did not return valid JSON after 3 attempts")


def _word_count(segments) -> int:
    return sum(len(line["display_text"].split()) for seg in segments for line in seg["lines"])


def _squash(text: str) -> str:
    return " " + " ".join(re.findall(r"\w+", text.lower())) + " "


def _extract(llm, sec: dict) -> list:
    """Fact sheet for one section. A fact keeps its quote only if the quote really occurs in
    the section, so every quote shown to students is the chapter's own words."""
    s, u = prompts.extract(sec["id"], sec["title"], sec["text"])
    source = _squash(sec["text"])
    facts = []
    for item in ask_json(llm, s, u, "facts", temperature=0.1)["facts"]:
        text = str(item.get("text") or "").strip() if isinstance(item, dict) else ""
        if not text:
            continue
        quote = str(item.get("quote") or "").strip()
        facts.append({
            "id": f"{sec['id']}-f{len(facts) + 1}",
            "section_id": sec["id"],
            "type": str(item.get("type") or "concept"),
            "text": text,
            "quote": quote if quote and _squash(quote) in source else "",
        })
    return facts


def _source_text(section_ids, section_index) -> str:
    return "\n\n".join(
        f"### Section {sid}: {section_index[sid]['title']}\n{section_index[sid]['text']}" for sid in section_ids
    )


def _normalise_plan(plan: dict, fact_index: dict, section_order: list) -> list:
    segments, used = [], set()
    for raw in plan.get("segments", []):
        if not isinstance(raw, dict):
            continue
        fact_ids = [f for f in raw.get("fact_ids") or [] if f in fact_index and f not in used]
        if not fact_ids:
            continue
        used.update(fact_ids)
        misconception = str(raw.get("misconception") or "").strip()
        refuting = str(raw.get("refuting_fact_id") or "").strip()
        if refuting not in fact_ids:  # a misconception needs a fact in this segment to refute it
            misconception = refuting = ""
        segments.append({
            "title": str(raw.get("title") or "Segment").strip(),
            "goal": str(raw.get("goal") or "").strip(),
            "deep_question": str(raw.get("deep_question") or "").strip(),
            "misconception": misconception,
            "refuting_fact_id": refuting if misconception else "",
            "fact_ids": fact_ids,
        })
    if not segments:
        raise ScriptError("The episode plan contained no usable segments")
    # Facts the planner forgot go to the segment already covering their section (or the last one).
    for fid, fact in fact_index.items():
        if fid in used:
            continue
        target = next(
            (s for s in segments if any(fact_index[f]["section_id"] == fact["section_id"] for f in s["fact_ids"])),
            segments[-1],
        )
        target["fact_ids"].append(fid)
    # Section ids are derived from the facts, not trusted from the planner.
    for i, seg in enumerate(segments, 1):
        seg["id"] = f"s{i}"
        ids = {fact_index[f]["section_id"] for f in seg["fact_ids"]}
        seg["section_ids"] = [sid for sid in section_order if sid in ids]
    return segments


def _normalise_lines(raw_lines, seg_id: str, fact_index: dict, host_b: str) -> list:
    lines = []
    for raw in raw_lines:
        if not isinstance(raw, dict):
            continue
        display = str(raw.get("display_text") or "").strip()
        spoken = str(raw.get("spoken_text") or "").strip() or display
        if not spoken:
            continue
        who = str(raw.get("speaker") or "A").strip().upper()
        speaker = "B" if who in ("B", host_b.upper()) else "A"
        kind = str(raw.get("kind") or "fact").strip().lower()
        fact_ids = [f for f in raw.get("fact_ids") or [] if f in fact_index]
        if lines and lines[-1]["speaker"] == speaker:  # one host twice in a row is one turn
            prev = lines[-1]
            prev["display_text"] += " " + (display or spoken)
            prev["spoken_text"] += " " + spoken
            prev["fact_ids"] += [f for f in fact_ids if f not in prev["fact_ids"]]
            if kind in KINDS:
                prev["kind"] = min(prev["kind"], kind, key=KINDS.index)
            continue
        lines.append({
            "id": f"{seg_id}-l{len(lines) + 1}",
            "speaker": speaker,
            "display_text": display or spoken,
            "spoken_text": spoken,
            "kind": kind if kind in KINDS else "fact",
            "fact_ids": fact_ids,
        })
    if not lines:
        raise ScriptError(f"Segment {seg_id} came back empty")
    return lines


def _apply_fixes(seg: dict, fixes: list) -> None:
    by_id = {line["id"]: line for line in seg["lines"]}
    delete = set()
    for fx in fixes:
        if not isinstance(fx, dict) or fx.get("line_id") not in by_id:
            continue
        if fx.get("action") == "delete":
            delete.add(fx["line_id"])
            continue
        line = by_id[fx["line_id"]]
        if fx.get("display_text"):
            line["display_text"] = str(fx["display_text"]).strip()
        if fx.get("spoken_text"):
            line["spoken_text"] = str(fx["spoken_text"]).strip()
    seg["lines"] = [line for line in seg["lines"] if line["id"] not in delete]
    if not seg["lines"]:
        raise ScriptError(f"All lines in segment {seg['id']} were deleted during fixing")


def _taught(lines) -> set:
    return {f for line in lines for f in line["fact_ids"]}


def _apply_edit(segments: list, edited: list, fact_index: dict, host_b: str, log) -> None:
    """Swaps in the editor's version of each segment, unless it is missing or drops a fact
    the draft taught: tightening must never cut teaching content."""
    by_id = {e.get("id"): e for e in edited if isinstance(e, dict)}
    for seg in segments:
        try:
            lines = _normalise_lines(by_id[seg["id"]].get("lines") or [], seg["id"], fact_index, host_b)
        except (KeyError, ScriptError):
            lines = []
        if lines and _taught(seg["lines"]) <= _taught(lines):
            seg["lines"] = lines
        else:
            log(f"    {seg['id']}: kept the draft (edit missing or dropped facts)")


def _rule_issues(lines: list, source: str) -> list:
    """Free checks that run alongside the LLM judge. Every number must appear in the chapter
    exactly (memory leaks hide where the chapter almost matches what the model knows), and
    no line may carry a link, email or phone number: a faithful pipeline would otherwise
    repeat one planted in the pasted text."""
    issues = []
    for line in lines:
        if CONTACT.search(line["display_text"] + " " + line["spoken_text"]):
            issues.append({"line_id": line["id"], "problem": "unsafe", "note": "contains a link, email or phone number"})
            continue
        if line["kind"] == "misconception":
            continue
        missing = [n for n in NUMBER.findall(line["display_text"]) if not re.search(rf"(?<!\d){re.escape(n)}(?!\d)", source)]
        if missing:
            issues.append({"line_id": line["id"], "problem": "unsupported",
                           "note": f"number not in the chapter: {', '.join(missing)}"})
    return issues


def _fallback(seg: dict, issues: list, fact_index: dict, log) -> tuple[list, list]:
    """Lines still flagged after the fix rounds (Dialog Inpainting): a line whose facts all
    have a verified quote is replaced by those quotes word for word; a line that teaches no
    fact is dropped. Returns (auto_resolved, unresolved)."""
    by_id = {line["id"]: line for line in seg["lines"]}
    resolved, unresolved, drop = [], [], set()
    for lid in dict.fromkeys(i["line_id"] for i in issues):
        line, mine = by_id[lid], [i for i in issues if i["line_id"] == lid]
        quotes = [fact_index[f]["quote"] for f in line["fact_ids"] if fact_index[f].get("quote")]
        unsafe = any(i.get("problem") == "unsafe" for i in mine)
        if not unsafe and quotes and len(quotes) == len(line["fact_ids"]):
            line.update(display_text=" ".join(quotes), spoken_text=" ".join(quotes), kind="fact")
            how = "replaced with the chapter's own words"
        elif not line["fact_ids"] and len(drop) + 1 < len(seg["lines"]):
            drop.add(lid)
            how = "deleted"
        else:
            unresolved += [{**i, "segment_id": seg["id"]} for i in mine]
            continue
        resolved += [{**i, "segment_id": seg["id"], "resolution": how} for i in mine]
    seg["lines"] = [line for line in seg["lines"] if line["id"] not in drop]
    if resolved:
        log(f"    {seg['id']}: {len(resolved)} issue(s) resolved from the chapter text")
    if unresolved:
        log(f"    {seg['id']}: {len(unresolved)} issue(s) left for human review")
    return resolved, unresolved


def _verify_and_fix(llm, seg: dict, source: str, fact_index: dict, log) -> tuple[list, list]:
    issues = []
    for round_no in range(MAX_FIX_ROUNDS + 1):
        s, u = prompts.verify(source, seg["lines"])
        data = ask_json(llm, s, u, "issues", temperature=0.0, max_tokens=16000, reasoning=True)  # think before judging
        by_id = {line["id"]: line for line in seg["lines"]}
        # Misconceptions are false on purpose; _pair_misconceptions checks they get refuted.
        issues = [i for i in data["issues"] if isinstance(i, dict) and i.get("line_id") in by_id
                  and by_id[i["line_id"]]["kind"] != "misconception"]
        flagged = {i["line_id"] for i in issues}
        issues += [i for i in _rule_issues(seg["lines"], source) if i["line_id"] not in flagged]
        if not issues:
            log(f"    {seg['id']}: verified, no unsupported claims")
            return [], []
        if round_no == MAX_FIX_ROUNDS:
            break
        log(f"    {seg['id']}: {len(issues)} line(s) flagged, fixing (round {round_no + 1})")
        s, u = prompts.fix(source, [(by_id[i["line_id"]], i) for i in issues])
        _apply_fixes(seg, ask_json(llm, s, u, "fixes", temperature=0.2)["fixes"])
    return _fallback(seg, issues, fact_index, log)


def _pair_misconceptions(seg: dict, log) -> None:
    """A voiced misconception may only stay if B refutes it within the next two lines."""
    lines = seg["lines"]
    keep = [line for i, line in enumerate(lines) if line["kind"] != "misconception"
            or any(n["kind"] == "refutation" and n["speaker"] == "B" for n in lines[i + 1:i + 3])]
    if len(keep) < len(lines):
        log(f"    {seg['id']}: dropped {len(lines) - len(keep)} misconception(s) with no refutation")
    if not keep:
        raise ScriptError(f"Segment {seg['id']} had nothing left after dropping unrefuted misconceptions")
    seg["lines"] = keep


def _tidy_spoken(text: str) -> str:
    text = re.sub(r"\s*[—–]\s*", ", ", text).replace("…", ".")
    return re.sub(r"\s+", " ", text).strip()


def _normalise_spoken(llm, segments: list, language: str, log) -> None:
    """Dashes and ellipses become pauses; lines with digits or symbols left in spoken_text go
    through one few-shot normalisation call. Best effort: TTS can still read what is left."""
    lines = [line for seg in segments for line in seg["lines"]]
    for line in lines:
        line["spoken_text"] = _tidy_spoken(line["spoken_text"])
    todo = [line for line in lines if SPOKEN_PROBLEM.search(line["spoken_text"])]
    if not todo:
        return
    log(f"  Normalising {len(todo)} spoken line(s) for TTS")
    by_id = {line["id"]: line for line in todo}
    try:
        s, u = prompts.normalise(todo, language)
        for item in ask_json(llm, s, u, "lines", temperature=0.0, max_tokens=8000)["lines"]:
            if isinstance(item, dict) and item.get("line_id") in by_id and str(item.get("spoken_text") or "").strip():
                by_id[item["line_id"]]["spoken_text"] = _tidy_spoken(str(item["spoken_text"]))
    except ScriptError as exc:
        log(f"    normalisation skipped: {exc}")
    left = [line["id"] for line in todo if SPOKEN_PROBLEM.search(line["spoken_text"])]
    if left:
        log(f"    still not word-only (TTS reads as-is): {', '.join(left)}")


def build_script(sections, llm, *, minutes, audience, language, host_a, host_b, log=print) -> dict:
    section_index = {s["id"]: s for s in sections}
    section_order = [s["id"] for s in sections]

    # 1. Extract a fact sheet per section, in parallel (sections are independent).
    log(f"  Extracting facts from {len(sections)} section(s)")
    with ThreadPoolExecutor(max_workers=4) as pool:
        per_section = list(pool.map(lambda sec: _extract(llm, sec), sections))
    sections_with_facts = list(zip(sections, per_section))
    facts = [f for sec_facts in per_section for f in sec_facts]
    log(f"  {len(facts)} facts, {sum(1 for f in facts if f['quote'])} with a verified quote")
    if not facts:
        raise ScriptError("No facts were extracted. Is the input text empty or garbled?")
    fact_index = {f["id"]: f for f in facts}

    # 2. Plan the episode.
    log("  Planning episode")
    s, u = prompts.plan(sections_with_facts, minutes, audience)
    plan = ask_json(llm, s, u, "segments", temperature=0.4)
    title = str(plan.get("title") or "Chapter podcast").strip()
    segments = _normalise_plan(plan, fact_index, section_order)
    goals = [seg["goal"] for seg in segments if seg["goal"]]

    # 3. Write each segment.
    total_words = minutes * WORDS_PER_MINUTE
    previous = []
    for i, seg in enumerate(segments):
        if len(segments) == 1:
            position = "only"
        else:
            position = "first" if i == 0 else "last" if i == len(segments) - 1 else "middle"
        words = max(120, round(total_words * len(seg["fact_ids"]) / len(facts)))
        source = _source_text(seg["section_ids"], section_index)
        log(f"  Writing {seg['id']}: {seg['title']} (~{words} words)")
        s, u = prompts.script_segment(
            episode_title=title, segment=seg, position=position,
            facts=[fact_index[f] for f in seg["fact_ids"]], source_text=source,
            previous_lines=previous, episode_goals=goals, words=words,
            audience=audience, language=language, host_a=host_a, host_b=host_b,
        )
        raw = ask_json(llm, s, u, "lines", temperature=0.7, max_tokens=6000)["lines"]
        seg["lines"] = _normalise_lines(raw, seg["id"], fact_index, host_b)
        previous = seg["lines"][-4:]

    # 4. Edit the whole episode for flow and length; a segment that loses facts keeps its draft.
    before = _word_count(segments)
    s, u = prompts.edit(facts=facts, segments=segments, current_words=before, target_words=total_words,
                        audience=audience, host_a=host_a, host_b=host_b)
    edited = ask_json(llm, s, u, "segments", temperature=0.3, max_tokens=16000, reasoning=True)["segments"]
    _apply_edit(segments, edited, fact_index, host_b, log)
    log(f"  Edited for flow and length: {before} -> {_word_count(segments)} words")

    # 5. Verify and fix each segment against its own source text; misconceptions must be refuted.
    unresolved, auto_resolved = [], []
    for seg in segments:
        resolved, left = _verify_and_fix(llm, seg, _source_text(seg["section_ids"], section_index), fact_index, log)
        auto_resolved += resolved
        unresolved += left
        _pair_misconceptions(seg, log)

    # 6. Make spoken_text safe for the TTS voice.
    _normalise_spoken(llm, segments, language, log)

    # Count-based pedagogy check, no judge needed: is Host A more than a question machine?
    a_lines = [line for seg in segments for line in seg["lines"] if line["speaker"] == "A"]
    kinds = [line["kind"] for seg in segments for line in seg["lines"]]
    log(f"  Host A: {sum(l['kind'] == 'question' for l in a_lines)}/{len(a_lines)} lines are questions; "
        f"{kinds.count('misconception')} misconception(s) voiced and refuted")

    return {
        "title": title,
        "language": language,
        "audience": audience,
        "hosts": {"A": host_a, "B": host_b},
        "segments": [
            {k: seg[k] for k in ("id", "title", "goal", "deep_question", "misconception", "section_ids", "fact_ids", "lines")}
            for seg in segments
        ],
        "facts": facts,
        "auto_resolved": auto_resolved,
        "unresolved_issues": unresolved,
    }
