"""Turns cleaned sections into a verified two-host script (script.json).

Stages: extract facts -> plan segments -> write each segment -> edit the whole episode
for flow and length -> verify against the source -> fix flagged lines. Verify always
runs last, so every line that reaches the audio has been checked. Anything still
flagged after the fix rounds is kept in script["unresolved_issues"] so a teacher can
review it before publishing.
"""
import json
import re

import prompts

WORDS_PER_MINUTE = 150
KINDS = {"fact", "analogy", "question", "banter", "recap"}
MAX_FIX_ROUNDS = 2


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
        segments.append({
            "title": str(raw.get("title") or "Segment").strip(),
            "goal": str(raw.get("goal") or "").strip(),
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
        lines.append({
            "id": f"{seg_id}-l{len(lines) + 1}",
            "speaker": speaker,
            "display_text": display or spoken,
            "spoken_text": spoken,
            "kind": kind if kind in KINDS else "fact",
            "fact_ids": [f for f in raw.get("fact_ids") or [] if f in fact_index],
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


def _apply_edits(segments: list, edits: list, fact_index: dict) -> None:
    """Applies the editor's rewrites and deletes through _apply_fixes, but skips any delete
    that would drop a fact or empty a segment: tightening must never cut teaching content."""
    edits = [e for e in edits if isinstance(e, dict)]
    lines = {line["id"]: line for seg in segments for line in seg["lines"]}
    for e in edits:  # a merged line takes over the facts of the line it absorbed
        if e.get("action") != "delete" and e.get("line_id") in lines:
            line = lines[e["line_id"]]
            line["fact_ids"] += [f for f in e.get("fact_ids") or [] if f in fact_index and f not in line["fact_ids"]]
    deleted = {e.get("line_id") for e in edits if e.get("action") == "delete"}
    kept = {f for lid, line in lines.items() if lid not in deleted for f in line["fact_ids"]}
    for seg in segments:
        ids = {line["id"] for line in seg["lines"]}
        mine = [e for e in edits if e.get("line_id") in ids
                and (e.get("action") != "delete" or set(lines[e["line_id"]]["fact_ids"]) <= kept)]
        if {e["line_id"] for e in mine if e.get("action") == "delete"} >= ids:
            mine = [e for e in mine if e.get("action") != "delete"]
        _apply_fixes(seg, mine)


def _verify_and_fix(llm, seg: dict, source: str, log) -> list:
    issues = []
    for round_no in range(MAX_FIX_ROUNDS + 1):
        s, u = prompts.verify(source, seg["lines"])
        data = ask_json(llm, s, u, "issues", temperature=0.0, max_tokens=16000, reasoning=True)  # think before judging
        by_id = {line["id"]: line for line in seg["lines"]}
        issues = [i for i in data["issues"] if isinstance(i, dict) and i.get("line_id") in by_id]
        if not issues:
            log(f"    {seg['id']}: verified, no unsupported claims")
            return []
        if round_no == MAX_FIX_ROUNDS:
            break
        log(f"    {seg['id']}: {len(issues)} line(s) flagged, fixing (round {round_no + 1})")
        s, u = prompts.fix(source, [(by_id[i["line_id"]], i) for i in issues])
        _apply_fixes(seg, ask_json(llm, s, u, "fixes", temperature=0.2)["fixes"])
    log(f"    {seg['id']}: {len(issues)} issue(s) left for human review")
    return [{**i, "segment_id": seg["id"]} for i in issues]


def build_script(sections, llm, *, minutes, audience, language, host_a, host_b, log=print) -> dict:
    section_index = {s["id"]: s for s in sections}
    section_order = [s["id"] for s in sections]

    # 1. Extract a fact sheet per section.
    facts, sections_with_facts = [], []
    for sec in sections:
        log(f"  Extracting facts: section {sec['id']} ({sec['words']} words)")
        s, u = prompts.extract(sec["id"], sec["title"], sec["text"])
        items = ask_json(llm, s, u, "facts", temperature=0.1)["facts"]
        sec_facts = []
        for item in items:
            text = str(item.get("text") or "").strip() if isinstance(item, dict) else ""
            if text:
                sec_facts.append({
                    "id": f"{sec['id']}-f{len(sec_facts) + 1}",
                    "section_id": sec["id"],
                    "type": str(item.get("type") or "concept"),
                    "text": text,
                })
        facts.extend(sec_facts)
        sections_with_facts.append((sec, sec_facts))
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

    # 4. Edit the whole episode for flow and length (rewrite/delete only, facts kept).
    before = _word_count(segments)
    s, u = prompts.edit(facts=facts, segments=segments, current_words=before, target_words=total_words,
                        audience=audience, host_a=host_a, host_b=host_b)
    _apply_edits(segments, ask_json(llm, s, u, "edits", temperature=0.2, max_tokens=12000)["edits"], fact_index)
    log(f"  Edited for flow and length: {before} -> {_word_count(segments)} words")

    # 5. Verify and fix each segment against its own source text.
    unresolved = []
    for seg in segments:
        unresolved.extend(_verify_and_fix(llm, seg, _source_text(seg["section_ids"], section_index), log))

    return {
        "title": title,
        "language": language,
        "audience": audience,
        "hosts": {"A": host_a, "B": host_b},
        "segments": [
            {k: seg[k] for k in ("id", "title", "goal", "section_ids", "fact_ids", "lines")} for seg in segments
        ],
        "facts": facts,
        "unresolved_issues": unresolved,
    }
