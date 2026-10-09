import pytest

from script_builder import ScriptError, _apply_edit, _apply_fixes, _normalise_lines, _normalise_plan, parse_json


# --- parse_json ---------------------------------------------------------------

def test_parse_json_plain_fenced_and_wrapped_in_prose():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_json('Sure! Here it is:\n```\n{"a": {"b": "}"}}\n```\nHope that helps.') == {"a": {"b": "}"}}


@pytest.mark.parametrize("reply", ["", "no json here", "[1, 2]", "{broken"])
def test_parse_json_rejects_non_objects(reply):
    with pytest.raises(ValueError):  # json.JSONDecodeError is a ValueError too, so ask_json retries
        parse_json(reply)


# --- plan normalisation -------------------------------------------------------

FACTS = {fid: {"id": fid, "section_id": fid.split("-")[0]} for fid in ("8.1-f1", "8.1-f2", "8.2-f1", "8.2-f2", "8.3-f1")}
ORDER = ["8.1", "8.2", "8.3"]


def test_plan_drops_bogus_and_duplicate_ids_and_reassigns_forgotten_facts():
    plan = {"segments": [
        "junk",
        {"title": "Start", "goal": "g1", "fact_ids": ["8.2-f1", "bogus-1"], "section_ids": ["9.9"]},
        {"title": "Only bogus", "fact_ids": ["nope"]},
        {"goal": "g2", "fact_ids": ["8.2-f1", "8.1-f1"]},  # 8.2-f1 already used
    ]}
    segments = _normalise_plan(plan, FACTS, ORDER)

    assert [(s["id"], s["title"], s["fact_ids"], s["section_ids"]) for s in segments] == [
        ("s1", "Start", ["8.2-f1", "8.2-f2"], ["8.2"]),                       # forgotten 8.2-f2 joins its section
        ("s2", "Segment", ["8.1-f1", "8.1-f2", "8.3-f1"], ["8.1", "8.3"]),   # unplaced 8.3-f1 goes last
    ]
    covered = [f for s in segments for f in s["fact_ids"]]
    assert sorted(covered) == sorted(FACTS)  # every fact exactly once


def test_plan_with_only_bogus_ids_fails():
    with pytest.raises(ScriptError):
        _normalise_plan({"segments": [{"title": "x", "fact_ids": ["nope"]}]}, FACTS, ORDER)


# --- fix application ----------------------------------------------------------

def _seg():
    return {"id": "s1", "lines": [
        {"id": f"s1-l{i}", "display_text": t, "spoken_text": t.lower()} for i, t in enumerate("ABC", 1)
    ]}


def test_apply_fixes_rewrites_deletes_and_ignores_junk():
    seg = _seg()
    _apply_fixes(seg, [
        {"line_id": "s1-l1", "action": "rewrite", "display_text": " A fixed ", "spoken_text": "a fixed"},
        {"line_id": "s1-l2", "action": "delete"},
        {"line_id": "s1-l3", "action": "rewrite", "display_text": "", "spoken_text": "c only spoken"},
        {"line_id": "s9-l9", "action": "delete"},
        "garbage",
    ])
    assert seg["lines"] == [
        {"id": "s1-l1", "display_text": "A fixed", "spoken_text": "a fixed"},
        {"id": "s1-l3", "display_text": "C", "spoken_text": "c only spoken"},
    ]


def test_apply_fixes_deleting_everything_fails():
    seg = _seg()
    with pytest.raises(ScriptError):
        _apply_fixes(seg, [{"line_id": line["id"], "action": "delete"} for line in seg["lines"]])


# --- editor changes -----------------------------------------------------------

def test_apply_edit_keeps_the_draft_when_facts_would_be_lost():
    def line(lid, facts):
        return {"id": lid, "speaker": "A", "kind": "fact", "display_text": lid, "spoken_text": lid, "fact_ids": facts}

    def draft():
        return [
            {"id": "s1", "lines": [line("s1-l1", []), line("s1-l2", ["8.1-f1"]), line("s1-l3", ["8.1-f2"])]},
            {"id": "s2", "lines": [line("s2-l1", ["8.2-f1"])]},
            {"id": "s3", "lines": [line("s3-l1", ["8.3-f1"])]},
        ]

    segments, logs = draft(), []
    _apply_edit(segments, [
        {"id": "s1", "lines": [{"speaker": "B", "display_text": "Both.", "fact_ids": ["8.1-f1", "8.1-f2", "bogus"]}]},
        {"id": "s2", "lines": [{"speaker": "B", "display_text": "Nothing taught.", "fact_ids": []}]},
        "junk",  # and s3 is missing entirely
    ], FACTS, "Arjun", logs.append)
    assert [(l["id"], l["display_text"], l["fact_ids"]) for l in segments[0]["lines"]] == [("s1-l1", "Both.", ["8.1-f1", "8.1-f2"])]
    assert segments[1:] == draft()[1:]
    assert len(logs) == 2


def test_normalise_lines_merges_one_host_talking_twice():
    lines = _normalise_lines([
        {"speaker": "A", "display_text": "Hi!", "kind": "banter"},
        {"speaker": "Arjun", "display_text": "Sound is vibration.", "kind": "question", "fact_ids": ["8.1-f1"]},
        {"speaker": "B", "display_text": "H₂O carries it.", "spoken_text": "H two O carries it.", "kind": "fact",
         "fact_ids": ["8.1-f1", "8.1-f2", "bogus"]},
        "junk", {"speaker": "A", "display_text": ""},
    ], "s1", FACTS, "Arjun")
    assert lines == [
        {"id": "s1-l1", "speaker": "A", "display_text": "Hi!", "spoken_text": "Hi!", "kind": "banter", "fact_ids": []},
        {"id": "s1-l2", "speaker": "B", "display_text": "Sound is vibration. H₂O carries it.",
         "spoken_text": "Sound is vibration. H two O carries it.", "kind": "fact", "fact_ids": ["8.1-f1", "8.1-f2"]},
    ]
