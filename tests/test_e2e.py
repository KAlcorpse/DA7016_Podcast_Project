"""Whole pipeline with a scripted fake LLM and MockTTS: no network, no credit spent."""
import json
import re
import shutil

import pytest

import prompts
from clean import prepare_sections
from providers import MockTTS
from render import Renderer
from script_builder import MAX_FIX_ROUNDS, build_script

CHAPTER = """8.1 Accumulation of Variation During Reproduction
Inheritance from the previous generation provides both a common basic body design, and subtle changes in it, for the next generation.

8.2 Heredity
The most obvious outcome of the reproductive process still remains the generation of individuals of similar design.
"""


def _ids(block: str) -> list:
    return re.findall(r"^\[([^\]]+)\]", block, re.M)


def _between(text: str, tag: str) -> str:
    return text.split(f"<{tag}>")[1].split(f"</{tag}>")[0]


class FakeLLM:
    """Answers each STAGE using only ids it finds in the prompt. The first verify call flags
    a line so the fix loop runs; a stubborn verifier never accepts anything."""

    def __init__(self, stubborn: bool = False):
        self.stubborn = stubborn
        self.calls = []  # (stage, system, user)
        self.reasoned = set()  # stages that asked the model to think first

    def complete(self, system, user, temperature=0.3, max_tokens=4000, reasoning=False):
        stage = system.splitlines()[0].removeprefix("STAGE: ")
        self.calls.append((stage, system, user))
        if reasoning:
            self.reasoned.add(stage)
        return getattr(self, stage)(user)

    def stage_calls(self, stage):
        return [c for c in self.calls if c[0] == stage]

    def extract(self, user):
        sid = re.search(r"Section (\S+):", user).group(1)
        return json.dumps({"facts": [{"type": "concept", "text": f"Fact {n} from {sid}."} for n in (1, 2)]})

    def plan(self, user):
        ids = _ids(_between(user, "source"))
        half = len(ids) // 2
        reply = {"title": "Test episode", "segments": [
            {"title": "Part one", "goal": "Learn one", "fact_ids": ids[:half] + ["made-up-id"]},
            {"title": "Part two", "goal": "Learn two", "fact_ids": ids[half:]},
        ]}
        return "Sure! Here is the plan:\n```json\n" + json.dumps(reply) + "\n```"  # exercises parse_json

    def script(self, user):
        lines = [{"speaker": "A", "display_text": "So what is this about?", "spoken_text": "So what is this about?",
                  "kind": "question", "fact_ids": []}]
        for i, fid in enumerate(_ids(_between(user, "facts"))):
            lines.append({"speaker": "B" if i % 2 == 0 else "Meera", "display_text": f"Here is {fid}. It is H₂O.",
                          "spoken_text": f"Here is {fid}. It is H two O.", "kind": "fact", "fact_ids": [fid]})
        return json.dumps({"lines": lines})

    def edit(self, user):
        return json.dumps({"edits": [
            {"line_id": "s1-l1", "action": "delete"},                       # fact-free question: allowed
            {"line_id": "s1-l2", "action": "delete"},                       # only line teaching 8.1-f1: refused
            {"line_id": "s2-l2", "action": "rewrite", "display_text": "Both facts.", "spoken_text": "Both facts.",
             "fact_ids": ["8.2-f2"]},                                       # merge s2-l3 into s2-l2
            {"line_id": "s2-l3", "action": "delete"},
        ]})

    def verify(self, user):
        flag = self.stubborn or len(self.stage_calls("verify")) == 1
        last = _ids(user.split("<lines>")[1])[-1]
        return json.dumps({"issues": [{"line_id": last, "problem": "unsupported", "note": "x"}] if flag else []})

    def fix(self, user):
        line_id = _ids(_between(user, "flagged_lines"))[0]
        return json.dumps({"fixes": [{"line_id": line_id, "action": "rewrite",
                                      "display_text": "Fixed line.", "spoken_text": "Fixed line."}]})


def _build(llm):
    return build_script(prepare_sections(CHAPTER), llm, minutes=2, audience="Class 10 students",
                        language="English", host_a="Meera", host_b="Arjun", log=lambda *_: None)


def test_script_is_grounded_verified_and_fixed():
    llm = FakeLLM()
    script = _build(llm)

    # Grounding: every stage sees <source>; every stage except verify carries the shared rules.
    for stage, system, user in llm.calls:
        assert "<source>" in user, stage
        assert stage == "verify" or prompts.GROUNDING_RULES in system, stage
    # Extraction only ever sees the pasted section text.
    sections = prepare_sections(CHAPTER)
    assert [_between(u, "source").strip() for _, _, u in llm.stage_calls("extract")] == [s["text"] for s in sections]

    # One whole-episode edit, and verify only after it, so edited lines are checked too.
    stages = [c[0] for c in llm.calls]
    assert stages.count("edit") == 1 and stages.index("edit") < stages.index("verify")
    s1, s2 = (seg["lines"] for seg in script["segments"])
    assert [line["id"] for line in s1] == ["s1-l2", "s1-l3"]
    assert [(line["id"], line["fact_ids"]) for line in s2] == [("s2-l1", []), ("s2-l2", ["8.2-f1", "8.2-f2"])]

    # Verify ran on every segment (plus a re-check after the fix), with reasoning on.
    assert len(llm.stage_calls("verify")) == len(script["segments"]) + 1
    assert llm.reasoned == {"verify"}
    assert len(llm.stage_calls("fix")) == 1

    assert [s["section_ids"] for s in script["segments"]] == [["8.1"], ["8.2"]]
    assert sorted(f for s in script["segments"] for f in s["fact_ids"]) == sorted(f["id"] for f in script["facts"])
    first = script["segments"][0]["lines"]
    assert first[-1]["display_text"] == "Fixed line."
    assert {line["speaker"] for line in first} == {"A", "B"}  # host name "Meera" mapped back to A
    assert script["unresolved_issues"] == []


def test_stubborn_verifier_leaves_issues_for_review():
    llm = FakeLLM(stubborn=True)
    script = _build(llm)
    assert len(llm.stage_calls("fix")) == MAX_FIX_ROUNDS * len(script["segments"])
    assert [i["segment_id"] for i in script["unresolved_issues"]] == ["s1", "s2"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_end_to_end_with_mock_tts(tmp_path):
    script = _build(FakeLLM())
    renderer = Renderer(MockTTS(), {"A": "mock-A", "B": "mock-B"}, "en", tmp_path / "cache", log=lambda *_: None)
    manifest = renderer.render(script, tmp_path / "out")

    out = tmp_path / "out"
    assert (out / "episode.mp3").stat().st_size > 1000
    assert (out / "episode.vtt").read_text().startswith("WEBVTT")
    assert (out / "episode.srt").read_text().startswith("1\n")
    assert json.loads((out / "manifest.json").read_text()) == manifest
    assert not (out / "master.wav").exists()

    script_ids = [line["id"] for seg in script["segments"] for line in seg["lines"]]
    assert [line["id"] for line in manifest["lines"]] == script_ids
    assert manifest["needs_review"] is False
    assert [s["id"] for s in manifest["segments"]] == ["s1", "s2"]
    ends = 0.0
    for line in manifest["lines"]:
        assert ends <= line["start"] < line["end"]
        ends = line["end"]
    assert manifest["duration"] >= ends
    assert [l["speaker_name"] for l in manifest["lines"][:2]] == ["Arjun", "Meera"]
