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


QUOTES = {  # the first quote is verbatim from the chapter, the second is invented
    "8.1": "Inheritance from the previous generation provides both a common basic body design, and subtle changes in it, for the next generation.",
    "8.2": "The most obvious outcome of the reproductive process still remains the generation of individuals of similar design.",
}
DIGIT_WORDS = dict(zip("0123456789", "zero one two three four five six seven eight nine".split()))


class FakeLLM:
    """Answers each STAGE using only ids it finds in the prompt. The first verify call flags
    the last line (and wrongly a misconception, which must be ignored); a stubborn verifier
    flags every fact and banter line on every call."""

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
        return json.dumps({"facts": [
            {"type": "concept", "text": f"Fact 1 from {sid}.", "quote": QUOTES[sid]},
            {"type": "concept", "text": f"Fact 2 from {sid}.", "quote": "A sentence the chapter never says."},
        ]})

    def plan(self, user):
        ids = _ids(_between(user, "source"))
        half = len(ids) // 2
        reply = {"title": "Test episode", "segments": [
            {"title": "Part one", "goal": "Learn one", "fact_ids": ids[:half] + ["made-up-id"],
             "deep_question": "Why?", "misconception": "It's all luck.", "refuting_fact_id": ids[0]},
            {"title": "Part two", "goal": "Learn two", "fact_ids": ids[half:],
             "deep_question": "How?", "misconception": "It's random.", "refuting_fact_id": "made-up-id"},
        ]}
        return "Sure! Here is the plan:\n```json\n" + json.dumps(reply) + "\n```"  # exercises parse_json

    def script(self, user):
        def line(speaker, text, kind, facts=(), spoken=None):
            return {"speaker": speaker, "display_text": text, "spoken_text": spoken or text, "kind": kind, "fact_ids": list(facts)}
        def fact(who, fid):  # the fact id in spoken_text gives the normaliser digits to remove
            return line(who, "Here is a fact. It is H₂O.", "fact", [fid], f"Here is {fid}. It is H two O.")
        fids = _ids(_between(user, "facts"))
        lines = [line("A", "Why does this happen?", "question"), fact("B", fids[0])]
        if _between(user, "misconception") != "(none)":
            lines += [line("A", "So it's all luck?", "misconception"), line("B", "No, the chapter says otherwise.", "refutation", fids[:1])]
        lines.append(fact("Meera", fids[1]))
        if _between(user, "misconception") == "(none)":  # a stray misconception nobody refutes
            lines += [line("B", "Hmm.", "banter"), line("A", "Maybe it's random?", "misconception")]
        return json.dumps({"lines": lines})

    def edit(self, user):
        def line(speaker, text, kind, facts=()):
            return {"speaker": speaker, "display_text": text, "spoken_text": text, "kind": kind, "fact_ids": list(facts)}
        return json.dumps({"segments": [
            {"id": "s1", "lines": [line("A", "Why?", "question"), line("B", "Tight 8.1-f1, at 343 m/s.", "fact", ["8.1-f1"]),
                                   line("A", "Luck?", "misconception"), line("B", "No.", "refutation", ["8.1-f1"]),
                                   line("Meera", "Tight 8.1-f2.", "fact", ["8.1-f2"])]},
            {"id": "s2", "lines": [line("B", "Only 8.2-f1.", "fact", ["8.2-f1"])]},  # drops 8.2-f2: draft must be kept
        ]})

    def verify(self, user):
        block = user.split("<lines>")[1]
        if self.stubborn:
            flagged = re.findall(r"^\[([^\]]+)\] \w \((?:fact|banter)\)", block, re.M)
        elif len(self.stage_calls("verify")) == 1:
            flagged = re.findall(r"^\[([^\]]+)\] \w \(misconception\)", block, re.M) + _ids(block)[-1:]
        else:
            flagged = []
        return json.dumps({"issues": [{"line_id": lid, "problem": "unsupported", "note": "x"} for lid in flagged]})

    def fix(self, user):
        return json.dumps({"fixes": [{"line_id": lid, "action": "rewrite", "display_text": "Fixed line.",
                                      "spoken_text": "Fixed line."} for lid in _ids(_between(user, "flagged_lines"))]})

    def normalise(self, user):
        found = re.findall(r"^\[([^\]]+)\] display: .*\n  spoken: (.*)$", _between(user, "lines"), re.M)
        return json.dumps({"lines": [{"line_id": lid, "spoken_text": "".join(f" {DIGIT_WORDS[c]} " if c.isdigit() else c for c in text)}
                                     for lid, text in found]})


def _build(llm):
    return build_script(prepare_sections(CHAPTER), llm, minutes=2, audience="Class 10 students",
                        language="English", host_a="Meera", host_b="Arjun", log=lambda *_: None)


def test_script_is_grounded_verified_and_fixed():
    llm = FakeLLM()
    script = _build(llm)

    # Grounding: every stage that reads chapter content sees <source>; all but verify carry the shared rules.
    for stage, system, user in llm.calls:
        assert stage == "normalise" or "<source>" in user, stage
        assert stage == "verify" or prompts.GROUNDING_RULES in system, stage
    # Extraction only ever sees the pasted section text (sections run in parallel).
    sections = prepare_sections(CHAPTER)
    assert sorted(_between(u, "source").strip() for _, _, u in llm.stage_calls("extract")) == sorted(s["text"] for s in sections)
    # Only quotes that really occur in the chapter are kept.
    assert [f["quote"] for f in script["facts"]] == [QUOTES["8.1"], "", QUOTES["8.2"], ""]

    # One whole-episode edit, verify only after it, and TTS normalisation last.
    stages = [c[0] for c in llm.calls]
    assert stages.count("edit") == 1 and stages.index("edit") < stages.index("verify")
    assert stages[-1] == "normalise"
    assert llm.reasoned == {"edit", "verify"}  # the judgement stages think first

    # s1: the 343 m/s line fails the free number check and the last line fails the judge; both get
    # fixed. The judge also flagged the misconception, which is ignored: it is refuted next turn.
    s1, s2 = (seg["lines"] for seg in script["segments"])
    assert [(l["display_text"], l["kind"]) for l in s1] == [
        ("Why?", "question"), ("Fixed line.", "fact"), ("Luck?", "misconception"), ("No.", "refutation"), ("Fixed line.", "fact")]
    assert {l["speaker"] for l in s1} == {"A", "B"}  # host name "Meera" mapped back to A
    assert len(llm.stage_calls("verify")) == len(script["segments"]) + 1
    assert len(llm.stage_calls("fix")) == 1

    # s2 keeps its draft (the edit dropped a fact); its stray unrefuted misconception is removed,
    # and the planner's misconception with a made-up refuting fact never reached the writer.
    assert [l["id"] for l in s2] == ["s2-l1", "s2-l2", "s2-l3", "s2-l4"]
    assert script["segments"][0]["misconception"] == "It's all luck." and script["segments"][1]["misconception"] == ""
    assert not any(re.search(r"\d", l["spoken_text"]) for seg in script["segments"] for l in seg["lines"])

    assert [s["section_ids"] for s in script["segments"]] == [["8.1"], ["8.2"]]
    assert sorted(f for s in script["segments"] for f in s["fact_ids"]) == sorted(f["id"] for f in script["facts"])
    assert script["unresolved_issues"] == [] and script["auto_resolved"] == []


def test_stubborn_verifier_falls_back_to_the_chapter_words():
    llm = FakeLLM(stubborn=True)
    script = _build(llm)
    assert len(llm.stage_calls("fix")) == MAX_FIX_ROUNDS * len(script["segments"])
    s1, s2 = (seg["lines"] for seg in script["segments"])
    # Lines whose facts have a verified quote now say the chapter's own words; fact-free banter is gone.
    assert s1[1]["display_text"] == QUOTES["8.1"] and s2[1]["display_text"] == QUOTES["8.2"]
    assert [(i["line_id"], i["resolution"]) for i in script["auto_resolved"]] == [
        ("s1-l2", "replaced with the chapter's own words"), ("s2-l2", "replaced with the chapter's own words"), ("s2-l4", "deleted")]
    # Facts without a verified quote can't be rescued, so a teacher must review them.
    assert [i["line_id"] for i in script["unresolved_issues"]] == ["s1-l5", "s2-l3"]


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
    assert [l["speaker_name"] for l in manifest["lines"][:2]] == ["Meera", "Arjun"]
