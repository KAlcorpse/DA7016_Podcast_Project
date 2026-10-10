import math
import struct

import pytest

from render import (MAX_CUE_CHARS, MIN_CUE_SECONDS, SAMPLE_RATE, TARGET_RMS, build_cues, duration_outliers,
                    gap_after, split_for_tts, trim_and_level, write_srt, write_vtt)

HOSTS = {"A": "Meera", "B": "Arjun"}


# --- split_for_tts ------------------------------------------------------------

def test_short_text_is_one_chunk_with_whitespace_collapsed():
    assert split_for_tts("  Hello\n there.  ", 250) == ["Hello there."]


def test_long_text_splits_on_sentences_then_commas_and_keeps_every_word():
    text = ("Mendel used pea plants. " * 6 + "When the F1 plants self-pollinated, the F2 generation had tall plants, "
            "short plants, and a ratio of about three is to one, which surprised everyone at the time.")
    chunks = split_for_tts(text, 60)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 60 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_word_longer_than_limit_is_hard_cut():
    chunks = split_for_tts("x" * 25, 10)
    assert chunks == ["x" * 10, "x" * 10, "x" * 5]


# --- caption cues -------------------------------------------------------------

def test_short_line_is_one_cue():
    line = {"id": "s1-l1", "speaker": "A", "display_text": "Hello there. Welcome!", "start": 0.4, "end": 3.0}
    assert build_cues([line], HOSTS) == [
        {"start": 0.4, "end": 3.0, "speaker": "Meera", "text": "Hello there. Welcome!", "line_id": "s1-l1"}
    ]


def test_long_line_packs_sentences_into_two_line_cues():
    line = {"id": "s1-l2", "speaker": "B", "start": 10.0, "end": 20.0,
            "display_text": "First sentence is here and it has some words. Second one is quite a bit longer than the first one is. Third."}
    cues = build_cues([line], HOSTS)
    assert [c["text"] for c in cues] == ["First sentence is here and it has some words.",
                                         "Second one is quite a bit longer than the first one is. Third."]
    assert all(len(c["text"]) <= MAX_CUE_CHARS for c in cues)
    assert {c["speaker"] for c in cues} == {"Arjun"} and {c["line_id"] for c in cues} == {"s1-l2"}
    assert cues[0]["start"] == 10.0 and cues[-1]["end"] == 20.0
    for a, b in zip(cues, cues[1:]):
        assert a["end"] == pytest.approx(b["start"])


def test_cue_time_follows_spoken_length():
    first = "Mendel saw a ratio of 3:1 in the F2 generation of his pea plants."
    second = "That surprised everyone who studied heredity back then in his time."
    said_first = "Mendel saw a ratio of three is to one in the F two generation of his pea plants."
    line = {"id": "x", "speaker": "A", "start": 0.0, "end": 10.0,
            "display_text": f"{first} {second}", "spoken_text": f"{said_first} {second}"}
    cues = build_cues([line], HOSTS)
    assert [c["text"] for c in cues] == [first, second]
    assert cues[0]["end"] == pytest.approx(10 * len(said_first) / (len(said_first) + len(second)))


def test_no_cue_shorter_than_minimum():
    line = {"id": "x", "speaker": "A", "start": 0.0, "end": 4.0,
            "display_text": "A long first sentence that takes most of the time to read out loud for everyone listening. Ok."}
    cues = build_cues([line], HOSTS)
    assert all(c["end"] - c["start"] >= MIN_CUE_SECONDS for c in cues)
    assert cues[-1]["text"].endswith("Ok.")


def test_long_single_sentence_stays_one_cue():
    line = {"id": "x", "speaker": "A", "display_text": "One long sentence, with commas, only.", "start": 0, "end": 12}
    assert len(build_cues([line], HOSTS)) == 1


# --- audio --------------------------------------------------------------------

def _tone(seconds, amplitude):
    return b"".join(struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)))
                    for i in range(int(seconds * SAMPLE_RATE)))


def test_trim_and_level_cuts_silence_and_matches_loudness():
    quiet = b"\x00\x00" * (SAMPLE_RATE // 2)
    out = trim_and_level(quiet + _tone(0.5, 2000) + quiet)
    assert 0.5 <= len(out) / (2 * SAMPLE_RATE) <= 0.6  # 0.5 s of speech plus two 0.04 s pads
    samples = struct.unpack(f"<{len(out) // 2}h", out)
    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    assert rms == pytest.approx(TARGET_RMS, rel=0.1)
    assert trim_and_level(quiet) == quiet  # all silence: leave it alone


def test_turn_gaps():
    short = {"id": "s1-l3", "display_text": "Wait, really?", "kind": "question"}
    question = {"id": "s1-l4", "display_text": "So why does the band stop making sound?", "kind": "question"}
    statement = {"id": "s1-l5", "display_text": "Because the vibration has stopped completely.", "kind": "fact"}
    assert 0.07 <= gap_after(short) <= 0.17
    assert 0.30 <= gap_after(question) <= 0.40
    assert 0.17 <= gap_after(statement) <= 0.27
    assert gap_after(statement) == gap_after(dict(statement))  # same line, same gap


def test_duration_outliers():
    text = "x" * 30  # 30 characters at 15 characters a second = 2 s expected
    clips = [(i, "Kavya", text, 2.0) for i in range(5)]
    clips += [(5, "Kavya", text, 4.4), (6, "Kavya", text, 0.8), (7, "Kavya", "Right.", 5.0), (8, "Arun", text, 9.0)]
    flagged = duration_outliers(clips)
    assert sorted(flagged) == [5, 6]  # looping and dropped words; short chunk and lone Arun clip not judged
    assert flagged[5] == pytest.approx(2.2) and flagged[6] == pytest.approx(0.4)


def test_vtt_and_srt_format(tmp_path):
    cues = [{"start": 3661.5, "end": 3662.0, "speaker": "Meera", "text": "H<sub>2</sub>O & more", "line_id": "x"}]
    write_vtt(cues, tmp_path / "e.vtt")
    write_srt(cues, tmp_path / "e.srt")
    vtt, srt = (tmp_path / "e.vtt").read_text(), (tmp_path / "e.srt").read_text()
    assert vtt.startswith("WEBVTT\n")
    assert "01:01:01.500 --> 01:01:02.000\n<v Meera>H&lt;sub&gt;2&lt;/sub&gt;O &amp; more" in vtt
    assert "01:01:01,500 --> 01:01:02,000\nMeera: H<sub>2</sub>O & more" in srt
