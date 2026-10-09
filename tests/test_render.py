import pytest

from render import build_cues, split_for_tts, write_srt, write_vtt

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


def test_long_line_splits_into_contiguous_sentence_cues():
    line = {"id": "s1-l2", "speaker": "B", "start": 10.0, "end": 20.0,
            "display_text": "First sentence here. Second one is quite a bit longer. Third."}
    cues = build_cues([line], HOSTS)
    assert [c["text"] for c in cues] == ["First sentence here.", "Second one is quite a bit longer.", "Third."]
    assert {c["speaker"] for c in cues} == {"Arjun"} and {c["line_id"] for c in cues} == {"s1-l2"}
    assert cues[0]["start"] == 10.0 and cues[-1]["end"] == pytest.approx(20.0)
    for a, b in zip(cues, cues[1:]):
        assert a["end"] == pytest.approx(b["start"])
    assert max(cues, key=lambda c: c["end"] - c["start"])["text"].startswith("Second")  # time follows length


def test_long_single_sentence_stays_one_cue():
    line = {"id": "x", "speaker": "A", "display_text": "One long sentence, with commas, only.", "start": 0, "end": 12}
    assert len(build_cues([line], HOSTS)) == 1


def test_vtt_and_srt_format(tmp_path):
    cues = [{"start": 3661.5, "end": 3662.0, "speaker": "Meera", "text": "H<sub>2</sub>O & more", "line_id": "x"}]
    write_vtt(cues, tmp_path / "e.vtt")
    write_srt(cues, tmp_path / "e.srt")
    vtt, srt = (tmp_path / "e.vtt").read_text(), (tmp_path / "e.srt").read_text()
    assert vtt.startswith("WEBVTT\n")
    assert "01:01:01.500 --> 01:01:02.000\n<v Meera>H&lt;sub&gt;2&lt;/sub&gt;O &amp; more" in vtt
    assert "01:01:01,500 --> 01:01:02,000\nMeera: H<sub>2</sub>O & more" in srt
