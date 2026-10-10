"""script.json -> episode.mp3 + episode.vtt + episode.srt + manifest.json

Each line is synthesised separately (Bodhan recommends short TTS inputs), converted
to one common PCM format, trimmed, gain-matched and stitched with short pauses. Because
we place every clip ourselves, we know exactly when each line starts and ends, which
gives caption timestamps without any extra alignment step.

Audio choices follow the research review (NCERT podcast generator research papers.md):
turn gaps near 0.2 s (Levinson & Torreira 2015), each clip gain-matched before the
master is normalised to -16 LUFS (Apple podcast spec), a duration check with retakes
because LLM-based TTS can repeat or drop words (Neekhara et al. 2024), and captions sized
to two 42-character lines (Netflix children's timed-text guide).
"""
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import tempfile
import wave
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clean import split_sentences

SAMPLE_RATE = 24000
GAP_WITHIN_LINE = 0.12     # seconds between TTS chunks of one line
GAP_BETWEEN_SEGMENTS = 0.8
MAX_CUE_CHARS = 84         # two caption lines of 42 characters
MIN_CUE_SECONDS = 0.83
TRIM_THRESHOLD = 300       # samples quieter than this (about -40 dBFS) count as silence
TRIM_PAD = 0.04            # seconds of silence left on each side of a trimmed clip
TARGET_RMS = 3000          # every clip is levelled to this before stitching (about -21 dBFS)
QC_MIN_CHARS = 15          # shorter chunks have too noisy a speaking rate to judge
QC_RANGE = (0.5, 1.8)      # allowed duration / expected duration
QC_RETAKES = 2


class RenderError(RuntimeError):
    pass


def _ffmpeg(*args: str) -> None:
    try:
        result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    except FileNotFoundError:
        raise RenderError("ffmpeg is not installed (sudo apt install ffmpeg)") from None
    if result.returncode != 0:
        raise RenderError(f"ffmpeg failed: {result.stderr[-800:]}")


def _loudness(path: Path):
    """Integrated loudness of the finished episode in LUFS, or None if ffmpeg can't say."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                            capture_output=True, text=True)
    found = re.findall(r"I:\s+(-?[\d.]+) LUFS", result.stderr)
    return float(found[-1]) if found else None


def _to_pcm_wav(audio: bytes, dest: Path) -> None:
    """Whatever format the TTS returns (wav/mp3/ogg), normalise it to mono 24 kHz 16-bit."""
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(audio)
        src = tmp.name
    try:
        _ffmpeg("-i", src, "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", str(dest))
    finally:
        os.unlink(src)


def _read_frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SAMPLE_RATE):
            raise RenderError(f"Unexpected audio format in {path}")
        return w.readframes(w.getnframes())


def trim_and_level(frames: bytes) -> bytes:
    """Cuts the voice's own leading and trailing silence, so our gaps are the real gaps, and
    gain-matches the clip so the hosts don't jump in loudness at every turn."""
    a = array("h")
    a.frombytes(frames)
    start = next((i for i in range(len(a)) if abs(a[i]) > TRIM_THRESHOLD), None)
    if start is None:
        return frames
    end = next(i for i in range(len(a) - 1, -1, -1) if abs(a[i]) > TRIM_THRESHOLD)
    pad = int(TRIM_PAD * SAMPLE_RATE)
    a = a[max(0, start - pad):end + pad + 1]
    rms = math.sqrt(sum(x * x for x in a) / len(a))
    gain = min(TARGET_RMS / rms, 32000 / max(map(abs, a)), 4.0)
    if abs(gain - 1) > 0.05:
        a = array("h", (int(x * gain) for x in a))
    return a.tobytes()


def _seconds(frames: bytes) -> float:
    return len(frames) / (2 * SAMPLE_RATE)


def gap_after(line: dict) -> float:
    """Pause before the next speaker: about 0.2 s, shorter after a quick reaction, a little
    longer after a question, with a fixed per-line jitter so it doesn't sound metronomic."""
    if len(line["display_text"].split()) <= 3:
        base = 0.12
    elif line.get("kind") in ("question", "misconception"):
        base = 0.35
    else:
        base = 0.22
    return round(base + random.Random(line["id"]).uniform(-0.05, 0.05), 3)


def split_for_tts(text: str, max_chars: int) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]
    pieces = []
    for sentence in split_sentences(text):
        for part in re.split(r"(?<=[,;:])\s+", sentence) if len(sentence) > max_chars else [sentence]:
            while len(part) > max_chars:
                cut = part.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                pieces.append(part[:cut].strip())
                part = part[cut:].strip()
            if part:
                pieces.append(part)
    packed = []
    for piece in pieces:
        if packed and len(packed[-1]) + 1 + len(piece) <= max_chars:
            packed[-1] += " " + piece
        else:
            packed.append(piece)
    return packed


def _timestamp(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _vtt_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_cues(timed_lines: list, hosts: dict) -> list:
    """Captions of at most two 42-character lines, never spanning a speaker change. Time is
    shared out by how long each sentence takes to SAY ("3:1" is short on screen but long
    aloud), then by characters within a sentence. Cues shorter than 0.83 s join a neighbour."""
    cues = []
    for line in timed_lines:
        shown = split_sentences(line["display_text"]) or [line["display_text"]]
        said = split_sentences(line.get("spoken_text", ""))
        weights = [len(x) for x in (said if len(said) == len(shown) else shown)]
        duration = line["end"] - line["start"]
        pieces = []  # (text, seconds)
        for sentence, weight in zip(shown, weights):
            parts = split_for_tts(sentence, MAX_CUE_CHARS)
            span = duration * weight / sum(weights)
            pieces += [(part, span * len(part) / sum(len(p) for p in parts)) for part in parts]
        packed = []
        for text, sec in pieces:
            if packed and (len(packed[-1][0]) + 1 + len(text) <= MAX_CUE_CHARS or packed[-1][1] < MIN_CUE_SECONDS):
                packed[-1] = (packed[-1][0] + " " + text, packed[-1][1] + sec)
            else:
                packed.append((text, sec))
        if len(packed) > 1 and packed[-1][1] < MIN_CUE_SECONDS:
            text, sec = packed.pop()
            packed[-1] = (packed[-1][0] + " " + text, packed[-1][1] + sec)
        t = line["start"]
        for text, sec in packed:
            cues.append({"start": t, "end": t + sec, "speaker": hosts[line["speaker"]], "text": text, "line_id": line["id"]})
            t += sec
        cues[-1]["end"] = line["end"]  # no float drift at the line's end
    return cues


def write_vtt(cues: list, path: Path) -> None:
    out = ["WEBVTT", ""]
    for i, cue in enumerate(cues, 1):
        out += [str(i), f"{_timestamp(cue['start'], '.')} --> {_timestamp(cue['end'], '.')}",
                f"<v {cue['speaker']}>{_vtt_escape(cue['text'])}", ""]
    path.write_text("\n".join(out), encoding="utf-8")


def write_srt(cues: list, path: Path) -> None:
    out = []
    for i, cue in enumerate(cues, 1):
        out += [str(i), f"{_timestamp(cue['start'], ',')} --> {_timestamp(cue['end'], ',')}",
                f"{cue['speaker']}: {cue['text']}", ""]
    path.write_text("\n".join(out), encoding="utf-8")


def duration_outliers(chunks: list) -> dict:
    """chunks: [(index, voice, text, seconds)]. Returns {index: duration / expected} for clips
    far longer (looping, runaway audio) or shorter (dropped words) than that voice's usual
    speaking rate predicts. Needs five judgeable clips per voice to know the rate."""
    flagged = {}
    for voice in {c[1] for c in chunks}:
        mine = [c for c in chunks if c[1] == voice and len(c[2]) >= QC_MIN_CHARS and c[3] > 0]
        if len(mine) < 5:
            continue
        rate = statistics.median(len(text) / sec for _, _, text, sec in mine)
        for i, _, text, sec in mine:
            ratio = sec / (len(text) / rate)
            if not QC_RANGE[0] <= ratio <= QC_RANGE[1]:
                flagged[i] = ratio
    return flagged


class Renderer:
    def __init__(self, tts, voices: dict, language: str, cache_dir: Path, max_chars: int = 250, workers: int = 3, log=print):
        self.tts, self.voices, self.language = tts, voices, language
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_chars, self.workers, self.log = max_chars, workers, log

    def _clip(self, text: str, voice: str, take: int = 0) -> Path:
        """Synthesises one chunk, cached by (voice, language, text, take) so re-runs don't spend credit."""
        key = hashlib.sha256(f"{voice}|{self.language}|{text}{f'|{take}' if take else ''}".encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.wav"
        if not path.exists():
            audio = self.tts.synthesize(text, voice=voice, language=self.language)
            partial = self.cache_dir / f"{key}.part.wav"
            _to_pcm_wav(audio, partial)
            partial.replace(path)
        return path

    def _retake(self, text: str, voice: str, frames: bytes, rate: float) -> tuple[bytes, bool]:
        """Re-synthesises a suspicious clip and keeps the take closest to the expected length."""
        def off(f):  # how far from the expected duration, in log terms
            return abs(math.log(max(_seconds(f), 0.01) / (len(text) / rate)))
        best = frames
        for take in range(1, QC_RETAKES + 1):
            candidate = trim_and_level(_read_frames(self._clip(text, voice, take)))
            if off(candidate) < off(best):
                best = candidate
        ratio = _seconds(best) / (len(text) / rate)
        return best, QC_RANGE[0] <= ratio <= QC_RANGE[1]

    def render(self, script: dict, out_dir: Path) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        jobs = []  # (line, chunks) in playback order
        for seg in script["segments"]:
            for line in seg["lines"]:
                jobs.append((line, split_for_tts(line["spoken_text"], self.max_chars)))
        flat = [(chunk, self.voices[line["speaker"]], line["id"]) for line, chunks in jobs for chunk in chunks]
        chars = sum(len(c) for c, _, _ in flat)
        self.log(f"  Synthesising {len(flat)} clips ({chars:,} characters)")

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._clip, text, voice) for text, voice, _ in flat]
            clip_paths = []
            for n, fut in enumerate(futures, 1):
                clip_paths.append(fut.result())
                if n % 20 == 0 or n == len(futures):
                    self.log(f"    {n}/{len(futures)} clips ready")
        clips = [trim_and_level(_read_frames(p)) for p in clip_paths]

        # Duration check: retake clips far off the voice's usual pace, flag any still off.
        outliers = duration_outliers([(i, voice, text, _seconds(clips[i])) for i, (text, voice, _) in enumerate(flat)])
        qc_flagged = []
        if outliers:
            self.log(f"  {len(outliers)} clip(s) with an unusual length, retaking")
        for i in outliers:
            text, voice, line_id = flat[i]
            mine = [(t, s) for (t, v, _), s in zip(flat, map(_seconds, clips)) if v == voice and len(t) >= QC_MIN_CHARS]
            rate = statistics.median(len(t) / s for t, s in mine)
            clips[i], ok = self._retake(text, voice, clips[i], rate)
            if not ok and line_id not in qc_flagged:
                qc_flagged.append(line_id)

        master = bytearray()
        now = lambda: len(master) / (2 * SAMPLE_RATE)
        silence = lambda sec: master.extend(b"\x00\x00" * int(sec * SAMPLE_RATE))

        clip_iter = iter(clips)
        job_iter = iter(jobs)
        timed_lines, timed_segments = [], []
        silence(0.4)
        for s_no, seg in enumerate(script["segments"]):
            if s_no:
                silence(GAP_BETWEEN_SEGMENTS)
            seg_start = now()
            for l_no in range(len(seg["lines"])):
                line, chunks = next(job_iter)
                if l_no:
                    silence(gap_after(seg["lines"][l_no - 1]))
                start = now()
                for j in range(len(chunks)):
                    if j:
                        silence(GAP_WITHIN_LINE)
                    master.extend(next(clip_iter))
                timed_lines.append({**line, "segment_id": seg["id"], "start": round(start, 3), "end": round(now(), 3)})
            timed_segments.append({"id": seg["id"], "title": seg["title"], "start": round(seg_start, 3), "end": round(now(), 3)})
        silence(0.6)

        master_wav = out_dir / "master.wav"
        with wave.open(str(master_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(bytes(master))
        self.log("  Encoding MP3 with loudness normalisation")
        _ffmpeg("-i", str(master_wav), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "44100", "-ac", "1",
                "-codec:a", "libmp3lame", "-b:a", "96k", str(out_dir / "episode.mp3"))
        master_wav.unlink()

        hosts = script["hosts"]
        cues = build_cues(timed_lines, hosts)
        write_vtt(cues, out_dir / "episode.vtt")
        write_srt(cues, out_dir / "episode.srt")

        manifest = {
            "title": script["title"],
            "language": script.get("language"),
            "duration": round(now(), 3),
            "audio": "episode.mp3",
            "captions": {"vtt": "episode.vtt", "srt": "episode.srt"},
            "loudness_lufs": _loudness(out_dir / "episode.mp3"),
            "hosts": hosts,
            "segments": timed_segments,
            "lines": [
                {k: l[k] for k in ("id", "segment_id", "speaker", "display_text", "kind", "fact_ids", "start", "end")}
                | {"speaker_name": hosts[l["speaker"]]}
                for l in timed_lines
            ],
            # Lines cite fact ids; the quote is the chapter's own sentence, for a "from the chapter" view.
            "facts": [{k: f.get(k, "") for k in ("id", "section_id", "text", "quote")} for f in script.get("facts", [])],
            "qc": {"retakes": len(outliers), "audio_flagged_lines": qc_flagged},
            "needs_review": bool(script.get("unresolved_issues") or qc_flagged),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest
