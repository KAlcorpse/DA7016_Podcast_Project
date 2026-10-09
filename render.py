"""script.json -> episode.mp3 + episode.vtt + episode.srt + manifest.json

Each line is synthesised separately (Bodhan recommends short TTS inputs), converted
to one common PCM format, and stitched with short pauses. Because we place every clip
ourselves, we know exactly when each line starts and ends, which gives caption
timestamps without any extra alignment step.
"""
import hashlib
import json
import os
import re
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clean import split_sentences

SAMPLE_RATE = 24000
GAP_WITHIN_LINE = 0.12   # seconds between TTS chunks of one line
GAP_BETWEEN_LINES = 0.35
GAP_BETWEEN_SEGMENTS = 0.9
MAX_CUE_SECONDS = 7.0    # longer lines are split into several caption cues


class RenderError(RuntimeError):
    pass


def _ffmpeg(*args: str) -> None:
    result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RenderError(f"ffmpeg failed: {result.stderr[-800:]}")


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
    cues = []
    for line in timed_lines:
        duration = line["end"] - line["start"]
        sentences = split_sentences(line["display_text"]) or [line["display_text"]]
        groups = sentences if duration > MAX_CUE_SECONDS and len(sentences) > 1 else [line["display_text"]]
        total_chars = sum(len(g) for g in groups)
        t = line["start"]
        for group in groups:
            d = duration * len(group) / total_chars
            cues.append({"start": t, "end": t + d, "speaker": hosts[line["speaker"]], "text": group, "line_id": line["id"]})
            t += d
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


class Renderer:
    def __init__(self, tts, voices: dict, language: str, cache_dir: Path, max_chars: int = 250, workers: int = 3, log=print):
        self.tts, self.voices, self.language = tts, voices, language
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_chars, self.workers, self.log = max_chars, workers, log

    def _clip(self, text: str, voice: str) -> Path:
        """Synthesises one chunk, cached by (voice, language, text) so re-runs don't spend credit."""
        key = hashlib.sha256(f"{voice}|{self.language}|{text}".encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.wav"
        if not path.exists():
            audio = self.tts.synthesize(text, voice=voice, language=self.language)
            partial = self.cache_dir / f"{key}.part.wav"
            _to_pcm_wav(audio, partial)
            partial.replace(path)
        return path

    def render(self, script: dict, out_dir: Path) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        jobs = []  # (line, chunks) in playback order
        for seg in script["segments"]:
            for line in seg["lines"]:
                jobs.append((line, split_for_tts(line["spoken_text"], self.max_chars)))
        flat = [(chunk, self.voices[line["speaker"]]) for line, chunks in jobs for chunk in chunks]
        chars = sum(len(c) for c, _ in flat)
        self.log(f"  Synthesising {len(flat)} clips ({chars:,} characters)")

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._clip, text, voice) for text, voice in flat]
            clip_paths = []
            for n, fut in enumerate(futures, 1):
                clip_paths.append(fut.result())
                if n % 20 == 0 or n == len(futures):
                    self.log(f"    {n}/{len(futures)} clips ready")

        master = bytearray()
        now = lambda: len(master) / (2 * SAMPLE_RATE)
        silence = lambda sec: master.extend(b"\x00\x00" * int(sec * SAMPLE_RATE))

        clips = iter(clip_paths)
        job_iter = iter(jobs)
        timed_lines, timed_segments = [], []
        silence(0.4)
        for seg in script["segments"]:
            seg_start = now()
            for _ in seg["lines"]:
                line, chunks = next(job_iter)
                start = now()
                for j in range(len(chunks)):
                    if j:
                        silence(GAP_WITHIN_LINE)
                    master.extend(_read_frames(next(clips)))
                timed_lines.append({**line, "segment_id": seg["id"], "start": round(start, 3), "end": round(now(), 3)})
                silence(GAP_BETWEEN_LINES)
            timed_segments.append({"id": seg["id"], "title": seg["title"], "start": round(seg_start, 3), "end": round(now(), 3)})
            silence(GAP_BETWEEN_SEGMENTS)

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
            "hosts": hosts,
            "segments": timed_segments,
            "lines": [
                {k: l[k] for k in ("id", "segment_id", "speaker", "display_text", "kind", "fact_ids", "start", "end")}
                | {"speaker_name": hosts[l["speaker"]]}
                for l in timed_lines
            ],
            "needs_review": bool(script.get("unresolved_issues")),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest
