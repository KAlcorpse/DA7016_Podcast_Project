"""Cleans text pasted from a PDF and splits it into sections.

Output of prepare_sections(): [{"id": "8.1", "title": "...", "text": "...", "words": 412}, ...]
This list is the ONLY knowledge the rest of the pipeline is allowed to use.
"""
import re
from collections import Counter

# "8.1 Accumulation of Variation" / "8.2.1 Inherited Traits"
HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,2})\s+([A-Z][^\n]{2,90})$")
PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,3}$", re.I)
REPRINT_FOOTER = re.compile(r"^reprint\s+\d{4}\s*[-–]\s*\d{2,4}$", re.I)
# Print-shop slugs in newer NCERT PDFs: "Chapter-10.indd 184" and "03-Apr-26 8:04:39 PM"
PRINT_SLUG = re.compile(r"^(\S+\.indd\s+\d+|\d{1,2}-[a-z]{3}-\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?\s*([ap]m)?)$", re.I)

# Pasted text is untrusted: drop invisible characters that can hide instructions (ZWJ/ZWNJ
# stay, Indic scripts need them) and any of our own prompt tags, so the text can't close
# <source> early and pose as instructions.
INVISIBLE = re.compile(r"[\u200b\u2060\ufeff\u202a-\u202e\u2066-\u2069]")
PROMPT_TAGS = re.compile(r"</?\s*(source|facts|script|lines|flagged_lines|previous_lines|episode_goals)\s*>", re.I)

MAX_SECTION_WORDS = 900      # longer sections are split so each LLM call stays small
FALLBACK_SECTION_WORDS = 500  # used when the paste has no numbered headings

_ABBREVIATIONS = ("fig", "e.g", "i.e", "etc", "vs", "no", "mr", "mrs", "dr", "approx", "eq")
_ABBR_END = re.compile(r"(?:^|\s)(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.$", re.I)


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that doesn't break on 'Fig.' or 'e.g.'; also handles the Hindi danda."""
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    sentences: list[str] = []
    for part in parts:
        if sentences and _ABBR_END.search(sentences[-1]):
            sentences[-1] += " " + part
        elif part:
            sentences.append(part)
    return sentences


def _clean_lines(raw: str) -> list[str]:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "").replace("\t", " ")
    text = PROMPT_TAGS.sub(" ", INVISIBLE.sub("", text))
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)  # re-join words hyphenated across lines
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    # Short lines that repeat 3+ times are usually running headers/footers from the PDF.
    counts = Counter(line for line in lines if line)
    repeated = {line for line, n in counts.items() if n >= 3 and len(line) <= 60 and not HEADING.match(line)}
    kept = []
    for line in lines:
        if line and (PAGE_NUMBER.match(line) or REPRINT_FOOTER.match(line) or PRINT_SLUG.match(line) or line in repeated):
            continue
        kept.append(line)
    return kept


def _join_paragraphs(lines: list[str]) -> str:
    paragraphs, current = [], []
    for line in lines:
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _chunk(text: str, max_words: int) -> list[str]:
    chunks, current, count = [], [], 0
    for sentence in split_sentences(text.replace("\n\n", " ")):
        n = len(sentence.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def prepare_sections(raw: str) -> list[dict]:
    lines = _clean_lines(raw)
    found: list[dict] = []
    current = {"id": "intro", "title": "Introduction", "lines": []}
    for line in lines:
        match = HEADING.match(line)
        if match and not match.group(2).rstrip().endswith("."):
            found.append(current)
            current = {"id": match.group(1), "title": match.group(2).strip(), "lines": []}
        else:
            current["lines"].append(line)
    found.append(current)

    sections, seen_ids = [], Counter()
    for sec in found:
        text = _join_paragraphs(sec["lines"])
        words = len(text.split())
        if words == 0 or (sec["id"] == "intro" and words < 25 and len(found) > 1):
            continue
        seen_ids[sec["id"]] += 1
        sec_id = sec["id"] if seen_ids[sec["id"]] == 1 else f"{sec['id']}#{seen_ids[sec['id']]}"
        sections.append({"id": sec_id, "title": sec["title"], "text": text})

    # No numbered headings at all: fall back to evenly sized parts.
    if len(sections) == 1 and sections[0]["id"] == "intro":
        parts = _chunk(sections[0]["text"], FALLBACK_SECTION_WORDS)
        sections = [{"id": f"part{i}", "title": f"Part {i}", "text": p} for i, p in enumerate(parts, 1)]

    # Split very long sections so each extraction call stays manageable.
    final = []
    for sec in sections:
        if len(sec["text"].split()) <= MAX_SECTION_WORDS:
            final.append(sec)
            continue
        for i, piece in enumerate(_chunk(sec["text"], MAX_SECTION_WORDS)):
            final.append({"id": f"{sec['id']}{chr(ord('a') + i)}", "title": f"{sec['title']} (part {i + 1})", "text": piece})

    for sec in final:
        sec["words"] = len(sec["text"].split())
    return final
