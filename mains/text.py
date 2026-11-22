"""Pure text helpers shared by every stage. No model and no I/O, so every rule here is unit-tested.

The quality rules in `clean` are what keep vague or invented feedback out of a student's report: a model field that
breaks one is rejected, and the model gets one corrected attempt before the check is marked unavailable.
"""
import re
import unicodedata
from collections import Counter

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
DEV_WORD = re.compile(r"[\u0900-\u097F]+")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'\u2019-]*")
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")
SOURCE_ID = re.compile(r"\b[sSpP]\d{1,3}\b")
# Numeric marks or grades in feedback text ("6/10", "out of 10", "10 में से 6", "7 marks", "7 अंक").
MARKS = re.compile(r"\d+(?:\.\d+)?\s*/\s*\d+|\bout of \d+|\d+\s*में से\s*\d+|\d+\s*(?:marks?\b|अंक)", re.I)
NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)*)(\s*%|\s*per\s*cent|\s*प्रतिशत)?", re.I)
TOKEN_SPLIT = re.compile(r"[\s,.;:!?।()\[\]\"'\u201c\u201d\u2018\u2019\u2014\u2013-]+")
# Praise or filler that would fit any answer. Feedback must point at the student's own words instead.
FILLER = (
    "keep up the good work", "good attempt", "well done", "great job", "nice answer", "overall a good answer",
    "overall good answer", "needs improvement overall", "as an ai", "language model",
    "अच्छा प्रयास", "बहुत बढ़िया", "शाबाश", "कुल मिलाकर अच्छा",
)


class Rejected(ValueError):
    """A model output broke a quality rule. `code` is a fixed label (never model or student text), safe to log."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def nfc(value):
    return unicodedata.normalize("NFC", value) if isinstance(value, str) else ""


def norm(value):
    return re.sub(r"\s+", " ", nfc(value)).strip()


def fold(value):
    return norm(value).casefold()


def word_count(value):
    return len(re.findall(r"\S+", nfc(value)))


def sentences(answer, max_words=55):
    """Number the answer's sentences s1..sN in order, keeping the student's exact wording and paragraph."""
    out = []
    paragraphs = [p for p in re.split(r"\n\s*\n", nfc(answer).strip()) if p.strip()]
    for para, block in enumerate(paragraphs, 1):
        for line in block.splitlines():
            for piece in SENTENCE_END.split(line.strip()):
                for part in _split_long(piece.strip(), max_words):
                    if part:
                        out.append({"id": f"s{len(out) + 1}", "text": part, "para": para})
    return out


def _split_long(piece, max_words):
    if len(piece.split()) <= max_words:
        return [piece]
    parts = [p for p in re.split(r"(?<=[;:])\s+", piece) if p.strip()]
    return parts if len(parts) > 1 else [piece]


def _numbers(value):
    found = []
    for m in NUMBER.finditer(nfc(value).translate(DEV_DIGITS)):
        found.append((m.group(1).replace(",", ""), bool(m.group(2))))
    return found


def unsourced_numbers(value, *sources):
    """Numbers in feedback that appear nowhere in the question, answer or reference points.

    Small counts (1-12, e.g. "two parts", "Part 2") are allowed; percentages, decimals, years and other figures must
    come from the materials, so the model cannot invent a statistic or a date.
    """
    allowed = {v for s in sources for v, _ in _numbers(s)}
    bad = []
    for v, percent in _numbers(value):
        small = not percent and v.isdigit() and int(v) <= 12
        if v not in allowed and not small:
            bad.append(v)
    return bad


def script_problem(value, lang, *sources):
    """Hindi feedback must be Devanagari (Roman only for acronyms or the student's own words); English must not
    switch script except to quote the materials."""
    src = fold(" ".join(sources))
    if lang == "hi":
        if len(DEVANAGARI.findall(value)) < 6:
            return "not_hindi"
        for w in LATIN_WORD.findall(value):
            if not (w.isupper() and len(w) <= 6) and w.casefold() not in src:
                return "roman_in_hindi"
    else:
        for w in DEV_WORD.findall(value):
            if fold(w) not in src:
                return "devanagari_in_english"
        if len(LATIN_WORD.findall(value)) < 2:
            return "not_english"
    return None


ALLOWED_NAMES = {"i", "india", "indian", "indians", "part", "parts", "mains", "prelims", "english", "hindi",
                 "upsc", "gs", "gdp"}
WORD = re.compile(r"[A-Za-z][A-Za-z'\u2019-]*")
STARTERS = ".!?\u0964:;\"'\u201c\u2018("


def unsourced_names(value, *sources):
    """Proper names and acronyms in feedback that the question, answer and reference points never mention.

    A capitalised word in the middle of a sentence (a scheme, river, city, court, report) or an all-caps acronym must
    come from the materials; otherwise the model is supplying a fact nobody can check against the student's work.
    """
    allowed = {t.casefold() for s in sources for t in WORD.findall(nfc(s))} | ALLOWED_NAMES
    text = nfc(value)
    bad = []
    for m in WORD.finditer(text):
        w = m.group(0)
        if w.casefold() in allowed:
            continue
        before = text[:m.start()].rstrip()
        sentence_start = not before or before[-1] in STARTERS
        if (len(w) >= 3 and w.isupper()) or (w[0].isupper() and not sentence_start):
            bad.append(w)
    return bad


def degenerate(value):
    """Looping or stuttering output: a repeated 4-word run, a duplicated sentence, or a long run of one character."""
    words = fold(value).split()
    if len(words) >= 12:
        grams = Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
        if grams.most_common(1)[0][1] >= 3:
            return True
    parts = [fold(s) for s in SENTENCE_END.split(value) if s.strip()]
    if len(parts) != len(set(parts)):
        return True
    return bool(re.search(r"(\w)\1{5,}", value))


def tokens(value):
    return {t for t in TOKEN_SPLIT.split(fold(value)) if t}


def similar(a, b, threshold=0.6):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def sentence_count(value):
    return len([s for s in SENTENCE_END.split(norm(value)) if s.strip()])


def clean(value, lang, sources, lo=8, hi=600, field="text", allow_empty=False):
    """Return the stripped text, or raise Rejected with a fixed code naming the broken rule."""
    if not isinstance(value, str):
        raise Rejected(f"{field}:type")
    out = norm(value)
    if not out:
        if allow_empty:
            return ""
        raise Rejected(f"{field}:empty")
    if not lo <= len(out) <= hi:
        raise Rejected(f"{field}:length")
    if SOURCE_ID.search(out):
        raise Rejected(f"{field}:source_id")
    if MARKS.search(out):
        raise Rejected(f"{field}:marks")
    if degenerate(out):
        raise Rejected(f"{field}:repetition")
    if any(p in fold(out) for p in FILLER):
        raise Rejected(f"{field}:filler")
    problem = script_problem(out, lang, *sources)
    if problem:
        raise Rejected(f"{field}:{problem}")
    if unsourced_numbers(out, *sources):
        raise Rejected(f"{field}:unsourced_number")
    if unsourced_names(out, *sources):
        raise Rejected(f"{field}:unsourced_name")
    return out
