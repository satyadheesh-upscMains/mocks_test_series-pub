"""Reference extracts for the examiner: the Constitution of India and NCERT Class 11 Political Science.

`tools/build_refs.py` builds refs/index.json.gz once per runner: the Constitution comes from the committed JSON (public
domain); NCERT chapters are downloaded from ncert.nic.in at build time, so no textbook text is ever committed. Search
is plain BM25 in Python, no model. Passages go to Gemma inside MATERIALS only: reports carry a passage's label (for
example "Constitution of India, Article 243K"), never its text, and logs print counts only.
"""
import collections
import gzip
import json
import math
import os
import re
from pathlib import Path

INDEX = Path(os.environ.get("MAINS_REFS", Path(__file__).resolve().parents[1] / "refs" / "index.json.gz"))
MAX_PASSAGES = 6
FROM_QUESTION_FIRST = 2   # passages for the question itself, before the answer's claims get theirs
CLAIM_SCORE = 0.6         # a claim's passage may score lower than a question's: claims are one short sentence
TOPIC_HITS = 15           # the question's top hits that define its topic (Articles, Schedules, NCERT chapters)
PASSAGE_WORDS = 130
MIN_SCORE = 10.5      # tuned on sample GS2 and non-polity questions: below this, the hits are noise
MIN_COVERAGE = 0.33   # share of the question's (idf-weighted) terms a passage must contain

STOP = set("""a an the of and or to in on for by with as at from is are was were be been being it its this that these
those which who whom whose what when where why how not no nor but if then than so such into over under about between
through during before after above below up down out off again further once here there all any both each few more most
other some own same too very can will just should would could may might must shall do does did doing have has had
having i you he she we they them their our your his her my me us also only upon whereas thereof therein said section
clause discuss examine explain analyse analyze evaluate critically comment suggest illustrate marks words""".split())
ARTICLE = re.compile(r"(?:\bart(?:icle)?s?\.?|अनुच्छेद)\s*(\d{1,3}[A-Z]{0,3})\b", re.I)
BOOK = {"keps1": "Political Theory", "keps2": "Indian Constitution at Work"}
NCERT_FILES = [f"keps10{i}" for i in (1, 2, 3, 4, 6, 7, 8)] + [f"keps2{i:02d}" for i in range(1, 11)]


def _stem(w):
    for suf in ("ations", "ation", "ments", "ment", "ities", "ity", "ings", "ing", "ies", "ive", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def terms(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in STOP and len(w) > 1]


# ------------------------------------------------------------------ building the index (tools/build_refs.py)

def _label(article):
    if article == "PREAMBLE":
        return "Constitution of India, Preamble"
    if article.startswith("SCH"):
        return f"Constitution of India, Schedule {article[3:]}"
    return f"Constitution of India, Article {article}"


def constitution_chunks(entries):
    out = []
    for e in entries:
        body = re.sub(r"\s+", " ", e.get("description") or "").strip()
        body = re.sub(r"\s+PART [IVXLC]+[A-Z]? [A-Z ,—.\-]+$", "", body)  # the next Part's heading, run into the text
        words, head = body.split(), f"{e['title']}. "
        art = str(e["article"])
        if len(words) <= 220:
            out.append({"id": "c" + art, "label": _label(art), "title": e["title"], "text": head + body})
            continue
        for k, i in enumerate(range(0, len(words), 180), 1):
            out.append({"id": f"c{art}.{k}", "label": _label(art), "title": e["title"],
                        "text": head + " ".join(words[i:i + 220])})
    return out


def _clean_page(text):
    text = (text or "").replace("\u00ad", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    keep = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or re.fullmatch(r"\d{1,3}", line) or line.startswith("±"):
            continue
        if re.fullmatch(r"(Indian Constitution at Work|Political Theory|Chapter \d+:.*|Reprint 20\d\d-\d\d)", line):
            continue
        keep.append(line)
    text = " ".join(keep)
    text = re.sub(r"Reprint 20\d\d-\d\d", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ncert_chunks(pages):
    """pages: [{"file": "keps208", "page": 1, "text": ...}] in page order."""
    by_file = collections.OrderedDict()
    for p in pages:
        by_file.setdefault(p["file"], []).append(p)
    out = []
    for f, ps in by_file.items():
        m = next((re.search(r"Chapter (\d+):\s*([^\n]+)", p["text"] or "") for p in ps
                  if re.search(r"Chapter (\d+):", p["text"] or "")), None)
        n, title = (int(m.group(1)), m.group(2).strip()) if m else (int(f[5:7]), "")
        label = f"NCERT Class 11, {BOOK.get(f[:5], 'Political Science')}, Ch. {n}" + (f" ({title})" if title else "")
        words = " ".join(_clean_page(p["text"]) for p in ps).split()
        for k, i in enumerate(range(0, len(words), 150), 1):
            out.append({"id": f"n{f[4:7]}.{k}", "label": label, "title": title, "text": " ".join(words[i:i + 180])})
    return out


# ------------------------------------------------------------------ search

class Index:
    def __init__(self, chunks, min_score=MIN_SCORE, min_coverage=MIN_COVERAGE):
        self.chunks = list(chunks)
        self.min_score, self.min_coverage = min_score, min_coverage
        self.docs = [collections.Counter(terms(c["title"] + " " + c["text"])) for c in self.chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df = collections.Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - k + 0.5) / (k + 0.5)) for t, k in df.items()}
        self.by_article = {}
        for i, c in enumerate(self.chunks):
            m = re.match(r"c(\w+?)(?:\.\d+)?$", c["id"])
            if m:
                self.by_article.setdefault(m.group(1).upper(), i)

    def search(self, query, k1=1.4, b=0.75):
        weights = {t: self.idf[t] for t in set(terms(query)) if t in self.idf}
        total = sum(weights.values()) or 1.0
        hits = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t, w in weights.items():
                tf = d.get(t)
                if tf:
                    s += w * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.lens[i] / self.avg))
            if s:
                hits.append((s, sum(w for t, w in weights.items() if t in d) / total, i))
        hits.sort(reverse=True)
        # Textbook exercise pages are mostly questions to the reader: they carry no facts to check against.
        return [(s, cov, i) for s, cov, i in hits if self.chunks[i]["text"].count("?") < 3]

    def article(self, number):
        i = self.by_article.get(str(number).upper())
        return None if i is None else self.chunks[i]

    def pack(self, question, answer=""):
        chosen = []

        def add(i):
            if i is not None and i not in chosen and len(chosen) < MAX_PASSAGES:
                chosen.append(i)

        # Articles named in the question or the answer: the text of the Article itself, to check the claim against.
        for m in ARTICLE.finditer(f"{question} {answer}"):
            add(self.by_article.get(m.group(1).upper()))
        hits = self.search(question)
        on_topic = [i for s, cov, i in hits if s >= self.min_score and cov >= self.min_coverage]
        if not on_topic:
            return self._out(chosen)
        for i in on_topic[:FROM_QUESTION_FIRST]:
            add(i)
        # The answer's own claims are where its errors are, so each claim gets the passage that can check it. A claim
        # is searched on its own words but only within the question's topic (the same Part of the Constitution or
        # the same NCERT chapter): "conducted by the Election Commission of India" in an answer on panchayats finds
        # Article 243K (State Election Commission), not Article 324 (run 45223096235 missed it when the claim was
        # searched together with the question).
        families = {family(self.chunks[i]) for s, _, i in hits[:TOPIC_HITS] if s >= self.min_score * CLAIM_SCORE}
        for sent in re.split(r"(?<=[.!?\u0964])\s+", answer or ""):
            for s, _, i in self.search(sent):
                if s < self.min_score * CLAIM_SCORE:
                    break
                if family(self.chunks[i]) in families and i not in chosen:
                    add(i)
                    break
        for i in on_topic[FROM_QUESTION_FIRST:]:
            add(i)
        return self._out(chosen)

    def _out(self, chosen):
        out = []
        for n, i in enumerate(chosen, 1):
            c = self.chunks[i]
            out.append({"id": f"r{n}", "label": c["label"], "text": " ".join(c["text"].split()[:PASSAGE_WORDS])})
        return out


def family(chunk):
    """The topic a passage belongs to: an Article and its sub-articles (243, 243K, 243ZA are all Part IX's 243
    family), a Schedule, or one NCERT chapter."""
    m = re.match(r"c(SCH\d+|\d+)", chunk["id"]) or re.match(r"(n\d+)", chunk["id"])
    return m.group(1) if m else chunk["id"]


_LOADED = {}


def load(path=None):
    path = Path(path or INDEX)
    if path not in _LOADED:
        try:
            _LOADED[path] = Index(json.loads(gzip.decompress(path.read_bytes())))
        except (OSError, ValueError, EOFError):
            _LOADED[path] = None
    return _LOADED[path]


def pack(question, answer="", index=None):
    idx = index if index is not None else load()
    return idx.pack(question, answer) if idx else []
