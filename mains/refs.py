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
MAX_PASSAGES = 6     # extracts shown to every check
MAX_REFS = 16        # with the extracts that check single claims
CLAIM_SOURCES = 5    # extracts per claim
TITLE_WEIGHT = 3.0  # how much a match in an Article's title adds
FROM_QUESTION_FIRST = 2   # passages for the question itself, before the answer's claims get theirs
CLAIM_SCORE = 0.6         # a claim's passage may score lower than a question's: claims are one short sentence
PASSAGE_WORDS = 130
MIN_SCORE = 10.5      # tuned on sample GS2 and non-polity questions: below this, the hits are noise
MIN_COVERAGE = 0.3   # share of the question's (idf-weighted) terms a passage must contain

STOP = set("""a an the of and or to in on for by with as at from is are was were be been being it its this that these
those which who whom whose what when where why how not no nor but if then than so such into over under about between
through during before after above below up down out off again further once here there all any both each few more most
other some own same too very can will just should would could may might must shall do does did doing have has had
having i you he she we they them their our your his her my me us also only upon whereas thereof therein said section
clause discuss examine explain analyse analyze evaluate critically comment suggest illustrate marks words
position role significance impact structure system challenges remain issues relevance importance way ways
measures steps need india indian scope widen widened constitution constitutional""".split())
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
        # An Article's own title says what it is about ("Appointment of Governor"). Titles get their own index, with
        # their own rarity weights: "Governor" is common in the text of the Constitution but rare in its titles, so a
        # question on the Governor finds Articles 153-156 before the many Articles that only mention a Governor.
        self.heads = [collections.Counter(terms(c["title"])) for c in self.chunks]
        hdf = collections.Counter(t for d in self.heads for t in d)
        self.hidf = {t: math.log(1 + (len(self.heads) - k + 0.5) / (k + 0.5)) for t, k in hdf.items()}
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

    def search(self, query, k1=1.4, b=0.75, titles=True):
        weights = {t: self.idf[t] for t in set(terms(query)) if t in self.idf}
        total = sum(weights.values()) or 1.0
        both = {t: max(w, self.hidf.get(t, 0)) for t, w in weights.items()}
        total_both = sum(both.values()) or 1.0
        hits = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t, w in weights.items():
                tf = d.get(t)
                if tf:
                    s += w * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.lens[i] / self.avg))
            head = self.heads[i]
            if titles:  # a claim is matched on its words only: a title like "Constitution of Panchayats" says little
                s += TITLE_WEIGHT * sum(self.hidf.get(t, 0) for t in weights if t in head) / max(1, len(head)) ** 0.5
            if s:
                # Coverage: how much of the question this passage is about, with a term counted when it is in the text
                # or the title, at its larger weight.
                cov = sum(both[t] for t in both if t in d or t in head) / total_both
                hits.append((s, cov, i))
        hits.sort(reverse=True)
        # Textbook exercise pages are mostly questions to the reader: they carry no facts to check against.
        return [(s, cov, i) for s, cov, i in hits if self.chunks[i]["text"].count("?") < 3]

    def article(self, number):
        i = self.by_article.get(str(number).upper())
        return None if i is None else self.chunks[i]

    def pack(self, question, answer=""):
        """Extracts for one answer. `answer` is its text or its numbered sentences ([{"id": "s1", "text": ...}]).

        Returns up to MAX_PASSAGES shared extracts (shown to every check), then extracts used only to check single
        claims ("shared": False). Each extract lists the answer sentences it may check ("claims")."""
        sents = answer if isinstance(answer, list) else [
            {"id": f"s{n}", "text": t} for n, t in enumerate(re.split(r"(?<=[.!?\u0964])\s+", answer or ""), 1) if t]
        order, shared, claims = [], set(), {}

        def take(i, share):
            # One extract per Article or chapter label: a second slice of the same Article adds little. Returns the
            # extract that now stands for i (itself, or the one with the same label), or None when there is no room.
            if i is None:
                return None
            same = next((j for j in order if self.chunks[j]["label"] == self.chunks[i]["label"]), None)
            if same is None:
                if len(order) >= MAX_REFS or (share and len(shared) >= MAX_PASSAGES):
                    return None
                order.append(i)
                same = i
            if share and len(shared) < MAX_PASSAGES:
                shared.add(same)
            return same

        def claim(sid, j):
            if j is not None and j not in claims.setdefault(sid, []) and len(claims[sid]) < CLAIM_SOURCES:
                claims[sid].append(j)

        # Articles named in the question or the answer: the text of the Article itself, shown to every check.
        named = []
        for m in ARTICLE.finditer(question + " " + " ".join(s["text"] for s in sents)):
            j = take(self.by_article.get(m.group(1).upper()), True)
            if j is not None and j not in named:
                named.append(j)
        hits = self.search(question)
        on_topic = [i for sc, cov, i in hits if sc >= self.min_score and cov >= self.min_coverage]
        for i in on_topic[:FROM_QUESTION_FIRST]:
            take(i, True)
        # Each claim gets the extracts that can check it, because the answer's claims are where its errors are:
        # the Articles it names, the Articles the question or answer names, and its best matches within the
        # question's topic (the Parts or chapters the question's own extracts come from), by its own words and by its
        # words with the question's. "Conducted by the Election Commission of India" in an answer on panchayats then
        # meets Article 243K (State Election Commission), not Article 324 or Schedule 6, which runs 45223096235 and
        # 45238473782 used instead. Tuned on eleven planted errors in eight Polity answers: all eleven reach a check.
        families = {family(self.chunks[i]) for i in on_topic}
        for s in sents:
            for m in ARTICLE.finditer(s["text"]):
                claim(s["id"], take(self.by_article.get(m.group(1).upper()), True))
            for j in named[:2]:
                claim(s["id"], j)
            if not families:
                continue
            for query, titles in ((s["text"], False), (f"{s['text']} {question}", True)):
                got = 0
                for sc, _, i in self.search(query, titles=titles):
                    if sc < self.min_score * CLAIM_SCORE or got >= 2:
                        break
                    if family(self.chunks[i]) in families:
                        j = take(i, False)
                        if j is not None and j not in claims.get(s["id"], []):
                            claim(s["id"], j)
                            got += 1
        for i in on_topic[FROM_QUESTION_FIRST:]:
            take(i, True)
        return self._out(order, shared, claims)

    def _out(self, order, shared, claims):
        ranked = [i for i in order if i in shared] + [i for i in order if i not in shared]
        out = []
        for n, i in enumerate(ranked, 1):
            c = self.chunks[i]
            words = c["text"].split()
            r = {"id": f"r{n}", "label": c["label"],
                 "text": " ".join(words[:PASSAGE_WORDS] if i in shared else words), "shared": i in shared}
            mine = sorted((sid for sid, js in claims.items() if i in js), key=lambda x: int(x[1:]))
            if mine:
                r["claims"] = mine
            out.append(r)
        return out


# Parts and chapters of the Constitution by Article number: a claim's passage must come from the same Part or chapter
# as the question's own passages (Articles 153-156 on the Governor are one chapter; Part IX is 243-243O).
_RANGES = [("I", 1, 4), ("II", 5, 11), ("III", 12, 35), ("IV", 36, 51), ("IVA", 51, 51), ("V.1", 52, 78),
           ("V.2", 79, 122), ("V.3", 123, 123), ("V.4", 124, 147), ("V.5", 148, 151), ("VI.1", 152, 152),
           ("VI.2", 153, 167), ("VI.3", 168, 212), ("VI.4", 213, 213), ("VI.5", 214, 232), ("VI.6", 233, 237),
           ("VIII", 239, 242), ("IX-X", 243, 244), ("XI", 245, 263), ("XII", 264, 300), ("XIII", 301, 307),
           ("XIV", 308, 323), ("XV", 324, 329), ("XVI", 330, 342), ("XVII", 343, 351), ("XVIII", 352, 360),
           ("XIX", 361, 367), ("XX", 368, 368), ("XXI", 369, 392), ("XXII", 393, 395)]


def part_of(article):
    """'243K' -> 'IX' (Panchayats), '243ZA' -> 'IXA' (Municipalities), '156' -> 'VI.2', 'SCH11' -> 'SCH11'."""
    m = re.match(r"(\d+)([A-Z]*)", str(article))
    if not m:
        return str(article)
    n, suffix = int(m.group(1)), m.group(2)
    if n == 243:
        if suffix.startswith("Z") and suffix[1:2] >= "H":
            return "IXB"
        if suffix >= "P" or suffix.startswith("Z"):
            return "IXA"
        return "IX"
    if n == 51 and suffix == "A":
        return "IVA"
    for name, lo, hi in _RANGES:
        if lo <= n <= hi:
            return name
    return str(article)


def family(chunk):
    """The topic a passage belongs to: a Part or chapter of the Constitution, a Schedule, or one NCERT chapter."""
    m = re.match(r"c(SCH\d+)", chunk["id"])
    if m:
        return m.group(1)
    m = re.match(r"c(\d+[A-Z]*)", chunk["id"])
    if m:
        return part_of(m.group(1))
    m = re.match(r"(n\d+)", chunk["id"])
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
