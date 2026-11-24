"""Reference extracts for the examiner: Constitution, NCERT and original case summaries.

`tools/build_refs.py` builds refs/index.json.gz once per runner: the Constitution comes from the committed JSON (public
domain); NCERT chapters are downloaded from ncert.nic.in at build time, so no textbook text is ever committed. Search
is plain BM25 in Python, no model. Passages go to Gemma inside MATERIALS only: reports carry a passage's label (for
example "Constitution of India, Article 243K"), never its text, and logs print counts only.
"""
import collections
import hashlib
import gzip
import json
import math
import os
import re
import hashlib
from pathlib import Path

INDEX = Path(os.environ.get("MAINS_REFS", Path(__file__).resolve().parents[1] / "refs" / "index.json.gz"))
MAX_PASSAGES = 6     # extracts shown to every check
MAX_REFS = 16        # with the extracts that check single claims
CLAIM_SOURCES = 5    # extracts per claim
MAX_CASES = 2        # landmark cases added to a Constitution question, for credit and additions only
MAX_BOOK = 2         # passages of the licensed book (private core only), for credit and additions only
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
EMBED_REPO = "keisuke-miyako/multilingual-e5-small-gguf-q8_0"   # intfloat/multilingual-e5-small, GGUF Q8_0
EMBED_FILE = "multilingual-e5-small-Q8_0.gguf"
EMBED_SHA256 = "0d5a5a0b0ad84faad6357a6145e769b0661f0efbf53acf74598afc34dab454f4"   # pinned: the file is checked
EMBED_DIR = Path(os.environ.get("MAINS_EMBED_DIR", Path(__file__).resolve().parents[1] / "models"))


class Embedder:
    """multilingual-e5-small through llama-cpp-python (already installed for Gemma): no torch, no API, CPU only."""

    def __init__(self, path):
        import llama_cpp
        self.llm = llama_cpp.Llama(model_path=str(path), embedding=True, n_ctx=512, n_batch=512,
                                   n_threads=max(2, os.cpu_count() or 2), pooling_type=1, verbose=False)

    def embed(self, texts):
        return [self.llm.embed(t[:1500], normalize=True, truncate=True) for t in texts]


def embedder():
    """The pinned e5 model when it is on this runner (models/, downloaded by the workflow) and its checksum matches;
    otherwise None, and search is keyword-only. Never raises: a missing model must not stop grading."""
    if os.environ.get("MAINS_EMBED", "1") == "0":
        return None
    path = EMBED_DIR / EMBED_FILE
    try:
        if not path.exists():
            if os.environ.get("MAINS_EMBED_DOWNLOAD") != "1":
                return None
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo_id=EMBED_REPO, filename=EMBED_FILE, local_dir=str(EMBED_DIR)))
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != EMBED_SHA256:
            print("refs embedding model checksum mismatch; using keyword search only", flush=True)
            return None
        return Embedder(path)
    except Exception as e:  # noqa: BLE001 - fail open to keyword search
        print(f"refs embedding unavailable ({type(e).__name__}); using keyword search only", flush=True)
        return None


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


def case_chunks(entries):
    """Our summaries are useful for credit/gaps, never contradiction evidence."""
    return [{"id": "case:" + e["id"], "label": f"Case: {e['name']} ({e['year']})",
             "title": e["name"], "text": e["holding"], "case": True,
             "articles": e.get("articles", [])} for e in entries]


def laxmikanth_chunks(entries):
    """The licensed book, private core only: labels name chapter and section; the text never leaves the runner."""
    out = []
    for n, e in enumerate(entries, 1):
        sec = e.get("section") or ""
        label = f"Laxmikanth, Indian Polity, Ch. {e['chapter']} ({e['title']})" + (f", {sec}" if sec else "")
        out.append({"id": f"l{n}", "label": label, "title": f"{e['title']}. {sec}", "text": e["text"],
                    "private": True})
    return out


# ------------------------------------------------------------------ search

class Index:
    def __init__(self, chunks, min_score=MIN_SCORE, min_coverage=MIN_COVERAGE, embedder=None, cache_dir=None,
                 extras=True):
        chunks = list(chunks)
        core = [c for c in chunks if not c.get("case") and not c.get("private")]
        extra = [c for c in chunks if c.get("case") or c.get("private")]
        # Keyword weights come from the Constitution and NCERT only. Adding 2,332 passages of the licensed book to the
        # same statistics made "panchayat" common, so no Article of Part IX reached the threshold for the 73rd
        # Amendment question (v10 retrieval check). Cases and the book are searched in their own index, after the core
        # search has decided the question's topic. Indices: 0..len(core)-1 are core, then the extras in order.
        if not extras:
            core, extra = chunks, []
        self.chunks = core + extra
        self.cache_dir = cache_dir
        self.extra = Index(extra, min_score, min_coverage, embedder=None, extras=False) if extra else None
        self.offset = len(core)
        self.min_score, self.min_coverage = min_score, min_coverage
        self.embedder = embedder
        self._vectors = None
        self.docs = [collections.Counter(terms(c["title"] + " " + c["text"])) for c in core]
        # An Article's own title says what it is about ("Appointment of Governor"). Titles get their own index, with
        # their own rarity weights: "Governor" is common in the text of the Constitution but rare in its titles, so a
        # question on the Governor finds Articles 153-156 before the many Articles that only mention a Governor.
        self.heads = [collections.Counter(terms(c["title"])) for c in core]
        hdf = collections.Counter(t for d in self.heads for t in d)
        self.hidf = {t: math.log(1 + (len(self.heads) - k + 0.5) / (k + 0.5)) for t, k in hdf.items()}
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df = collections.Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - k + 0.5) / (k + 0.5)) for t, k in df.items()}
        self.by_article = {}
        for i, c in enumerate(core):
            m = re.match(r"c(\w+?)(?:\.\d+)?$", c["id"])
            if m:
                self.by_article.setdefault(m.group(1).upper(), i)

    def _embedding_ranks(self, query):
        """Optional e5 second ranker. Fail-open: graders work without the model, and unit tests inject a tiny
        deterministic embedder. Passage vectors are cached as float16 next to the index, keyed by model and text."""
        if not self.embedder:
            return []
        try:
            import numpy as np
            if self._vectors is None:
                self._vectors = self._passage_vectors(np)
            q = np.asarray(self.embedder.embed(["query: " + query])[0], dtype="float32")
            den = np.linalg.norm(self._vectors, axis=1) * max(float(np.linalg.norm(q)), 1e-9)
            return list(np.argsort(-(self._vectors @ q / np.maximum(den, 1e-9))))
        except Exception as e:  # noqa: BLE001
            print(f"refs embedding unavailable ({type(e).__name__}); using keyword search only", flush=True)
            self.embedder = None
            return []

    def _passage_vectors(self, np):
        texts = ["passage: " + c["title"] + " " + c["text"] for c in self.chunks[:len(self.docs)]]
        key = hashlib.sha256((EMBED_SHA256 + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
        cache = Path(self.cache_dir) / f"vectors-{key}.npy" if self.cache_dir else None
        if cache and cache.exists():
            vec = np.load(cache)
            if vec.shape[0] == len(texts):
                return vec.astype("float32")
        vec = np.asarray(self.embedder.embed(texts), dtype="float32")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, vec.astype("float16"))
        return vec

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
        # Reciprocal-rank fusion preserves BM25's exact Article-number advantage
        # while allowing paraphrases to surface a relevant passage.
        eranks = self._embedding_ranks(query)
        if eranks and hits:
            bm = {i: n for n, (_, _, i) in enumerate(hits, 1)}
            em = {i: n for n, i in enumerate(eranks, 1)}
            hits.sort(key=lambda x: (1 / (60 + bm[x[2]]) + 0.7 / (60 + em.get(x[2], 10**6))), reverse=True)
        # Textbook exercise pages are mostly questions to the reader: they carry no facts to check against.
        return [(s, cov, i) for s, cov, i in hits if self.chunks[i]["text"].count("?") < 3]

    def article(self, number):
        i = self.by_article.get(str(number).upper())
        return None if i is None else self.chunks[i]

    def _core(self, i):
        """Constitution and NCERT text: the only passages that can check a claim or fill the main extract slots."""
        c = self.chunks[i]
        return not c.get("case") and not c.get("private")

    def pack(self, question, answer=""):
        """Extracts for one answer. `answer` is its text or its numbered sentences ([{"id": "s1", "text": ...}]).

        Returns up to MAX_PASSAGES shared extracts (shown to every check), then extracts used only to check single
        claims ("shared": False). Each extract lists the answer sentences it may check ("claims"). On a question the
        Constitution speaks to, up to MAX_CASES landmark cases and MAX_BOOK passages of the licensed book are added
        as shared extracts for credit and additions only: they never check a claim."""
        sents = answer if isinstance(answer, list) else [
            {"id": f"s{n}", "text": t} for n, t in enumerate(re.split(r"(?<=[.!?\u0964])\s+", answer or ""), 1) if t]
        order, shared, claims = [], set(), {}

        def take(i, share, extra=False):
            # One extract per Article or chapter label: a second slice of the same Article adds little. Returns the
            # extract that now stands for i (itself, or the one with the same label), or None when there is no room.
            if i is None:
                return None
            same = next((j for j in order if self.chunks[j]["label"] == self.chunks[i]["label"]), None)
            if same is None:
                if not extra and (len(order) >= MAX_REFS or (share and len(shared) >= MAX_PASSAGES)):
                    return None
                order.append(i)
                same = i
            if extra or (share and len(shared) < MAX_PASSAGES):
                shared.add(same)
            return same

        def claim(sid, j):
            # Only Constitution/NCERT can be claim evidence. Cases and the licensed book enrich credit and additions
            # but can never mark a sentence wrong.
            if (j is not None and self._core(j)
                    and j not in claims.setdefault(sid, []) and len(claims[sid]) < CLAIM_SOURCES):
                claims[sid].append(j)

        # Articles named in the question or the answer: the text of the Article itself, shown to every check.
        named = []
        for m in ARTICLE.finditer(question + " " + " ".join(s["text"] for s in sents)):
            j = take(self.by_article.get(m.group(1).upper()), True)
            if j is not None and j not in named:
                named.append(j)
        hits = self.search(question)
        on_topic = [i for sc, cov, i in hits if sc >= self.min_score and cov >= self.min_coverage and self._core(i)]
        for i in on_topic[:FROM_QUESTION_FIRST]:
            take(i, True)
        # Each claim gets the extracts that can check it, because the answer's claims are where its errors are:
        # the Articles it names, the Articles the question or answer names, and its best matches within the
        # question's topic (the Parts or chapters the question's own extracts come from), by its own words and by its
        # words with the question's. "Conducted by the Election Commission of India" in an answer on panchayats then
        # meets Article 243K (State Election Commission), not Article 324 or Schedule 6, which runs 45223096235 and
        # 45238473782 used instead. Tuned on eleven planted errors in eight Polity answers: all eleven reach a check.
        families = {family(self.chunks[i]) for i in on_topic}
        if on_topic:
            # The answer's own claims vote for the Constitution's Parts: each sentence's best Article, and a Part named
            # by at least two sentences joins the question's topics. "The 73rd Amendment ... rural local
            # self-government" matches NCERT's Local Governments chapter (and the word "amend" Article 368's title),
            # while Part IX's titles say "Gram Sabha" and "Constitution of Panchayats": the v10 retrieval check gave
            # that answer no Part IX Article, so none of its claims could be checked. Off-syllabus questions have no
            # on-topic passage and never get here.
            votes = collections.Counter()
            for s in sents:
                # Each sentence on its own words, Constitution passages only (textbook passages outrank Articles on
                # every claim, as the v10 diagnostic showed), the Parts of its five best Articles, one vote each.
                arts = [i for sc, _, i in self.search(s["text"], titles=True)
                        if sc >= self.min_score * CLAIM_SCORE and self.chunks[i]["id"].startswith("c")][:5]
                votes.update({family(self.chunks[i]) for i in arts})
            self.last_votes = dict(votes)
            whole = question + " " + " ".join(x["text"] for x in sents)
            need = max(2, math.ceil(0.6 * len(sents)))   # most of the answer points at this Part
            for fam, n in votes.most_common(2):
                if n >= need and fam not in families:
                    families.add(fam)
                    best = next((i for sc, cov, i in self.search(whole) if family(self.chunks[i]) == fam), None)
                    if best is not None:
                        take(best, True)
                        on_topic.append(best)
        ranked_families = list(dict.fromkeys(family(self.chunks[i]) for i in named + on_topic))
        named_articles = {m.group(1).upper() for m in ARTICLE.finditer(question + " " + " ".join(s["text"] for s in sents))}
        for s in sents:
            for m in ARTICLE.finditer(s["text"]):
                claim(s["id"], take(self.by_article.get(m.group(1).upper()), True))
            for j in named[:2]:
                claim(s["id"], j)
            if not families:
                continue
            for query, titles in ((s["text"], False), (f"{s['text']} {question}", True)):
                got = 0
                # Thresholds on the keyword score first, then the fused order (keyword + embedding ranks): a
                # paraphrase can lift a passage that keyword search ranked lower, never one it did not find.
                for sc, _, i in self.search(query, titles=titles):
                    if got >= 2:
                        break
                    if sc < self.min_score * CLAIM_SCORE or not self._core(i):
                        continue
                    if family(self.chunks[i]) in families:
                        j = take(i, False)
                        if j is not None and j not in claims.get(s["id"], []):
                            claim(s["id"], j)
                            got += 1
        for i in on_topic[FROM_QUESTION_FIRST:]:
            take(i, True)
        if families and self.extra:
            self._extras(question, sents, ranked_families, named_articles, take)
        return self._out(order, shared, claims)

    def _extras(self, question, sents, families, named, take):
        """Landmark cases and the licensed book's best sections, on a question the Constitution speaks to.

        Cases: those on an Article the question or answer names come first, then those in the question's own Parts of
        the Constitution, in the order the core search ranked those Parts; ties go to the better keyword match. (The
        v10 check gave the Article 21 question the Judges cases, because "Supreme Court" in the question put Part V
        chapter 4 among its topics.) Book: the best section for the question, and the best for question and answer."""
        text = question + " " + " ".join(s["text"] for s in sents)
        score = {j: sc for sc, _, j in self.extra.search(text)}
        rank = {f: n for n, f in enumerate(families)}
        picks = []
        for j, c in enumerate(self.extra.chunks):
            if not c.get("case"):
                continue
            arts = {str(a).upper() for a in c.get("articles", [])}
            near = [rank[part_of(a)] for a in arts if part_of(a) in rank]
            if not (arts & named or near):
                continue
            picks.append((0 if arts & named else 1, -len(arts & named), min(near) if near else 99,
                          -score.get(j, 0.0), j))
        for *_, j in sorted(picks)[:MAX_CASES]:
            take(self.offset + j, True, extra=True)
        got, seen = 0, set()
        for query in (question, text):
            for sc, cov, j in self.extra.search(query):
                c = self.extra.chunks[j]
                if not c.get("private") or j in seen:
                    continue
                if sc >= self.min_score and cov >= self.min_coverage and take(self.offset + j, True, extra=True) is not None:
                    seen.add(j)
                    got += 1
                break
            if got >= MAX_BOOK:
                break

    def _out(self, order, shared, claims):
        ranked = [i for i in order if i in shared] + [i for i in order if i not in shared]
        out = []
        for n, i in enumerate(ranked, 1):
            c = self.chunks[i]
            words = c["text"].split()
            r = {"id": f"r{n}", "label": c["label"],
                 "text": " ".join(words[:PASSAGE_WORDS] if i in shared else words), "shared": i in shared}
            if i in shared and len(words) > PASSAGE_WORDS and not c.get("private"):
                r["full"] = c["text"]  # claim checks and corrections read the whole Article (time limits sit late)
            if c.get("case"):
                r["case"] = True
            if c.get("private"):
                r["private"] = True
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
            _LOADED[path] = Index(json.loads(gzip.decompress(path.read_bytes())), embedder=embedder(),
                                  cache_dir=path.parent / "cache")
        except (OSError, ValueError, EOFError):
            _LOADED[path] = None
    return _LOADED[path]


def pack(question, answer="", index=None):
    idx = index if index is not None else load()
    return idx.pack(question, answer) if idx else []
