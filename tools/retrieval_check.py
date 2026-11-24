"""Source check before any Gemma call: do the planted errors reach the Article that settles them?

    python tools/retrieval_check.py            # uses refs/index.json.gz (run tools/build_refs.py first)

For each answer in the Polity test set, every planted error's sentence must have, among its claim extracts, one of
the Articles that settle it; the off-syllabus wetlands answer must get no source at all. Exit code 1 if not, so the
two-hour Gemma run never starts on sources that cannot work (v10 found its source problems only after the run).
Prints labels only, never passage text. With the embedding model present it also prints, for each planted sentence,
what the embedding model would pick and its cosine score: measurement only, to tune it before it is switched on.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mains import refs  # noqa: E402
from mains.text import sentences  # noqa: E402
import e2e_qa  # noqa: E402

WETLANDS = (e2e_qa.BASE["question"], e2e_qa.BASE["answer"])
HINDI_PROBES = ["राज्यपाल की नियुक्ति कैसे होती है? भारत की संघीय व्यवस्था में राज्यपाल की स्थिति का विश्लेषण कीजिए।",
                "धन विधेयक के पारित होने में राज्यसभा की भूमिका की चर्चा कीजिए।"]


def label(a):
    return f"Constitution of India, {'Schedule ' + a[3:] if a.startswith('SCH') else 'Article ' + a}"


def short(value):
    return value.replace("Constitution of India, ", "").replace("Laxmikanth, Indian Polity, ", "Laxmikanth ")


def main():
    idx = refs.load()
    if idx is None:
        sys.exit("refs/index.json.gz missing: run tools/build_refs.py first")
    print(f"embeddings: {'on (measured only)' if idx.embedder else 'off'}")
    ok, report = True, {}
    for case, v in e2e_qa.POLITY_SET.items():
        sents = sentences(v["answer"])
        got = idx.pack(v["question"], sents)
        rows = []
        for phrase, arts in v["planted"]:
            sid = next(s["id"] for s in sents if phrase in s["text"])
            mine = [r["label"] for r in got if sid in r.get("claims", [])]
            hit = any(m in {label(a) for a in arts} for m in mine)
            ok &= hit
            rows.append({"phrase": phrase, "sentence": sid, "want": arts, "reached": hit,
                         "claimSources": [short(m) for m in mine],
                         "embedding": idx.shadow(next(s["text"] for s in sents if s["id"] == sid)),
                         "embeddingCasesAndBook": idx.extra.shadow(next(s["text"] for s in sents if s["id"] == sid))
                         if idx.extra else []})
            print(f"{'PASS' if hit else 'FAIL'} {case:17} {sid} {phrase!r} -> {', '.join(short(m) for m in mine)}")
        report[case] = {"shared": [short(r["label"]) for r in got if r["shared"]], "planted": rows}
    q, a = WETLANDS
    wet = idx.pack(q, sentences(a))
    ok &= not wet
    print(f"{'PASS' if not wet else 'FAIL'} wetlands gets no source ({len(wet)})")
    report["wetlands"] = {"shared": [short(r["label"]) for r in wet], "embedding": idx.shadow(q)}
    report["hindiProbes"] = [{"question": h, "keyword": [short(r["label"]) for r in idx.pack(h, "")],
                              "embedding": idx.shadow(h)} for h in HINDI_PROBES]
    for h in report["hindiProbes"]:
        print(f"hindi probe: keyword {len(h['keyword'])} sources; embedding {h['embedding']}")
    Path("retrieval-out").mkdir(exist_ok=True)
    Path("retrieval-out/retrieval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
