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
HINDI_PROBES = [  # (Hindi claim or question, Articles that settle or answer it): measured, not gated
    ("राज्यपाल का चुनाव राज्य की विधानसभा पाँच वर्ष के लिए करती है।", ["155", "156"]),
    ("धन विधेयक संसद के किसी भी सदन में पेश किया जा सकता है।", ["109", "110"]),
    ("पंचायतों के चुनाव भारत का निर्वाचन आयोग कराता है।", ["243K"]),
    ("राष्ट्रपति शासन अनिश्चित काल तक जारी रह सकता है।", ["356"]),
    ("नीति निदेशक तत्वों को मौलिक अधिकारों की तरह अदालत में लागू कराया जा सकता है।", ["37"]),
    ("मुख्य निर्वाचन आयुक्त को प्रधानमंत्री की सलाह पर राष्ट्रपति हटा सकते हैं।", ["324"]),
    ("राज्यपाल की नियुक्ति कैसे होती है? भारत की संघीय व्यवस्था में राज्यपाल की स्थिति का विश्लेषण कीजिए।", ["153", "155", "156"]),
    ("धन विधेयक के पारित होने में राज्यसभा की भूमिका की चर्चा कीजिए।", ["109", "110"]),
]


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001 - measurement only
        return [["error", f"{type(e).__name__}"]]


def in_top(shadow, arts):
    want = {label(a) for a in arts}
    return any(lab in want for lab, _ in shadow)


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
    for case, v in {**e2e_qa.POLITY_SET, **e2e_qa.BAND_SET}.items():
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
                         "embedding": safe(idx.shadow, next(s["text"] for s in sents if s["id"] == sid), k=10),
                         "embeddingCasesAndBook": safe(idx.extra.shadow, next(s["text"] for s in sents
                                                                               if s["id"] == sid)) if idx.extra else []})
            print(f"{'PASS' if hit else 'FAIL'} {case:17} {sid} {phrase!r} -> {', '.join(short(m) for m in mine)}")
        report[case] = {"shared": [short(r["label"]) for r in got if r["shared"]], "planted": rows}
    q, a = WETLANDS
    wet = idx.pack(q, sentences(a))
    ok &= not wet
    print(f"{'PASS' if not wet else 'FAIL'} wetlands gets no source ({len(wet)})")
    report["wetlands"] = {"shared": [short(r["label"]) for r in wet], "embedding": safe(idx.shadow, q)}
    report["hindiProbes"] = [{"question": h, "want": arts, "keyword": [short(r["label"]) for r in idx.pack(h, "")],
                              "embedding": safe(idx.shadow, h, k=10)} for h, arts in HINDI_PROBES]
    try:
        summarise(idx, report)
    except Exception as e:  # noqa: BLE001 - a measurement must never fail the source gate
        print(f"embedding measurement failed ({type(e).__name__}: {e})")
    for h in report["hindiProbes"]:
        print(f"hindi probe: keyword {len(h['keyword'])} sources; embedding top {h['embedding'][:1]}")
    Path("retrieval-out").mkdir(exist_ok=True)
    Path("retrieval-out/retrieval.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    sys.exit(0 if ok else 1)


def summarise(idx, report):
    if idx.embedder:
        eng = [row for case in report.values() if isinstance(case, dict) for row in case.get("planted", [])]
        e_hits = sum(in_top(r["embedding"], r["want"]) for r in eng)
        h_hits = sum(in_top(h["embedding"], h["want"]) for h in report["hindiProbes"])
        report["embeddingSummary"] = {"englishTop10": e_hits, "english": len(eng), "hindiTop10": h_hits,
                                      "hindi": len(report["hindiProbes"])}
        print(f"embeddings (measured only): settling Article in top 10 for {e_hits}/{len(eng)} English planted "
              f"claims and {h_hits}/{len(report['hindiProbes'])} Hindi claims and questions")


if __name__ == "__main__":
    main()
