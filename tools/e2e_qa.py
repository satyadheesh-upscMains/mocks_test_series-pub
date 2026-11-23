"""End-to-end on the GitHub runner with real Gemma 4 12B: a synthetic submission goes through the queue worker.

    python tools/e2e_qa.py --case photo-en --dir e2e-out

The site is simulated in memory (no production URL, no database). The worker claims the job, reads the photo,
grades it and posts the report, exactly as it will against the real site. Synthetic students only.
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import worker  # noqa: E402
import photo_qa  # noqa: E402

STUDENTS = {s["id"]: s for s in json.loads((ROOT / "qa" / "students.json").read_text())}
BASE = STUDENTS["wetlands-overclaim"]
CASES = {
    "typed-en": dict(input="typed"),
    "photo-en": dict(input="photos", page="clear-en"),
    "photo-smudged": dict(input="photos", page="smudged-en"),
}


class Site:
    def __init__(self, job):
        self.jobs, self.calls = [job], []

    def __call__(self, p):
        a = p["action"]
        self.calls.append({k: v for k, v in p.items() if k not in ("lease",)})
        if a == "claim":
            return {"job": self.jobs.pop(0) if self.jobs else None}
        if a == "transcribe":
            return {"ok": True, "blocked": p["transcript"]["blocked"]}
        if a == "deliver":
            return {"processed": 0}
        return {"ok": True}


def job_for(case):
    c = CASES[case]
    job = {"id": "e2e-" + case, "lease": "lease", "lang": "en", "question": BASE["question"],
           "references": BASE["references"], "marks": 10, "input": c["input"]}
    if c["input"] == "typed":
        job["answer"] = BASE["answer"]
    else:
        job["pages"] = [base64.b64encode(photo_qa.draw(c["page"])).decode()]
    return job


def gates(case, site, counts):
    actions = [c["action"] for c in site.calls]
    tr = next((c["transcript"] for c in site.calls if c["action"] == "transcribe"), None)
    rep = next((c["result"] for c in site.calls if c["action"] == "complete"), None)
    g = {"claimed and delivered": actions[0] == "claim" and actions[-1] == "deliver"}
    if case == "photo-smudged":
        g["photo blocked for a retake"] = bool(tr and tr["blocked"])
        g["not graded"] = rep is None
        g["retake message given"] = bool(tr and tr["retakeMessage"])
        return g
    g["report completed"] = counts["done"] == 1 and rep is not None
    if rep is None:
        return g
    g["report version v5"] = rep["version"] == "mains-report-v5"
    g["three or fewer fixes"] = len(rep["fixes"]) <= 3
    g["every check judged"] = not rep["trust"]["unavailableChecks"]
    des = [h for h in rep["highlights"] if "desilt" in h["text"].lower()]
    g["routine desilting flagged, not praised"] = bool(des) and all(h["tone"] != "green" for h in des) and not any(
        "desilt" in s["quote"].lower() for s in rep["strengths"])
    red = {h["text"] for h in rep["highlights"] if h["tone"] == "red"}
    g["80% claim marked red"] = any("80%" in t for t in red)
    g["concreting marked red"] = any("concrete" in t for t in red)
    if case == "photo-en":
        g["photo read without a retake"] = bool(tr) and not tr["blocked"]
        g["every highlight on the page"] = all(h.get("lines") for h in rep["highlights"])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--dir", default="e2e-out")
    a = ap.parse_args()
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    job = job_for(a.case)
    site = Site(job)
    started = time.monotonic()
    counts = worker.process(site)
    g = gates(a.case, site, counts)
    record = {"case": a.case, "counts": counts, "gates": g, "seconds": round(time.monotonic() - started, 1),
              "calls": site.calls}
    (out / f"e2e-{a.case}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    print(f"{a.case}: {counts} in {record['seconds']} s", flush=True)
    for k, v in g.items():
        print(f"  {'PASS' if v else 'FAIL'} {k}", flush=True)
    sys.exit(0 if all(g.values()) else 1)


if __name__ == "__main__":
    main()
