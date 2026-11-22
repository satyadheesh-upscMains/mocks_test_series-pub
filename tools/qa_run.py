"""Synthetic QA on the GitHub runner: the examiner pipeline, stage by stage, on fixture students only.

    python tools/qa_run.py prepare --case ID --dir OUT      question breakdown (1 runner per student)
    python tools/qa_run.py shard   --case ID --shard NAME --dir OUT   focused checks (5 runners per student)
    python tools/qa_run.py finish  --case ID --dir OUT      second opinion, report, QA gates

No production data, no worker URL. Logs carry only stage, check, timing and status; reports go to artifacts.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mains import pipeline, report  # noqa: E402
from mains.model import Examiner, load_model  # noqa: E402

STUDENTS = ROOT / "qa" / "students.json"


def student(case):
    return next(s for s in json.loads(STUDENTS.read_text()) if s["id"] == case)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1))


def gates(rep, expect):
    """Checks a human would make on this synthetic student. Reported, never used to change the report."""
    out = {}
    out["level"] = rep["level"] in expect["levels"]
    statuses = [p["status"] for p in rep["parts"]]
    if expect.get("noPartMissing"):
        out["no part missing"] = "missing" not in statuses
    if "partNotMet" in expect:
        out["part not credited as answered"] = statuses[expect["partNotMet"]] != "met"
    if "partMissing" in expect:
        out["part marked missing"] = statuses[expect["partMissing"]] == "missing"
    if expect.get("redSentences"):
        tone = {h["sourceId"]: h["tone"] for h in rep["highlights"]}
        out["problem sentences marked red"] = all(tone.get(s) == "red" for s in expect["redSentences"])
    out["three or fewer fixes"] = len(rep["fixes"]) <= 3
    out["every check ran"] = not rep["trust"]["unavailableChecks"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "shard", "finish"])
    ap.add_argument("--case", required=True)
    ap.add_argument("--shard")
    ap.add_argument("--dir", default="qa-out")
    a = ap.parse_args()
    out = Path(a.dir)
    s = student(a.case)
    started = time.monotonic()
    ex = Examiner(load_model())
    try:
        if a.stage == "prepare":
            prep = pipeline.prepare(ex, s)
            write(out / f"prep-{a.case}.json", prep)
            print(f"{a.case}: {len(prep['parts'])} parts ({prep['breakdown']['source']}), "
                  f"{prep['stats']['sentences']} sentences, {prep['stats']['words']} words", flush=True)
        elif a.stage == "shard":
            prep = json.loads((out / f"prep-{a.case}.json").read_text())
            results = pipeline.run_shard(ex, prep, a.shard)
            write(out / f"shard-{a.case}-{a.shard}.json", results)
            for r in results:
                print(f"{a.case} {r['id']}: {r['status']} {r.get('verdict', r.get('reason', ''))}", flush=True)
        else:
            prep = json.loads((out / f"prep-{a.case}.json").read_text())
            results = []
            for shard in pipeline.SHARD_ORDER:
                f = out / f"shard-{a.case}-{shard}.json"
                if f.exists():
                    results += json.loads(f.read_text())
                else:
                    print(f"{a.case}: shard {shard} missing", flush=True)
            rep = pipeline.finish(ex, prep, results)
            g = gates(rep, s["expect"])
            calls = []
            for f in sorted(out.glob(f"calls-{a.case}-*.json")):
                calls += json.loads(f.read_text())
            calls += ex.calls
            write(out / f"report-{a.case}.json", {
                "case": a.case, "what": s["expect"]["what"], "gates": g, "report": rep,
                "calls": calls, "modelSeconds": round(sum(c["seconds"] for c in calls), 1),
            })
            print(f"{a.case}: level={rep['level']} parts={[p['status'] for p in rep['parts']]} "
                  f"fixes={len(rep['fixes'])} second_opinions={rep['trust']['secondOpinions']} "
                  f"overturned={rep['trust']['overturned']}", flush=True)
            for name, ok in g.items():
                print(f"  {'PASS' if ok else 'FAIL'} {name}", flush=True)
    except Exception as e:
        write(out / f"error-{a.case}-{a.stage}{'-' + a.shard if a.shard else ''}.json",
              {"error": type(e).__name__, "code": getattr(e, "code", None), "trace": traceback.format_exc()})
        print(f"{a.case} {a.stage}: {type(e).__name__} {getattr(e, 'code', '') or ''}", flush=True)
        sys.exit(1)
    finally:
        if a.stage != "finish":
            tag = a.stage if a.stage == "prepare" else a.shard
            write(out / f"calls-{a.case}-{tag}.json", ex.calls)
        print(f"{a.case} {a.stage}{' ' + a.shard if a.shard else ''}: {round(time.monotonic() - started)} s total", flush=True)


if __name__ == "__main__":
    main()
