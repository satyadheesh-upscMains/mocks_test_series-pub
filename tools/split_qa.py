"""One synthetic answer spread across runners: the faster path, measured end to end on GitHub.

    python tools/split_qa.py prepare --case typed-polity --dir out     question breakdown + sources (1 runner)
    python tools/split_qa.py task    --case typed-polity --check accuracy --dir out    one check (1 runner each)
    python tools/split_qa.py finish  --case typed-polity --dir out     second opinion, report, the same gates as
                                                                        tools/e2e_qa.py (1 runner)

Synthetic fixtures only (the cases in tools/e2e_qa.py). Intermediate files travel as workflow artifacts, which is fine
for fixtures and never for real students: the live queue keeps them on the site. Logs carry stage, check, timing and
status only.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import e2e_qa  # noqa: E402
from mains import pipeline  # noqa: E402
from mains.model import Examiner, load_model  # noqa: E402


def slug(check_id):
    return check_id.replace(":", "-")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1))


class Calls:
    """What the queue site would have seen, so `e2e_qa.gates` judges this report exactly as the single-runner run."""

    def __init__(self, report):
        self.calls = [{"action": "claim"}, {"action": "complete", "result": report}, {"action": "deliver"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "task", "finish"])
    ap.add_argument("--case", required=True, choices=sorted(e2e_qa.CASES))
    ap.add_argument("--check")
    ap.add_argument("--dir", default="split-out")
    a = ap.parse_args()
    out, case = Path(a.dir), a.case
    if e2e_qa.CASES[case]["input"] != "typed":
        raise SystemExit("split QA covers typed answers; photos are read on one runner first")
    started = time.monotonic()
    ex = Examiner(load_model())
    tag = a.stage if a.stage != "task" else slug(a.check)
    try:
        if a.stage == "prepare":
            prep = pipeline.prepare(ex, e2e_qa.job_for(case))
            tasks = pipeline.task_ids(prep)
            write(out / f"prep-{case}.json", prep)
            write(out / f"tasks-{case}.json", tasks)
            print(json.dumps({"case": case, "tasks": len(tasks), "sources": len(prep.get("refs") or [])}), flush=True)
        elif a.stage == "task":
            prep = json.loads((out / f"prep-{case}.json").read_text())
            r = pipeline.run_task(ex, prep, a.check)
            write(out / f"task-{case}-{slug(a.check)}.json", r)
            print(f"{case} {a.check}: {r['status']} {r.get('verdict', r.get('reason', ''))}", flush=True)
        else:
            prep = json.loads((out / f"prep-{case}.json").read_text())
            tasks = json.loads((out / f"tasks-{case}.json").read_text())
            results, lost = [], []
            for cid in tasks:
                f = out / f"task-{case}-{slug(cid)}.json"
                if f.exists():
                    results.append(json.loads(f.read_text()))
                else:  # a runner that failed: the check shows as "Not checked", nothing is invented
                    lost.append(cid)
                    results.append({"id": cid, "kind": "part" if cid.startswith("part:") else "check",
                                    "status": "unavailable", "reason": "runner"})
            rep = pipeline.finish(ex, prep, results)
            g = e2e_qa.gates(case, Calls(rep), {"done": 1, "retake": 0, "failed": 0, "lost": 0})
            calls = []
            for f in sorted(out.glob(f"calls-{case}-*.json")):
                calls += json.loads(f.read_text())
            calls += ex.calls
            write(out / f"split-{case}.json", {
                "case": case, "gates": g, "lostTasks": lost, "report": rep, "modelCalls": calls,
                "modelSeconds": round(sum(c.get("seconds", 0) for c in calls), 1),
                # The extracts this answer was checked against, labels only (no text): which sentence each may check.
                "prepSources": [{"id": r["id"], "label": r["label"], "shared": r.get("shared", True),
                                 "claims": r.get("claims", [])} for r in prep.get("refs") or []],
                "claimResults": [{"id": r["id"], "status": r["status"], "verdict": r.get("verdict"),
                                  "unproven": bool(r.get("unproven")), "contradicted": bool(r.get("contradicted"))}
                                 for r in results if r["id"].startswith("facts:")]})
            for k, v in g.items():
                print(f"  {'PASS' if v else 'FAIL'} {k}", flush=True)
            if not all(g.values()):
                sys.exit(1)
    except Exception as e:
        if isinstance(e, SystemExit):
            raise
        write(out / f"error-{case}-{tag}.json",
              {"error": type(e).__name__, "code": getattr(e, "code", None), "trace": traceback.format_exc()})
        print(f"{case} {a.stage} {a.check or ''}: {type(e).__name__} {getattr(e, 'code', '') or ''}", flush=True)
        sys.exit(1)
    finally:
        if a.stage != "finish":
            write(out / f"calls-{case}-{tag}.json", ex.calls)
        print(f"{case} {a.stage} {a.check or ''}: {round(time.monotonic() - started)} s on this runner", flush=True)


if __name__ == "__main__":
    main()
