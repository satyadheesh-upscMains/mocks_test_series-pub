"""Assemble the full report deterministically from twelve validated checks, without inventing findings."""
import json
from pathlib import Path

from deep_review import DIMENSIONS

CASES = ("wetlands-one-sided", "wetlands-overclaim", "heat-strong-hi")
ORDER = {s: i for i, s in enumerate(("major", "medium", "minor", "none"))}


def assemble(case, directory):
    rows = []
    groups = {}
    for group, dimensions in DIMENSIONS.items():
        group_rows = json.loads((directory / f"review-{case}-{group}.json").read_text())
        assert [r["dimension"] for r in group_rows] == [d[0] for d in dimensions]
        rows.extend(group_rows)
        groups[group] = group_rows
    assert len({r["dimension"] for r in rows}) == 12
    prioritized = sorted(
        (r for r in rows if r["priority"] != "none" and r["verdict"] not in {"strong", "not_applicable"}),
        key=lambda r: ORDER[r["priority"]],
    )
    next_steps = []
    for row in prioritized:
        if row["nextEdit"].casefold() not in {r["nextEdit"].casefold() for r in next_steps}:
            next_steps.append({k: row[k] for k in ("dimension", "label", "priority", "nextEdit")})
        if len(next_steps) == 3:
            break
    return {
        "case": case, "checks": len(rows), "revisionPlan": next_steps, "dimensions": groups,
        "totalInferenceSeconds": round(sum(r["seconds"] for r in rows), 1),
        "slowestGroupSeconds": round(max(sum(r["seconds"] for r in items) for items in groups.values()), 1),
        "retriedChecks": sum(r["attempts"] > 1 for r in rows),
    }


if __name__ == "__main__":
    Path("deep-reports").mkdir(exist_ok=True)
    for case in CASES:
        result = assemble(case, Path("checks"))
        Path(f"deep-reports/{case}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(case, result["checks"], result["slowestGroupSeconds"], result["retriedChecks"], flush=True)
