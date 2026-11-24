"""Stage: second opinion. Before a serious criticism reaches the student, a separate Gemma pass re-checks it.

Re-checked: every "wrong" or "unsupported" finding, every major finding, and every question part marked missing.
An overturned finding is dropped, a softened one loses one level of priority, and a part that the second pass finds
covered becomes "partial". If the second pass itself fails, the finding is kept one level softer, never stronger.
"""
import json

from .rubric import DIM
from .text import Rejected, norm

LIMIT = 8
LOWER = {"major": "medium", "medium": "minor", "minor": "minor"}
RANK = {"major": 0, "medium": 1, "minor": 2}

TASK = """TASK [verify]
A first examiner wrote the criticism below about this answer. Check it yourself, independently, before it reaches the student.
CRITICISM
{criticism}
Fill the JSON fields in order:
- coveredBy: IDs of the student's sentences that already do what the criticism says is missing or wrong. Empty if none.
- reason: one sentence in English explaining your decision.
- decision: upheld (correct and as serious as stated), softened (partly correct, or less serious than stated), overturned (wrong: the answer already does this, or the cited sentence is actually correct).
A sentence that says the same thing as the source in other words, or uses a common name for the same body, House or level ("block" for the intermediate level, "Lok Sabha" for the House of the People), is correct."""


def schema(sids):
    return {"type": "object", "additionalProperties": False, "required": ["coveredBy", "reason", "decision"],
            "properties": {"coveredBy": {"type": "array", "maxItems": 3, "items": {"type": "string", "enum": list(sids)}},
                           "reason": {"type": "string"},
                           "decision": {"type": "string", "enum": ["upheld", "softened", "overturned"]}}}


def _validate(raw):
    if raw.get("decision") not in ("upheld", "softened", "overturned"):
        raise Rejected("verify:decision")
    if not 5 <= len(norm(raw.get("reason"))) <= 400:
        raise Rejected("verify:reason")
    return {"decision": raw["decision"], "coveredBy": list(dict.fromkeys(raw.get("coveredBy") or []))}


def targets(prep, results):
    """(rank, result index, concern index or None) for each finding to re-check, most serious first.

    Every contradiction found by a claim check is re-checked first, outside the budget: in v11 the Panchayat report
    spent its eight re-checks on other findings and its one false contradiction ("block" for the intermediate level)
    reached the student, while in the four reports where the claim contradictions were re-checked, all four false
    ones were overturned and every planted error stood. The rest follow within LIMIT; ties follow the fixed check
    order (question parts, then the twelve checks), never the order in which runner shards delivered their results."""
    canon = [f"part:{p['id']}" for p in prep["parts"]] + [d for d in DIM]
    pos = {cid: i for i, cid in enumerate(canon)}
    first, out = [], []
    for ri, r in enumerate(results):
        if r["status"] != "ok":
            continue
        at = pos.get(r["id"], len(canon))
        if r["kind"] == "part" and r["verdict"] == "missing":
            out.append((-1, at, -1, ri, None))
        for ci, c in enumerate(r["concerns"]):
            if c.get("fromClaim") and c["kind"] == "wrong":
                first.append((at, ci, ri, ci))
                continue
            if r["kind"] == "part" and r["verdict"] == "missing" and c["sourceId"] is None:
                continue  # covered by the part-level re-check above
            if c["kind"] in ("wrong", "unsupported") or c["priority"] == "major":
                out.append((RANK[c["priority"]], at, ci, ri, ci))
    return ([(-2, t[2], t[3]) for t in sorted(first, key=lambda t: t[:2])]
            + [(t[0], t[3], t[4]) for t in sorted(out, key=lambda t: t[:3])[:LIMIT]])


def _about(prep, r):
    if r["kind"] == "part":
        part = next(p for p in prep["parts"] if f"part:{p['id']}" == r["id"])
        return f"the question part \u00ab{part['quote']}\u00bb"
    return f"the check '{DIM[r['id']]['en']}'"


def _source(prep, ref_id):
    """The extract a finding relies on, given to the second opinion in full: claim-check extracts are not in
    MATERIALS."""
    r = next((x for x in prep.get("refs") or [] if x["id"] == ref_id), None)
    return {"source": f"{r['label']}: {r.get('full') or r['text']}"} if r else {}


def run(examiner, prep, system, results):
    sids = [s["id"] for s in prep["sentences"]]
    for _, ri, ci in targets(prep, results):
        r = results[ri]
        if ci is None:
            criticism = {"about": _about(prep, r), "kind": "missing", "sentence": None,
                         "criticism": "No sentence of the answer addresses this part."}
        else:
            c = r["concerns"][ci]
            criticism = {"about": _about(prep, r), "kind": c["kind"], "priority": c["priority"],
                         "sentence": c["sourceId"], "criticism": c["reason"],
                         **_source(prep, c.get("ref"))}
            if c.get("fromClaim") and c["sourceId"]:
                # The sentence itself and the two quotes, so the second opinion judges this sentence's words: in v12 it
                # upheld a true reason that belonged to the sentence before.
                criticism.update({
                    "sentenceText": next(s["text"] for s in prep["sentences"] if s["id"] == c["sourceId"]),
                    "quotedAsWrong": (c.get("evidence") or {}).get("wrongWords"),
                    "quotedFromSource": (c.get("evidence") or {}).get("sourceWords"),
                    "question": "Do the words of sentenceText itself contradict the source? Overturn if they say the "
                                "same thing in other words, or if the criticism is about a different sentence."})
        try:
            out = examiner.ask(stage="verify", check=r["id"], system=system,
                               task=TASK.format(criticism=json.dumps(criticism, ensure_ascii=False)),
                               schema=schema(sids), max_tokens=260, validate=_validate)
        except Rejected:
            out = {"decision": "failed", "coveredBy": []}
        apply(r, ci, out)
    settle(results)
    return results


def settle(results):
    """Remove overturned findings and make each verdict agree with what is left."""
    for r in results:
        if r["status"] != "ok":
            continue
        r["concerns"] = [c for c in r["concerns"] if not c.get("dropped")]
        if not r.get("revised") or r["concerns"]:
            continue
        if r["kind"] == "check" and r["verdict"] != "good":
            r["verdict"] = "good" if r["credit"] else "not_applicable"
            # The first examiner's note described a problem that no longer stands.
            r["note"] = r["credit"][0]["reason"] if r["credit"] else None
        elif r["kind"] == "part" and r["credit"] and r["verdict"] != "met":
            r["verdict"] = "met"
            r["note"] = r["credit"][0]["reason"]


def apply(r, ci, out):
    decision = out["decision"]
    r.setdefault("opinions", []).append(decision)
    if ci is None:  # a question part marked missing
        r["verification"] = decision
        if decision == "overturned" and out["coveredBy"]:
            r["verdict"] = "partial"
            r["revised"] = True
            for c in r["concerns"]:
                if c["sourceId"] is None and c["priority"] == "major":
                    c["priority"] = "medium"
        return
    c = r["concerns"][ci]
    c["verification"] = decision
    if decision == "overturned":
        c["dropped"] = True  # removed in settle(), so the other indices stay valid meanwhile
        r["revised"] = True
    elif decision in ("softened", "failed"):
        c["priority"] = LOWER[c["priority"]]
        if decision == "softened" and c["kind"] == "wrong":
            c["kind"] = "weak"
