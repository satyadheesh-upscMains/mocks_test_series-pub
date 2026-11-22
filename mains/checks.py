"""Stage: the focused checks. One Gemma call per question part, plus the twelve dimension checks.

Every finding points at a sentence ID that Gemma can only choose from the student's own sentences (the JSON schema
lists them), so a quote shown to the student is always the student's exact words. Text fields go through
`text.clean`, and the verdict must agree with the evidence (for example "met" needs credit, "missing" allows none).
"""
from . import prompts
from .rubric import DIM
from .text import Rejected, SENTENCE_END, clean, similar

VERDICTS = ["good", "partial", "missing", "unsupported", "wrong", "not_applicable"]
PART_VERDICTS = ["met", "partial", "missing"]
KINDS = ["wrong", "unsupported", "weak", "missing"]
PRIORITY = ["major", "medium", "minor"]
NO_REWRITE = ("evidence", "accuracy")

FIELDS = """Fill the JSON fields in this order:
- credit: up to 2 of the student's sentences that earn credit for THIS check, each with a specific reason. Empty if none does.
- concerns: up to 2 of the most important problems for THIS check, most important first. sourceId is the sentence with the problem, or "none" when something required is absent. kind: wrong (incorrect, or works against the goal), unsupported (a claim without support, exaggerated or universal), weak (right idea but vague or underdeveloped), missing (a required point is absent). priority: major (costs a required part, is a material error, or breaks the central argument), medium, minor (optional polish). reason: why it costs credit, tied to the words. fix: one concrete change that starts with a verb. Empty if there is no problem.
- note: 1 or 2 sentences summing up this check for the student.
- verdict: {verdicts}
- rewrite: the most useful improved version for your top concern. sourceId is the sentence it replaces, or "none" if it is a new addition. 1 or 2 sentences that keep the student's useful ideas; not a full answer. Use empty text when nothing should change."""

DIM_VERDICTS = ("good (clearly done), partial (done with a material gap), missing (not done at all), unsupported "
                "(the main problem is unsupported or exaggerated claims), wrong (the main problem is an error or a "
                "harmful suggestion), not_applicable (this check does not apply to this question; then credit and "
                "concerns are empty).")
PART_VERDICT_TEXT = ("met (this part is answered), partial (relevant substance with a material gap), missing (no "
                     "sentence really answers this part; credit must then be empty).")


def schema(sids, verdicts):
    text = {"type": "string"}
    ids = {"type": "string", "enum": list(sids)}
    ids_or_none = {"type": "string", "enum": list(sids) + ["none"]}
    credit = {"type": "object", "additionalProperties": False, "required": ["sourceId", "reason"],
              "properties": {"sourceId": ids, "reason": text}}
    concern = {"type": "object", "additionalProperties": False,
               "required": ["sourceId", "kind", "priority", "reason", "fix"],
               "properties": {"sourceId": ids_or_none, "kind": {"type": "string", "enum": KINDS},
                              "priority": {"type": "string", "enum": PRIORITY}, "reason": text, "fix": text}}
    # Evidence first, verdict after it: the model commits to what it found before it labels the result.
    return {"type": "object", "additionalProperties": False,
            "required": ["credit", "concerns", "note", "verdict", "rewrite"],
            "properties": {
                "credit": {"type": "array", "maxItems": 2, "items": credit},
                "concerns": {"type": "array", "maxItems": 2, "items": concern},
                "note": text,
                "verdict": {"type": "string", "enum": list(verdicts)},
                "rewrite": {"type": "object", "additionalProperties": False, "required": ["sourceId", "text"],
                            "properties": {"sourceId": ids_or_none, "text": text}},
            }}


def task(prep, check_id):
    if check_id.startswith("part:"):
        part = next(p for p in prep["parts"] if p["id"] == check_id[5:])
        count = (f" It asks for {part['count']} items: count the distinct valid items in the answer, even if two share "
                 "one sentence. If the count is already met, deepen one existing item instead of asking for more."
                 if part.get("count") else "")
        others = " Ignore the other parts of the question." if len(prep["parts"]) > 1 else ""
        body = (f"Check only this part of the question: \u00ab{part['quote']}\u00bb.{others}{count} Find every sentence "
                "that answers this part and credit the best ones. A sentence about a different part does not answer "
                "this one. If no sentence answers this part, the verdict is 'missing' with no credit and one concern "
                "(sourceId 'none', kind 'missing').")
        return f"TASK [{check_id}]\n{body}\n\n" + FIELDS.format(verdicts=PART_VERDICT_TEXT)
    d = DIM[check_id]
    return (f"TASK [{check_id}]\nCheck only this: {d['en']}. {d['task']} Judge only this check, not the whole "
            "answer.\n\n" + FIELDS.format(verdicts=DIM_VERDICTS))


def bind(raw, prep, check_id):
    """Validate one check's JSON against the student's sentences and the quality rules."""
    is_part = check_id.startswith("part:")
    spans = {s["id"]: s["text"] for s in prep["sentences"]}
    lang, src = prep["lang"], prompts.sources(prep)
    if not isinstance(raw, dict):
        raise Rejected("shape")
    verdict = raw.get("verdict")
    if verdict not in (PART_VERDICTS if is_part else VERDICTS):
        raise Rejected("verdict")

    credit, seen = [], set()
    for c in raw.get("credit") or []:
        sid = c.get("sourceId")
        if sid not in spans or sid in seen:
            raise Rejected("credit:source")
        seen.add(sid)
        credit.append({"sourceId": sid, "quote": spans[sid],
                       "reason": clean(c.get("reason"), lang, src, hi=400, field="credit")})

    concerns, seen = [], set()
    for c in raw.get("concerns") or []:
        sid, kind, priority = c.get("sourceId"), c.get("kind"), c.get("priority")
        if kind not in KINDS or priority not in PRIORITY:
            raise Rejected("concern:enum")
        if sid == "none":
            if kind != "missing":
                raise Rejected("concern:none_needs_missing")
            sid = None
        elif sid not in spans:
            raise Rejected("concern:source")
        if (sid, kind) in seen:
            continue  # two concerns about the same sentence with the same kind are redundant; keep the first
        seen.add((sid, kind))
        concerns.append({"sourceId": sid, "quote": spans.get(sid), "kind": kind, "priority": priority,
                         "reason": clean(c.get("reason"), lang, src, hi=400, field="reason"),
                         "fix": clean(c.get("fix"), lang, src, hi=300, field="fix")})
    if len(concerns) == 2 and similar(concerns[0]["fix"], concerns[1]["fix"], 0.7):
        raise Rejected("concern:same_fix_twice")

    if is_part:
        if verdict == "missing" and credit:
            credit = []  # a part with no real answer earns no credit; keep the verdict instead of rejecting
        if verdict == "met" and not credit:
            raise Rejected("part:met_without_credit")
        if verdict in ("partial", "missing") and not concerns:
            raise Rejected("part:gap_without_concern")
    else:
        if verdict == "not_applicable" and concerns:
            # The model judged the check not needed but still found problems: the problems stand, so it is a
            # partial result, not a rejected one.
            verdict = "partial"
        if verdict == "good" and not credit:
            raise Rejected("check:good_without_credit")
        if verdict in ("partial", "missing", "unsupported", "wrong") and not concerns:
            raise Rejected("check:gap_without_concern")
        if verdict == "missing" and credit and check_id == "scope":
            credit = []  # an off-topic answer earns no credit; keep scope 'missing' instead of rejecting

    if check_id == "evidence":
        # A missing example weakens an answer but does not cost a required part: never a top-priority finding.
        for c in concerns:
            if c["priority"] == "major":
                c["priority"] = "medium"

    rewrite = None
    rw = raw.get("rewrite") or {}
    # The examiner cannot supply a real example or a corrected fact without inventing one, so the evidence and
    # accuracy checks give advice only, never a "try writing" line.
    if check_id not in NO_REWRITE and isinstance(rw.get("text"), str) and rw["text"].strip():
        rsid = rw.get("sourceId")
        if rsid == "none":
            rsid = None
        elif rsid not in spans:
            raise Rejected("rewrite:source")
        out = clean(rw["text"], lang, src, lo=15, hi=450, field="rewrite")
        if len([s for s in SENTENCE_END.split(out) if s.strip()]) > 3:
            raise Rejected("rewrite:too_long")
        if rsid and similar(out, spans[rsid], 0.85):
            raise Rejected("rewrite:unchanged")
        rewrite = {"sourceId": rsid, "text": out}

    return {"verdict": verdict, "credit": credit, "concerns": concerns, "rewrite": rewrite,
            "note": clean(raw.get("note"), lang, src, hi=500, field="note")}


def run(examiner, prep, system, check_id):
    kind = "part" if check_id.startswith("part:") else "check"
    sids = [s["id"] for s in prep["sentences"]]
    try:
        value = examiner.ask(stage="check", check=check_id, system=system, task=task(prep, check_id),
                             schema=schema(sids, PART_VERDICTS if kind == "part" else VERDICTS), max_tokens=900,
                             validate=lambda raw: bind(raw, prep, check_id))
    except Rejected as e:
        # Not judged: the report shows this check as "Not checked" and adds no finding for it. Inventing a
        # verdict here would put feedback in front of the student that no examiner pass actually made.
        return {"id": check_id, "kind": kind, "status": "unavailable", "reason": e.code}
    return {"id": check_id, "kind": kind, "status": "ok", **value}
