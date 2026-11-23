"""Stage: the focused checks. One Gemma call per question part, plus the twelve dimension checks.

Every finding points at a sentence ID that Gemma can only choose from the student's own sentences (the JSON schema
lists them), so a quote shown to the student is always the student's exact words. Text fields go through
`text.clean`, and the verdict must agree with the evidence (for example "met" needs credit, "missing" allows none).
"""
from . import prompts
from .rubric import DIM
from .text import Rejected, SENTENCE_END, clean, similar, tokens, unsourced_names, unsourced_numbers

VERDICTS = ["good", "partial", "missing", "unsupported", "wrong", "not_applicable"]
PART_VERDICTS = ["met", "partial", "missing"]
KINDS = ["wrong", "unsupported", "weak", "missing"]
PRIORITY = ["major", "medium", "minor"]
NO_REWRITE = ("evidence", "accuracy")

FIELDS = """Fill the JSON fields in this order:
- credit: up to 2 of the student's sentences that earn credit for THIS check, each with a specific reason. Empty if none does.
- concerns: up to 2 of the most important problems for THIS check, most important first. sourceId is the sentence with the problem, or "none" when something required is absent. kind: wrong (incorrect, or works against the goal), unsupported (a claim without support, exaggerated or universal), weak (right idea but vague or underdeveloped), missing (a required point is absent). priority: major (costs a required part, is a material error, or breaks the central argument), medium, minor (optional polish). reason: why it costs credit, tied to the words. fix: one concrete change that starts with a verb. refId: the id of the source the reason or fix relies on, or "none". Empty if there is no problem.
- note: 1 or 2 sentences summing up this check for the student.
- verdict: {verdicts}
- rewrite: the most useful improved version for your top concern. sourceId is the sentence it replaces, or "none" if it is a new addition. 1 or 2 sentences that keep the student's useful ideas; not a full answer. refId as for concerns. Use empty text when nothing should change."""

DIM_VERDICTS = ("good (clearly done), partial (done with a material gap), missing (not done at all), unsupported "
                "(the main problem is unsupported or exaggerated claims), wrong (the main problem is an error or a "
                "harmful suggestion), not_applicable (this check does not apply to this question; then credit and "
                "concerns are empty).")
PART_VERDICT_TEXT = ("met (this part is answered with its main substance: what an examiner expects for this demand "
                     "is there), partial (relevant substance, but only one narrow aspect or with a material gap: for "
                     "example only awareness when the part asks how a plan reduces risk), missing (no sentence really "
                     "answers this part; credit must then be empty).")


def schema(sids, verdicts, ref_ids=()):
    text = {"type": "string"}
    ids = {"type": "string", "enum": list(sids)}
    ids_or_none = {"type": "string", "enum": list(sids) + ["none"]}
    ref = {"type": "string", "enum": list(ref_ids) + ["none"]}
    credit = {"type": "object", "additionalProperties": False, "required": ["sourceId", "reason"],
              "properties": {"sourceId": ids, "reason": text}}
    concern = {"type": "object", "additionalProperties": False,
               "required": ["sourceId", "kind", "priority", "reason", "fix", "refId"],
               "properties": {"sourceId": ids_or_none, "kind": {"type": "string", "enum": KINDS},
                              "priority": {"type": "string", "enum": PRIORITY}, "reason": text, "fix": text,
                              "refId": ref}}
    # Evidence first, verdict after it: the model commits to what it found before it labels the result.
    return {"type": "object", "additionalProperties": False,
            "required": ["credit", "concerns", "note", "verdict", "rewrite"],
            "properties": {
                "credit": {"type": "array", "maxItems": 2, "items": credit},
                "concerns": {"type": "array", "maxItems": 2, "items": concern},
                "note": text,
                "verdict": {"type": "string", "enum": list(verdicts)},
                "rewrite": {"type": "object", "additionalProperties": False,
                            "required": ["sourceId", "text", "refId"],
                            "properties": {"sourceId": ids_or_none, "text": text, "refId": ref}},
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


def _cited(prep, ref_id):
    return next((r for r in prep.get("refs") or [] if r["id"] == ref_id), None)


def _uses(prep, ref, text):
    """A citation is kept only when the cited words really come from the source, so the student never sees a source
    label on advice that does not use it."""
    base = prompts.sources(prep)
    if unsourced_names(text, *base) or unsourced_numbers(text, *base):
        return True  # it names something only the source has (and `clean` has already checked it is there)
    new = {t for t in tokens(text) & tokens(ref["text"]) - tokens(" ".join(base)) if len(t) >= 5}
    return len(new) >= 2


def bind(raw, prep, check_id):
    """Validate one check's JSON against the student's sentences and the quality rules."""
    is_part = check_id.startswith("part:")
    spans = {s["id"]: s["text"] for s in prep["sentences"]}
    lang, src = prep["lang"], prompts.sources(prep)
    # Credit and notes may use any source extract given; a concern or rewrite may name a source's facts only when it
    # cites that source, and is checked against that source alone.
    src_all = prompts.sources(prep, True)
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
                       "reason": clean(c.get("reason"), lang, src_all, hi=400, field="credit")})

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
                         **_with_source(prep, c, lang, src)})
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
        ref = _cited(prep, rw.get("refId"))
        out = clean(rw["text"], lang, prompts.sources(prep, (ref["id"],)) if ref else src, lo=15, hi=450,
                    field="rewrite")
        if len([s for s in SENTENCE_END.split(out) if s.strip()]) > 3:
            raise Rejected("rewrite:too_long")
        if rsid and similar(out, spans[rsid], 0.85):
            raise Rejected("rewrite:unchanged")
        rewrite = {"sourceId": rsid, "text": out, "ref": ref["id"] if ref and _uses(prep, ref, out) else None}

    return {"verdict": verdict, "credit": credit, "concerns": concerns, "rewrite": rewrite,
            "note": clean(raw.get("note"), lang, src_all, hi=500, field="note")}


def _with_source(prep, c, lang, src):
    ref = _cited(prep, c.get("refId"))
    use = prompts.sources(prep, (ref["id"],)) if ref else src
    reason = clean(c.get("reason"), lang, use, hi=400, field="reason")
    fix = clean(c.get("fix"), lang, use, hi=300, field="fix")
    return {"reason": reason, "fix": fix, "ref": ref["id"] if ref and _uses(prep, ref, f"{reason} {fix}") else None}


def run(examiner, prep, system, check_id):
    kind = "part" if check_id.startswith("part:") else "check"
    sids = [s["id"] for s in prep["sentences"]]
    ref_ids = [r["id"] for r in prep.get("refs") or []]
    if check_id == "provisions" and not ref_ids:
        # No checked source fits this question (for example a geography or economy topic): nothing to compare with.
        return {"id": check_id, "kind": kind, "status": "ok", "verdict": "not_applicable", "credit": [],
                "concerns": [], "rewrite": None, "note": None}
    try:
        value = examiner.ask(stage="check", check=check_id, system=system, task=task(prep, check_id),
                             schema=schema(sids, PART_VERDICTS if kind == "part" else VERDICTS, ref_ids),
                             max_tokens=900, validate=lambda raw: bind(raw, prep, check_id))
    except Rejected as e:
        # Not judged: the report shows this check as "Not checked" and adds no finding for it. Inventing a
        # verdict here would put feedback in front of the student that no examiner pass actually made.
        return {"id": check_id, "kind": kind, "status": "unavailable", "reason": e.code}
    if check_id == "solutions" and value["verdict"] != "not_applicable":
        value = second_look(examiner, prep, system, value)
    return {"id": check_id, "kind": kind, "status": "ok", **value}


ONE_TASK = ("TASK [solutions:{sid}]\nJudge only this one sentence of the answer: \u00ab{text}\u00bb\n"
            "First decide: does it propose a measure, that is, something someone should do? A claim, a description "
            "of how things work or a conclusion is not a measure: then the verdict is not_applicable and credit and "
            "concerns are empty. If it is a measure, check who acts, what exactly, and where or when it applies. A "
            "measure that should be targeted at a stated need but is offered as a blanket routine (for example "
            "yearly dredging of every canal instead of where a survey shows it is needed) is a concern of kind weak, "
            "with a fix that says where or when it applies. A measure that already says who acts and where it "
            "applies earns credit.\n\n")
MAX_SECOND_LOOKS = 6


def second_look(examiner, prep, system, value):
    """Gemma 12B judges the worst suggestion and stops (run 45169827637: concreting flagged, routine desilting
    credited). Asked about all the remaining sentences at once, it again picks the most obvious problem, even a
    claim (run 45178857154). So each remaining sentence gets its own short call, with only that sentence to cite."""
    flagged = sorted({c["sourceId"] for c in value["concerns"] if c["sourceId"]})
    rest = [s for s in prep["sentences"] if s["id"] not in flagged][:MAX_SECOND_LOOKS]
    if not flagged or not rest:
        return value
    added = []
    for s in rest:
        try:
            said, one = examiner.ask(
                stage="check", check=f"solutions:{s['id']}", system=system,
                task=ONE_TASK.format(sid=s["id"], text=s["text"]) + FIELDS.format(verdicts=DIM_VERDICTS),
                schema=schema([s["id"]], VERDICTS, [r["id"] for r in prep.get("refs") or []]), max_tokens=600,
                validate=lambda raw: (raw.get("verdict") if isinstance(raw, dict) else None,
                                      bind(raw, prep, "solutions")))
        except Rejected:
            continue  # this sentence keeps whatever the first pass said about it
        # Only a concern about this very sentence counts, and only when Gemma itself said it is a measure.
        hit = [c for c in one["concerns"] if c["sourceId"] == s["id"]]
        if said != "not_applicable" and hit:
            added.append(hit[0])
    if not added:
        return value
    out = dict(value, concerns=value["concerns"] + added)
    if out["verdict"] == "good":
        out["verdict"] = "partial"
    return out
