"""Stage: the focused checks. One Gemma call per question part, plus the twelve dimension checks.

Every finding points at a sentence ID that Gemma can only choose from the student's own sentences (the JSON schema
lists them), so a quote shown to the student is always the student's exact words. Text fields go through
`text.clean`, and the verdict must agree with the evidence (for example "met" needs credit, "missing" allows none).
"""
import math
import re

from . import prompts
from .rubric import DIM
from .text import Rejected, SENTENCE_END, clean, fold, similar, tokens, unsourced_names, unsourced_numbers

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

    credit, seen, dropped = [], set(), None
    for c in raw.get("credit") or []:
        sid = c.get("sourceId")
        if sid not in spans or sid in seen:
            raise Rejected("credit:source")
        seen.add(sid)
        try:
            reason = clean(c.get("reason"), lang, src_all, hi=400, field="credit")
        except Rejected as e:
            # Credit is optional: a credit line naming something MATERIALS does not have is dropped, not retried.
            # Three rejected attempts over one credit name left a whole check "Not checked" (run 45223096235).
            if e.code.endswith(("unsourced_name", "unsourced_number")):
                dropped = e.code
                continue
            raise
        credit.append({"sourceId": sid, "quote": spans[sid], "reason": reason})

    concerns, seen, dropped_c = [], set(), None
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
        try:
            text = _with_source(prep, c, lang, src)
        except Rejected as e:
            # A finding that names a fact no source has is dropped, not retried: three rejected attempts over one
            # name left whole checks "Not checked" and three Polity reports unbuilt (split run 45252594185).
            if e.code.endswith(("unsourced_name", "unsourced_number", "copied")):
                dropped_c = e.code
                continue
            raise
        seen.add((sid, kind))
        concerns.append({"sourceId": sid, "quote": spans.get(sid), "kind": kind, "priority": priority, **text})
    if len(concerns) == 2 and similar(concerns[0]["fix"], concerns[1]["fix"], 0.7):
        raise Rejected("concern:same_fix_twice")

    if is_part:
        if verdict == "missing" and credit:
            credit = []  # a part with no real answer earns no credit; keep the verdict instead of rejecting
        if verdict == "met" and not credit:
            raise Rejected("part:met_without_credit")
        if verdict in ("partial", "missing") and not concerns and not dropped_c:
            # v10: when the only concern named an unsourced fact it is dropped and the verdict stands; three retries
            # over one name left parts and checks "Not checked" in v9 (split run 45324728302).
            raise Rejected("part:gap_without_concern")
    else:
        if verdict == "not_applicable" and concerns:
            # The model judged the check not needed but still found problems: the problems stand, so it is a
            # partial result, not a rejected one.
            verdict = "partial"
        if verdict == "good" and not credit:
            raise Rejected(dropped or "check:good_without_credit")
        if verdict in ("partial", "missing", "unsupported", "wrong") and not concerns:
            if not dropped_c:
                raise Rejected("check:gap_without_concern")
            if verdict in ("unsupported", "wrong"):
                verdict = "partial"  # the finding behind a red verdict was dropped: no red without a finding
        if verdict == "missing" and credit and check_id == "scope":
            credit = []  # an off-topic answer earns no credit; keep scope 'missing' instead of rejecting

    if check_id == "evidence":
        # A missing example weakens an answer but does not cost a required part: never a top-priority finding, and
        # never red. A general claim with no example is "could be stronger" (split run 45252594185 marked "frequent
        # use weakened State autonomy" red as if it were a factual error).
        for c in concerns:
            if c["priority"] == "major":
                c["priority"] = "medium"
            if c["kind"] == "unsupported":
                c["kind"] = "weak"

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
        try:
            out = clean(rw["text"], lang, prompts.sources(prep, (ref["id"],)) if ref else src, lo=15, hi=450,
                        field="rewrite")
        except Rejected as e:
            if not e.code.endswith(("unsourced_name", "unsourced_number")):
                raise
            out = None  # a rewrite is optional: one that names an unsourced fact is left out, not retried
        if out is not None and licensed(ref) and copies(out, ref["text"]):
            out = None
        if out is not None:
            if len([s for s in SENTENCE_END.split(out) if s.strip()]) > 3:
                raise Rejected("rewrite:too_long")
            if not (rsid and similar(out, spans[rsid], 0.85)):
                rewrite = {"sourceId": rsid, "text": out,
                           "ref": ref["id"] if ref and _uses(prep, ref, out) else None}

    try:
        note = clean(raw.get("note"), lang, src_all, hi=500, field="note")
    except Rejected as e:
        if not e.code.endswith(("unsourced_name", "unsourced_number")):
            raise
        note = None  # the note is a summary line: without it the findings still stand
    return {"verdict": verdict, "credit": credit, "concerns": concerns, "rewrite": rewrite, "note": note}


COPY_RUN = 12  # most consecutive words a finding may take from a licensed book or textbook passage


def copies(text, passage, run=COPY_RUN):
    """True when `text` repeats `run` or more consecutive words of `passage`."""
    a = re.findall(r"\w+", fold(text or ""))
    b = " " + " ".join(re.findall(r"\w+", fold(passage or ""))) + " "
    return any(" " + " ".join(a[i:i + run]) + " " in b for i in range(0, max(0, len(a) - run + 1)))


def licensed(ref):
    return bool(ref and (ref.get("private") or ref.get("label", "").startswith(("NCERT", "Laxmikanth"))))


def _with_source(prep, c, lang, src):
    ref = _cited(prep, c.get("refId"))
    use = prompts.sources(prep, (ref["id"],)) if ref else src
    reason = clean(c.get("reason"), lang, use, hi=400, field="reason")
    fix = clean(c.get("fix"), lang, use, hi=300, field="fix")
    if licensed(ref) and copies(f"{reason} {fix}", ref["text"]):
        raise Rejected("fix:copied")  # dropped like an unsourced fact: paraphrase only
    return {"reason": reason, "fix": fix, "ref": ref["id"] if ref and _uses(prep, ref, f"{reason} {fix}") else None}


FACT_TASK = """TASK [facts:{sid}]
Check one claim of the student's answer against the extracts below. They are exact text from the Constitution of India or NCERT textbooks.
CLAIM ({sid}): \u00ab{text}\u00bb
EXTRACTS
{extracts}
First answer bothTrue: could the claim and the extracts all be true at the same time? "no" only when the claim and an extract cannot both hold: a different body, office, House, power, time limit, number or condition for the same thing. An extract that describes one way something is done does not rule out a claim that names another way, unless the extract excludes it ("only", "shall not", another body taking the same decision). A different or more common name for the same thing is not a contradiction: "block" for the intermediate level, "Lok Sabha" for the House of the People, "Rajya Sabha" for the Council of States, "Centre" for the Union. Then decide, from these extracts only:
- contradicted: bothTrue is "no".
- confirmed: an extract states the same thing as the claim.
- not_covered: no extract speaks to this claim. This is the usual case for opinions, examples, effects and claims about practice.
If contradicted: sourceWords = the exact words copied from the extract that contradict the claim; wrongWords = the exact words copied from THIS claim that are wrong; one concern on {sid}, kind wrong, priority major (minor only for a small slip), refId = the extract that contradicts it; reason = what that extract says instead; fix = the corrected statement in one sentence, in words from that extract. Judge only the words of this claim: an error in another sentence of the answer is not this claim's error. If confirmed: credit {sid} with a reason that names the extract's provision; concerns empty. If not_covered: credit and concerns empty. sourceWords and wrongWords are empty unless contradicted.

"""
FACT_VERDICTS = {"contradicted": "wrong", "confirmed": "good", "not_covered": "not_applicable"}
MAX_FACTS = 8


GAP_TASK = """TASK [gap:{pid}]
Check only this part of the question: \u00ab{part}\u00bb. Using only the sources in MATERIALS, name the single most important point that answers this part, that a sharp examiner expects, and that no sentence of the answer makes: a provision, body, power, safeguard or limit stated in a source. Give it as one concern with sourceId "none", kind missing, priority medium, refId = that source; reason = why this point matters for this part; fix = what to add, in words from that source. If the answer already makes every such point, or no source fits this part, concerns are empty, and the verdict is good (credit the sentence that makes the point) or not_applicable.

"""


def gap_ids(prep):
    """One "what the sources add" check per question part, when the question has shared extracts."""
    if not any(r.get("shared", True) for r in prep.get("refs") or []):
        return []
    return [f"gap:{p['id']}" for p in prep.get("parts") or []]


def run_gap(examiner, prep, system, check_id):
    """The one combined "provisions" check named the answer's error and stopped: none of its suggested additions
    used a source (split run 45238473782). A separate small call per part asks only for the missing point."""
    pid = check_id.split(":", 1)[1]
    part = next(p for p in prep["parts"] if p["id"] == pid)
    shown = [r["id"] for r in prep["refs"] if r.get("shared", True)]
    sids = [s["id"] for s in prep["sentences"]]

    def validate(raw):
        if isinstance(raw, dict):
            raw = dict(raw, concerns=[c for c in raw.get("concerns") or []
                                      if isinstance(c, dict) and c.get("sourceId") == "none"],
                       rewrite={"sourceId": "none", "text": "", "refId": "none"})
            for c in raw["concerns"]:
                c["kind"] = "missing"
        out = bind(raw, prep, "gap")
        out["concerns"] = [c for c in out["concerns"] if c["kind"] == "missing" and c.get("ref")][:1]
        out["rewrite"] = None
        return out

    try:
        value = examiner.ask(stage="check", check=check_id, system=system,
                             task=GAP_TASK.format(pid=pid, part=prompts.escape(part["quote"]))
                             + FIELDS.format(verdicts=DIM_VERDICTS),
                             schema=schema(sids, VERDICTS, shown), max_tokens=500, validate=validate)
    except Rejected as e:
        return {"id": check_id, "kind": "fact", "status": "unavailable", "reason": e.code}
    return {"id": check_id, "kind": "fact", "status": "ok", **value}


def fact_ids(prep):
    """One claim check per answer sentence that has its own extracts, in answer order, at most MAX_FACTS."""
    claimed = {sid for r in prep.get("refs") or [] for sid in r.get("claims", [])}
    return [f"facts:{s['id']}" for s in prep["sentences"] if s["id"] in claimed][:MAX_FACTS]


def _flat(value):  # case, spacing and punctuation do not matter when words are compared
    return re.findall(r"\w+", fold(value or ""))


def longest_run(words, text_words):
    """Longest run of consecutive `words` that appears, in order and unbroken, in `text_words`."""
    hay = " " + " ".join(text_words) + " "
    best = 0
    for i in range(len(words)):
        for j in range(len(words), i + best, -1):
            if " " + " ".join(words[i:j]) + " " in hay:
                best = j - i
                break
    return best


def quoted_from(said, extracts):
    """Indices of the extracts the quoted words come from. Gemma 12B trims or joins a few words when it copies a
    clause (v10 Governor: "appointed by the President" was rejected twice and the planted error went unmarked), so a
    quote counts when most of it is one unbroken run of an extract's words."""
    words = _flat(said)
    if len(words) < 3:
        return []
    runs = [longest_run(words, _flat(t)) for t in extracts]
    need = max(3, math.ceil(0.6 * len(words)))
    top = max(runs) if runs else 0
    return [n for n, r in enumerate(runs) if r == top and r >= need]


AGREE = 0.8   # share of the claim's quoted words that the source quote repeats, beyond which they say the same thing


def quote_agrees(wrong_words, source_words):
    """True when the source words Gemma quotes as the clash say what the claim says, in the Constitution's words.

    v12 Money Bill: "After the Lok Sabha passes it, the Bill is sent to the Rajya Sabha for its recommendations" was
    marked wrong with the quote "After a Money Bill has been passed by the House of the People it shall be transmitted
    to the Council of States for its recommendations": the same statement, while the reason was the previous
    sentence's error. A contradiction needs a difference: a negation or an "only" on one side, a different number,
    or words of the claim that the source does not repeat. Tuned on every v12 finding: it rejects that one, and none
    of the eleven planted errors (the closest, "Election Commission of India" against "State Election Commission",
    repeats 0.67 of the claim's words)."""
    from .refs import vocabulary
    from .report import stems
    w, q = stems(vocabulary(wrong_words or "")), stems(vocabulary(source_words or ""))
    if not w or hard_difference(wrong_words, source_words):
        return False
    return len(w & q) >= AGREE * len(w)


def hard_difference(wrong_words, source_words):
    """A difference code can see without judging meaning: a "not" or an "only" on one side and not the other, or a
    different number. Eight of the eleven planted errors have one; the other three (elected/appointed, Election
    Commission of India/State Election Commission, State List/powers a State law gives) need a reading."""
    from .refs import vocabulary
    from .report import number_terms, stems
    w, q = stems(vocabulary(wrong_words or "")), stems(vocabulary(source_words or ""))
    if ("not" in q) != ("not" in w) or ("only" in q) != ("only" in w):
        return True
    a, b = number_terms(wrong_words or ""), number_terms(source_words or "")
    return bool((a or b) and a != b)


def run_fact(examiner, prep, system, check_id):
    """Gemma 12B judges a whole answer by its worst problem (runs 45169827637, 45238473782): a wrong body or time
    limit in a lesser sentence goes unseen. So each claim is checked on its own, against only its own extracts.

    The result is built here, not by `bind`: a contradiction is shown only with verified quotes, and it always carries
    the extract those quotes come from. In v10 the general rule that drops a source label when the advice does not
    seem to use the source removed the Article from six correct findings (the student's own words already named it),
    and the sentence then counted as cleared."""
    sid = check_id.split(":", 1)[1]
    sent = next(s for s in prep["sentences"] if s["id"] == sid)
    mine = [r for r in prep.get("refs") or [] if sid in r.get("claims", [])]
    mine = [dict(r, text=r.get("full") or r["text"]) for r in mine]
    extracts = "\n".join(f"[{r['id']}] {r['label']}: {prompts.escape(r['text'])}" for r in mine)
    task = (FACT_TASK.format(sid=sid, text=prompts.escape(sent["text"]), extracts=extracts)
            + FIELDS.format(verdicts="contradicted, confirmed or not_covered, as decided above"))
    base = schema([sid], list(FACT_VERDICTS), [r["id"] for r in mine])
    props = dict(base["properties"])
    # Evidence first: can both be true, the words that clash, then the finding, then the verdict.
    sch = dict(base, required=["bothTrue", "sourceWords", "wrongWords"] + base["required"],
               properties={"bothTrue": {"type": "string", "enum": ["yes", "no"]}, "sourceWords": {"type": "string"},
                           "wrongWords": {"type": "string"}, **props})
    claim_words = " " + " ".join(_flat(sent["text"])) + " "
    lang = prep["lang"]
    base_src = prompts.sources(prep)

    def text_or(value, fallback, sources, field, hi):
        try:
            return clean(value, lang, sources, hi=hi, field=field)
        except Rejected:
            return fallback

    def validate(raw):
        if not isinstance(raw, dict) or raw.get("verdict") not in FACT_VERDICTS:
            raise Rejected("fact:verdict")
        verdict = raw["verdict"]
        if verdict == "contradicted" and raw.get("bothTrue") == "yes":
            verdict = "not_covered"  # Gemma itself says both can hold: no contradiction (v10 Governor, Article 154)
        if verdict == "not_covered":
            return {"verdict": "not_applicable", "credit": [], "concerns": [], "rewrite": None, "note": None}
        if verdict == "confirmed":
            c = next((c for c in raw.get("credit") or [] if isinstance(c, dict) and c.get("sourceId") == sid), {})
            reason = text_or(c.get("reason"), "This matches the text of the Constitution.",
                             base_src + [prompts.ref_text(r) for r in mine], "credit", 400)
            return {"verdict": "good", "credit": [{"sourceId": sid, "quote": sent["text"], "reason": reason}],
                    "concerns": [], "rewrite": None, "note": None}
        # contradicted: the wrong words must be this claim's own, and the source words an extract's
        wrong = _flat(raw.get("wrongWords"))
        if len(wrong) < 2 or " " + " ".join(wrong) + " " not in claim_words:
            raise Rejected("fact:wrong_words")
        hits = quoted_from(raw.get("sourceWords"), [r["text"] for r in mine])
        if not hits:
            raise Rejected("fact:source_words")
        if quote_agrees(raw.get("wrongWords"), raw.get("sourceWords")):
            raise Rejected("fact:quote_agrees")
        c = next((c for c in raw.get("concerns") or [] if isinstance(c, dict) and c.get("sourceId") == sid), {})
        ref = next((mine[n] for n in hits if mine[n]["id"] == c.get("refId")), mine[hits[0]])
        use = base_src + [prompts.ref_text(ref)]
        quote = " ".join(str(raw.get("sourceWords") or "").split()[:12])
        reason = text_or(c.get("reason"), f"{ref['label']} says: \u00ab{quote}\u00bb.", use, "reason", 400)
        if copies(reason, ref["text"]) and licensed(ref):
            reason = f"This does not match {ref['label']}."
        fix = text_or(c.get("fix"), f"Correct this sentence using {ref['label']}.", use, "fix", 300)
        if copies(fix, ref["text"]) and licensed(ref):
            fix = f"Correct this sentence using {ref['label']}."
        priority = c.get("priority") if c.get("priority") in PRIORITY else "major"
        return {"verdict": "wrong", "credit": [], "rewrite": None, "note": None, "contradicted": True,
                "concerns": [{"sourceId": sid, "quote": sent["text"], "kind": "wrong", "priority": priority,
                              "reason": reason, "fix": fix, "ref": ref["id"], "fromClaim": True,
                              # The verified clash: the student's wrong words and the source's words. The correction
                              # step checks its line against these (see report.correct).
                              "evidence": {"wrongWords": " ".join(str(raw.get("wrongWords") or "").split()),
                                           "sourceWords": " ".join(str(raw.get("sourceWords") or "").split())}}]}

    try:
        value = examiner.ask(stage="check", check=check_id, system=system, task=task, schema=sch, max_tokens=500,
                             validate=validate, attempts=2)
    except Rejected as e:
        if e.code in ("fact:wrong_words", "fact:source_words", "fact:quote_agrees"):
            # Gemma said "contradicted" twice but could not show the clash. Neither an error nor a clearance: the
            # sentence keeps whatever the other checks found (v10 counted this as cleared and lost the Governor error).
            return {"id": check_id, "kind": "fact", "status": "ok", "verdict": "not_applicable", "credit": [],
                    "concerns": [], "rewrite": None, "note": None, "unproven": True}
        return {"id": check_id, "kind": "fact", "status": "unavailable", "reason": e.code}
    return {"id": check_id, "kind": "fact", "status": "ok", **value}


def run(examiner, prep, system, check_id):
    if check_id.startswith("facts:"):
        return run_fact(examiner, prep, system, check_id)
    if check_id.startswith("gap:"):
        return run_gap(examiner, prep, system, check_id)
    kind = "part" if check_id.startswith("part:") else "check"
    sids = [s["id"] for s in prep["sentences"]]
    ref_ids = [r["id"] for r in prep.get("refs") or [] if r.get("shared", True)]
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
