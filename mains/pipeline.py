"""The evaluation pipeline as separate stages, so each stage can run on its own GitHub runner.

    prepare  -> number the answer's sentences, count words, break the question into parts   (1 runner)
    shards   -> parts | question | content | reasoning | writing, each a few focused checks (5 runners in parallel)
    finish   -> second opinion on serious findings, assemble the report, short summary      (1 runner)

`evaluate` runs the same stages one after another on a single runner. Both paths give the same report.
"""
from . import breakdown, checks, prompts, refs, report, verify
from .model import log
from .rubric import FORMATS, SHARDS, word_limit
from .text import Rejected, norm, sentences, similar, word_count

SHARD_ORDER = tuple(SHARDS)


def prepare(examiner, job, cached_breakdown=None):
    if job.get("lang") not in ("en", "hi"):
        raise Rejected("job:lang")
    sents = sentences(job.get("answer") or "")
    if not sents:
        raise Rejected("job:empty_answer")
    # The student chooses the question's format on the site (gs-10, gs-15, ethics-case-20, ...); marks come with it.
    fmt = job.get("format") if job.get("format") in FORMATS else None
    marks = job.get("marks") if isinstance(job.get("marks"), int) else (FORMATS[fmt]["marks"] if fmt else None)
    prep = {
        "id": job["id"], "lang": job["lang"], "question": norm(job["question"]),
        "references": [norm(r) for r in job.get("references") or [] if norm(r)], "marks": marks,
        "format": fmt, "paper": FORMATS[fmt]["paper"] if fmt in FORMATS else "gs",
        "sentences": sents,
        "stats": {"words": word_count(job["answer"]), "sentences": len(sents),
                  "paragraphs": max(s["para"] for s in sents),
                  "limit": word_limit(job["question"], fmt, marks, job.get("wordLimit"))},
    }
    prep["refs"] = sources_for(prep)
    bd = cached_breakdown or breakdown.run(examiner, prep)
    prep.update(parts=bd["parts"], directives=bd["directives"], breakdown={"source": bd.get("source", "cache"),
                                                                          "key": breakdown.key(prep["question"])})
    return prep


def sources_for(prep):
    """Checked extracts from the Constitution and NCERT for this question. A missing index never stops grading."""
    try:
        found = refs.pack(prep["question"], prep["sentences"])
    except Exception:
        found = []
    log(stage="refs", passages=len(found), shared=sum(r.get("shared", True) for r in found),
        claims=len({sid for r in found for sid in r.get("claims", [])}),
        constitution=sum(r["label"].startswith("Constitution") for r in found))
    return found


def check_ids(prep, shard):
    if shard == "parts":
        return [f"part:{p['id']}" for p in prep["parts"]]
    if shard == "content":
        return list(SHARDS[shard]) + checks.fact_ids(prep) + checks.gap_ids(prep)
    return list(SHARDS[shard])


def merge_facts(results):
    """Claim checks become findings of the "Constitution and textbook basics" check, so the report, the second
    opinion and the student see one check, with each wrong claim marked on its own sentence."""
    extra = ("facts:", "gap:")
    facts = [r for r in results if r["id"].startswith(extra) and r["status"] == "ok"]
    rest = [r for r in results if not r["id"].startswith(extra)]
    # A claim check that read a sentence against its own extracts and found no clash overrules any other check that
    # calls that sentence wrong on a source's authority; and an error found in one sentence is not repeated on
    # another (split run 45252594185: the DPSP error was marked on three correct sentences).
    # Cleared: the claim check read the sentence against its extracts and gave a verdict, confirmed or not covered.
    # A contradiction, or a claim it could not settle ("unproven"), clears nothing.
    cleared = {r["id"].split(":", 1)[1] for r in facts if r["id"].startswith("facts:")
               and r.get("verdict") in ("good", "not_applicable") and not r.get("unproven")
               and not any(c["kind"] == "wrong" for c in r["concerns"])}
    found = [c for r in facts if r["id"].startswith("facts:") for c in r["concerns"] if c["kind"] == "wrong"]
    wrong_at = {c["sourceId"] for c in found}
    for r in rest:
        if r["status"] != "ok":
            continue
        rw = r.get("rewrite") or {}
        keep = []
        for c in r["concerns"]:
            sid = c.get("sourceId")
            # Only the claim check may call a sentence wrong on a source's authority. v9 Governor s3 was cleared by its
            # claim check yet marked wrong by "Constitution and textbook basics" against Article 154's literal
            # wording: the source came through the check's rewrite, not the concern, so the old rule missed it.
            on_source = bool(c.get("ref")) or r["id"] == "provisions" or (rw.get("ref") and rw.get("sourceId") == sid)
            if c["kind"] in ("wrong", "unsupported") and on_source and (sid in cleared or sid in wrong_at):
                continue
            if sid and any(f["sourceId"] != sid and similar(f["reason"], c["reason"], 0.45) for f in found):
                continue
            keep.append(c)
        if len(keep) != len(r["concerns"]):
            r["concerns"] = keep
            r["revised"] = True
    concerns = [c for r in facts for c in r["concerns"]]
    flagged = {c["sourceId"] for c in concerns}
    credit = [c for r in facts for c in r["credit"] if c["sourceId"] not in flagged]
    if not concerns and not credit:
        return rest
    at = next((n for n, r in enumerate(rest) if r["id"] == "provisions"), None)
    prov = rest[at] if at is not None else None
    if prov is None or prov["status"] != "ok":
        prov = {"id": "provisions", "kind": "check", "status": "ok", "verdict": "not_applicable", "credit": [],
                "concerns": [], "rewrite": None, "note": None}
        if at is None:
            rest.append(prov)
        else:
            rest[at] = prov
    # The claim check's finding replaces the check's own one on the same sentence: it carries the Article and the
    # verified quotes (v9: four planted errors were caught with no Article shown, because the uncited finding won).
    prov["concerns"] = [c for c in prov["concerns"] if not (c["kind"] == "wrong" and c["sourceId"] in wrong_at)]
    have = {(c["sourceId"], c["kind"]) for c in prov["concerns"]}
    for c in concerns:
        if (c["sourceId"], c["kind"]) not in have:
            prov["concerns"].append(c)
            have.add((c["sourceId"], c["kind"]))
    seen = {c["sourceId"] for c in prov["credit"]}
    prov["credit"] = [c for c in prov["credit"] if c["sourceId"] not in flagged]
    for c in credit:
        if c["sourceId"] not in seen and len(prov["credit"]) < 3:
            prov["credit"].append(c)
            seen.add(c["sourceId"])
    if any(c["kind"] == "wrong" for c in concerns) and prov["verdict"] in ("good", "partial", "not_applicable",
                                                                            "missing"):
        prov["verdict"] = "wrong"
    elif concerns and prov["verdict"] in ("good", "not_applicable"):
        prov["verdict"] = "partial"
    elif prov["verdict"] == "not_applicable" and prov["credit"]:
        prov["verdict"] = "good"
    if not prov.get("note"):
        prov["note"] = (concerns or prov["credit"])[0]["reason"]
    return rest


def task_ids(prep):
    """Every focused check for this answer, in the fixed order: one per question part, then each dimension. Each is
    independent of the others, so each can run on its own runner; `finish` takes their results in any order."""
    return [cid for shard in SHARD_ORDER for cid in check_ids(prep, shard)]


def run_task(examiner, prep, check_id):
    return checks.run(examiner, prep, prompts.system(prep), check_id)


def run_shard(examiner, prep, shard):
    system = prompts.system(prep)
    return [checks.run(examiner, prep, system, cid) for cid in check_ids(prep, shard)]


def finish(examiner, prep, results):
    system = prompts.system(prep)
    # A question part that could not be judged would leave the report unbuilt: it gets one more try here (split run
    # 45252594185 lost three of nine reports this way).
    results = [checks.run(examiner, prep, system, r["id"]) if r["kind"] == "part" and r["status"] != "ok" else r
               for r in results]
    results = verify.run(examiner, prep, system, merge_facts(results))
    out = report.assemble(prep, results)
    report.correct(examiner, prep, system, out)
    out["summary"] = report.summarize(examiner, prep, system, out)
    return out


def evaluate(examiner, job, between=None):
    """All stages on one runner. `between()` runs after each stage (the queue worker renews its lease there)."""
    tick = between or (lambda: None)
    prep = prepare(examiner, job)
    tick()
    results = []
    for shard in SHARD_ORDER:
        results += run_shard(examiner, prep, shard)
        tick()
    return finish(examiner, prep, results)
