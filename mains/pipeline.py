"""The evaluation pipeline as separate stages, so each stage can run on its own GitHub runner.

    prepare  -> number the answer's sentences, count words, break the question into parts   (1 runner)
    shards   -> parts | question | content | reasoning | writing, each a few focused checks (5 runners in parallel)
    finish   -> second opinion on serious findings, assemble the report, short summary      (1 runner)

`evaluate` runs the same stages one after another on a single runner. Both paths give the same report.
"""
from . import breakdown, checks, prompts, refs, report, verify
from .model import log
from .rubric import SHARDS, usual_limit
from .text import Rejected, norm, sentences, word_count

SHARD_ORDER = tuple(SHARDS)


def prepare(examiner, job, cached_breakdown=None):
    if job.get("lang") not in ("en", "hi"):
        raise Rejected("job:lang")
    sents = sentences(job.get("answer") or "")
    if not sents:
        raise Rejected("job:empty_answer")
    prep = {
        "id": job["id"], "lang": job["lang"], "question": norm(job["question"]),
        "references": [norm(r) for r in job.get("references") or [] if norm(r)], "marks": job.get("marks"),
        "sentences": sents,
        "stats": {"words": word_count(job["answer"]), "sentences": len(sents),
                  "paragraphs": max(s["para"] for s in sents), "limit": usual_limit(job.get("marks"))},
    }
    prep["refs"] = sources_for(prep)
    bd = cached_breakdown or breakdown.run(examiner, prep)
    prep.update(parts=bd["parts"], directives=bd["directives"], breakdown={"source": bd.get("source", "cache"),
                                                                          "key": breakdown.key(prep["question"])})
    return prep


def sources_for(prep):
    """Checked extracts from the Constitution and NCERT for this question. A missing index never stops grading."""
    try:
        found = refs.pack(prep["question"], " ".join(s["text"] for s in prep["sentences"]))
    except Exception:
        found = []
    log(stage="refs", passages=len(found), constitution=sum(r["label"].startswith("Constitution") for r in found))
    return found


def check_ids(prep, shard):
    if shard == "parts":
        return [f"part:{p['id']}" for p in prep["parts"]]
    return list(SHARDS[shard])


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
    results = verify.run(examiner, prep, system, results)
    out = report.assemble(prep, results)
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
