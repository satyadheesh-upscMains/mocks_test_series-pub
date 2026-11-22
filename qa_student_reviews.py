"""QA only: synthetic student cases taken from the local submission endpoint."""
import json
import os
from pathlib import Path
import time

import grader

ADDENDUM = """
Before choosing the main gap, map EACH explicit task in the question against what is actually in the answer.
If one task is missing or a requested number is not met, name that first and anchor the next step to it.
Never describe something the answer already does as missing; for a strong answer give one incremental improvement.
If misleading or unsupported evidence is the biggest problem, name the exact claim and say what to verify or replace.
Do not substitute a made-up fact or number. Distinguish 'unsupported by this answer' from 'wrong'.
For a completely off-topic answer, relevance, coverage, analysis and evidence are all 0. Explain how to start over.
For the strength field of an off-topic answer, write a complete coaching sentence explaining that no exam-relevant strength is present. Do not write only "None".
Make nextStep specific enough to write immediately, in one or two short sentences. Feedback in natural Hindi if requested.
"""


def main():
    group = os.environ["QA_CASE"]
    jobs = [r for r in json.loads(Path("student_cases.json").read_text()) if r["id"] == group]
    assert len(jobs) == 1
    base = grader.SYSTEM
    print("Loading exact production Gemma build", flush=True)
    llm = grader.load_model()
    original_call = llm.create_chat_completion
    diag = {}
    def inspected_completion(*args, **kwargs):
        out = original_call(*args, **kwargs)
        choice = out["choices"][0]
        diag["finish"] = choice["finish_reason"]
        diag["tokens"] = out.get("usage", {}).get("completion_tokens")
        try:
            obj = json.loads(choice["message"]["content"])
            diag["field_lengths"] = {k: len(obj[k]) for k in ("strength", "gap", "nextStep") if isinstance(obj.get(k), str)}
            diag["scores"] = obj.get("scores")
        except (TypeError, ValueError, AttributeError):
            pass
        return out
    llm.create_chat_completion = inspected_completion
    output = []
    for variant in ("baseline", "candidate"):
        grader.SYSTEM = base if variant == "baseline" else base + ADDENDUM
        for j in jobs:
            diag.clear()
            t0 = time.monotonic()
            try:
                report = grader.evaluate(llm, j)
                result = {"report": report}
            except Exception as exc:
                print(f"Validation failed: {type(exc).__name__}", flush=True)
                result = {"error": type(exc).__name__, "detail": str(exc)[:140], "diagnostic": dict(diag)}
            elapsed = round(time.monotonic() - t0, 1)
            print(f"{variant}: {elapsed}s", flush=True)
            output.append({"id": j["id"], "variant": variant, "seconds": elapsed, **result})
            Path(f"reports-{group}.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
