"""Public QA with synthetic answers only. No DB, real student data or production secrets."""
import json
import os
from pathlib import Path
import time

import llama_cpp
from grader import MODEL_FILE, MODEL_REPO, RUBRIC_VERSION, evaluate, load_model


def main():
    case = os.environ["QA_CASE"]
    [job] = [r for r in json.loads(Path("student_cases.json").read_text()) if r["id"] == case]
    print(f"Loading Gemma 4 12B · {MODEL_REPO}/{MODEL_FILE}", flush=True)
    llm = load_model()
    model = Path(llm.model_path)
    arch = llm.metadata.get("general.architecture", "")
    assert model.name == MODEL_FILE and model.stat().st_size == 7_121_861_440
    assert arch.startswith("gemma"), arch
    model_details = f"{model.name} · architecture: {arch} · llama-cpp-python {llama_cpp.__version__}"
    print(f"Loaded Gemma 4 12B: {model_details}", flush=True)
    started = time.monotonic()
    # Keep native validation fail-closed. A failed grade must fail the job, not result in a green check.
    report = evaluate(llm, job)
    seconds = round(time.monotonic() - started, 1)
    print(f"Gemma inference completed: {seconds}s ({RUBRIC_VERSION})", flush=True)
    result = {"id": case, "variant": "candidate", "rubricVersion": RUBRIC_VERSION,
              "seconds": seconds, "model": model_details, "report": report}
    Path(f"reports-{case}.json").write_text(json.dumps([result], ensure_ascii=False, indent=2))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as s:
            s.write(f"### Gemma 4 12B · {case}\n\nLoaded `{model_details}`. Valid report in {seconds}s.\n")
    score = report["scores"]
    if case.endswith("off-topic"):
        assert all(score[k] == 0 for k in ("relevance", "coverage", "analysis", "evidence")), score
    if case in {"wetlands-strong", "heat-strong-hi"}:
        assert score["relevance"] >= 3
    if case == "wetlands-overclaim":
        assert "80" in (report["gap"] + report["nextStep"]), "Missed the unsupported statistic"


if __name__ == "__main__":
    main()
