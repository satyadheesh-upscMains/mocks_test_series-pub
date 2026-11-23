"""Queue worker: claims submissions from the site, reads photo answers, runs the examiner, returns the report.

One runner handles one submission at a time, end to end, so a student's answer and photos never leave the runner's
memory: no artifacts, no files, nothing in the logs except stage, timing and status. Several runners work in
parallel, each holding its own lease from the site.

Site protocol (POST JSON to MAINS_WORKER_URL with "Authorization: Bearer MAINS_WORKER_SECRET"):
  claim       -> {"job": {id, lease, lang, question, references, marks?, input: "typed"|"photos", answer?, pages?}}
                 pages: base64 JPEGs, at most 2
  renew       {id, lease}                       -> {"ok": true} | {"ignored": true}   (keeps the lease alive)
  transcribe  {id, lease, transcript}           -> {"ok": true, "blocked": bool} | {"ignored": true}
                 transcript: text, pages, lines (id, page, box, text, status), evidence, blocked, retake,
                 retakeMessage. When blocked, the site asks the student for a retake and the job ends here.
  complete    {id, lease, result, reportVersion} -> {"ok": true} | {"ignored": true}
                 result: the student report (mains-report-v2), with page marks for photo answers
  fail        {id, lease}                       -> {"ok": true}   (back to the queue, at most 3 attempts)
  deliver     {}                                -> {"processed": n} (sends the opt-in alerts)
"""
import base64
import binascii
import json
import os
import sys
import time
from urllib import request

from mains import photo, pipeline
from mains.model import Examiner, load_model, log
from mains.report import REPORT_VERSION, Incomplete, attach_page
from mains.text import Rejected

RUN_BUDGET = 5 * 3600     # a GitHub job may run 6 h; stop claiming well before that
JOB_ROOM = 150 * 60       # do not start a submission with less than this left: photos + 15 checks fit comfortably
MAX_PAGES = 2


class ModelUnavailable(RuntimeError):
    """The runner cannot load Gemma. Not the student's fault: the job goes back to the queue and the batch stops."""


class Models:
    """Loads Gemma per runner: with the vision projector to read photos, text-only to grade. Grading on the text-only
    model keeps the shared examiner prompt cached between the 15 checks, so a photo answer grades as fast as a typed
    one (end-to-end run 45116948023: 31 min typed vs 94 min photo while grading ran on the vision-loaded model)."""

    def __init__(self, factory):
        self.factory, self.llm, self.vision = factory, None, False

    def get(self, vision=False):
        if self.llm is None or vision != self.vision:
            self.llm = None
            try:
                self.llm = self.factory(vision=vision)
            except Exception:
                raise ModelUnavailable() from None
            self.vision = vision
        return self.llm


def pages_of(job):
    pages = job.get("pages") or []
    if not 1 <= len(pages) <= MAX_PAGES:
        raise ValueError("pages")
    return [base64.b64decode(p, validate=True) for p in pages]


def handle(call, job, models, reader_factory=photo.GemmaReader):
    key = {"id": job["id"], "lease": job["lease"]}
    transcript = None
    if job.get("input") == "photos":
        reader = reader_factory(models.get(vision=True))
        transcript = photo.transcribe(pages_of(job), job["lang"], reader)
        transcript["retakeMessage"] = photo.retake_message(transcript, job["lang"])
        answer = transcript["text"]
        if call({"action": "transcribe", **key, "transcript": transcript}).get("ignored"):
            return "lost"
        if transcript["blocked"]:
            return "retake"
    else:
        answer = job.get("answer") or ""

    def renew():
        if call({"action": "renew", **key}).get("ignored"):
            raise LeaseLost()

    examiner = Examiner(models.get(vision=False))
    report = pipeline.evaluate(examiner, dict(job, answer=answer), between=renew)
    if transcript is not None:
        attach_page(report, transcript)
    done = call({"action": "complete", **key, "result": report, "reportVersion": REPORT_VERSION})
    return "lost" if done.get("ignored") else "done"


class LeaseLost(RuntimeError):
    """The site took the job back (lease expired, or the student deleted the answer)."""


def process(call, model_factory=load_model, reader_factory=photo.GemmaReader, clock=time.monotonic,
            budget=RUN_BUDGET, room=JOB_ROOM):
    models = Models(model_factory)
    counts = {"done": 0, "retake": 0, "failed": 0, "lost": 0}
    start = clock()
    while clock() - start + room <= budget:
        job = call({"action": "claim"}).get("job")
        if not job:
            break
        key = {"id": job["id"], "lease": job["lease"]}
        began = clock()
        try:
            outcome = handle(call, job, models, reader_factory)
        except ModelUnavailable:
            call({"action": "fail", **key})
            log(stage="job", status="model_unavailable")
            raise
        except LeaseLost:
            outcome = "lost"
        except (Rejected, Incomplete, ValueError, KeyError, binascii.Error):
            call({"action": "fail", **key})
            outcome = "failed"
        counts[outcome] += 1
        log(stage="job", status=outcome, input=job.get("input", "typed"), seconds=round(clock() - began, 1))
    call({"action": "deliver"})
    return counts


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    url = os.environ["MAINS_WORKER_URL"]
    secret = os.environ["MAINS_WORKER_SECRET"]
    if not url.startswith("https://"):
        raise ValueError("Use HTTPS")
    opener = request.build_opener(NoRedirect())

    def call(payload):
        req = request.Request(url, json.dumps(payload, ensure_ascii=False).encode("utf-8"), {
            "Content-Type": "application/json", "Authorization": f"Bearer {secret}"})
        with opener.open(req, timeout=90) as r:
            return json.load(r)

    counts = process(call)
    print("Finished: " + ", ".join(f"{k} {v}" for k, v in counts.items()), flush=True)
    return int(counts["failed"] > 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("Worker could not finish. Check runner and endpoint configuration.", file=sys.stderr)
        sys.exit(1)
