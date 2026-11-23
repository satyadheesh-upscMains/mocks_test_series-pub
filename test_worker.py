"""Unit tests for the queue worker. No model and no network: a fake site and scripted Gemma stand-ins.

Run: python -m unittest -v test_worker
"""
import base64
import contextlib
import io
import unittest

import worker
from mains.report import REPORT_VERSION
from test_examiner import A_EN, EN, FakeGemma, Q_EN
from test_photo import TRUTH, Script, page, same


class Site:
    """The site's worker API in memory: hands out queued jobs and records every call."""

    def __init__(self, jobs, ignore=()):
        self.jobs, self.calls, self.ignore = list(jobs), [], set(ignore)

    def __call__(self, p):
        self.calls.append(p)
        a = p["action"]
        if a == "claim":
            return {"job": self.jobs.pop(0) if self.jobs else None}
        if a in self.ignore:
            return {"ignored": True}
        if a == "transcribe":
            return {"ok": True, "blocked": p["transcript"]["blocked"]}
        if a == "deliver":
            return {"processed": 0}
        return {"ok": True}

    def actions(self):
        return [c["action"] for c in self.calls]


def typed(i="t1"):
    return {"id": i, "lease": "L" + i, "lang": "en", "question": Q_EN, "references": [], "marks": 10,
            "input": "typed", "answer": A_EN}


PHOTO_LINES = A_EN.replace(". ", ".\n").split("\n")   # one sentence per handwritten line


def photo_job(i="p1", smudge=None):
    raw = page(lines=PHOTO_LINES, para_after=99, smudge=smudge)
    return {"id": i, "lease": "L" + i, "lang": "en", "question": Q_EN, "references": [], "marks": 10,
            "input": "photos", "pages": [base64.b64encode(raw).decode()]}


def run(site, reads=None):
    models = lambda vision=False: FakeGemma(EN)
    reader = lambda llm: Script(reads or same(lines=PHOTO_LINES))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        counts = worker.process(site, model_factory=models, reader_factory=reader)
    return counts, out.getvalue()


class Worker(unittest.TestCase):
    def test_typed_answer_gets_a_report(self):
        site = Site([typed()])
        counts, logs = run(site)
        self.assertEqual(counts["done"], 1)
        a = site.actions()
        self.assertEqual((a[0], a[-2], a[-1]), ("claim", "claim", "deliver"))
        self.assertIn("renew", a)
        done = next(c for c in site.calls if c["action"] == "complete")
        self.assertEqual(done["reportVersion"], REPORT_VERSION)
        self.assertEqual([p["status"] for p in done["result"]["parts"]], ["met", "partial"])
        self.assertNotIn("stormwater", logs)          # logs never carry the answer

    def test_photo_answer_is_read_then_graded_with_page_marks(self):
        site = Site([photo_job()])
        counts, logs = run(site)
        self.assertEqual(counts["done"], 1)
        a = site.actions()
        self.assertLess(a.index("transcribe"), a.index("complete"))
        tr = next(c for c in site.calls if c["action"] == "transcribe")["transcript"]
        self.assertFalse(tr["blocked"])
        rep = next(c for c in site.calls if c["action"] == "complete")["result"]
        self.assertEqual(rep["input"], "photos")
        marked = {h["sourceId"]: h["lines"] for h in rep["highlights"]}
        self.assertTrue(marked["s2"] and all("box" in x for x in marked["s2"]))   # the 80% sentence, on the page
        self.assertEqual(len(rep["page"]["pages"]), 1)
        self.assertNotIn("stormwater", logs)

    def test_photo_is_read_with_vision_and_graded_text_only(self):
        loads = []

        def models(vision=False):
            loads.append(vision)
            return FakeGemma(EN)

        site = Site([photo_job(), typed("t2")])
        with contextlib.redirect_stdout(io.StringIO()):
            counts = worker.process(site, model_factory=models,
                                    reader_factory=lambda llm: Script(same(lines=PHOTO_LINES)))
        self.assertEqual(counts["done"], 2)
        self.assertEqual(loads, [True, False])   # vision to read, text-only to grade, reused for the typed answer

    def test_vision_model_is_freed_before_the_text_model_loads(self):
        events = []

        class Tracked(FakeGemma):
            def close(self):
                events.append("close")

        def models(vision=False):
            events.append("load-vision" if vision else "load-text")
            return Tracked(EN)

        site = Site([photo_job()])
        with contextlib.redirect_stdout(io.StringIO()):
            counts = worker.process(site, model_factory=models,
                                    reader_factory=lambda llm: Script(same(lines=PHOTO_LINES)))
        self.assertEqual(counts["done"], 1)
        self.assertEqual(events[:3], ["load-vision", "close", "load-text"])   # never two 12B copies at once

    def test_unreadable_photo_asks_for_a_retake_and_is_not_graded(self):
        reads = same(lines=PHOTO_LINES)
        reads[2] = [None] * 6
        site = Site([photo_job()])
        counts, _ = run(site, reads)
        self.assertEqual(counts["retake"], 1)
        self.assertNotIn("complete", site.actions())
        tr = next(c for c in site.calls if c["action"] == "transcribe")["transcript"]
        self.assertTrue(tr["blocked"])
        self.assertEqual(tr["retakeMessage"][0]["lines"], [3])

    def test_bad_photo_data_goes_back_to_the_queue(self):
        job = dict(photo_job(), pages=["not base64!"])
        site = Site([job, typed("t2")])
        counts, _ = run(site)
        self.assertEqual((counts["failed"], counts["done"]), (1, 1))
        self.assertEqual(site.calls[1]["action"], "fail")

    def test_lost_lease_stops_that_job_cleanly(self):
        site = Site([typed()], ignore={"renew"})
        counts, _ = run(site)
        self.assertEqual(counts["lost"], 1)
        self.assertNotIn("complete", site.actions())

    def test_model_failure_returns_the_job_and_stops(self):
        site = Site([typed()])

        def broken(vision=False):
            raise OSError("no model")

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(worker.ModelUnavailable):
            worker.process(site, model_factory=broken)
        self.assertEqual(site.actions(), ["claim", "fail"])

    def test_stops_claiming_when_time_is_short(self):
        site = Site([typed("a"), typed("b")])
        t = iter([0, 0, 0, 4000, 4000, 4000])
        with contextlib.redirect_stdout(io.StringIO()):
            counts = worker.process(site, model_factory=lambda vision=False: FakeGemma(EN),
                                    clock=lambda: next(t, 4000), budget=5000, room=1500)
        self.assertEqual(counts["done"], 1)
        self.assertEqual(len(site.jobs), 1)


if __name__ == "__main__":
    unittest.main()
