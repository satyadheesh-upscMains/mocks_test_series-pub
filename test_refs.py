"""Unit tests for the Constitution/NCERT reference extracts. No model and no network: a tiny fixture index.

Run: python -m unittest -v test_refs
"""
import contextlib
import io
import unittest
from unittest import mock

from mains import checks, pipeline, refs, text
from mains.model import Examiner
from mains.text import Rejected
from test_examiner import EN, FakeGemma, concern, job

CONST = [
    {"article": "243K", "title": "Elections to the Panchayats", "notes": [],
     "description": "(1) The superintendence, direction and control of the preparation of electoral rolls for, and the "
                    "conduct of, all elections to the Panchayats shall be vested in a State Election Commission "
                    "consisting of a State Election Commissioner to be appointed by the Governor."},
    {"article": "243G", "title": "Powers, authority and responsibilities of Panchayats", "notes": [],
     "description": "The Legislature of a State may, by law, endow the Panchayats with such powers and authority as may "
                    "be necessary to enable them to function as institutions of self-government in relation to the "
                    "matters listed in the Eleventh Schedule."},
    {"article": "21", "title": "Protection of life and personal liberty", "notes": [],
     "description": "No person shall be deprived of his life or personal liberty except according to procedure "
                    "established by law."},
]
PAGES = [{"file": "keps208", "page": 1, "text": (
    "Chapter 8: Local Governments\nThe 73rd amendment made panchayats a constitutional body for rural local "
    "self-government. Every State has a State Election Commission for panchayat elections and a State Finance "
    "Commission to review the finances of local governments. One third of seats in panchayats are reserved for women. "
    "The Gram Sabha comprises all adult members of the village. Elections to village, block and district panchayats "
    "are held every five years.\n208")}]
Q = "Discuss how the 73rd Constitutional Amendment strengthened rural local self-government in India."
A = ("Panchayat elections are conducted by the Election Commission of India. One third of seats in panchayats are "
     "reserved for women.")


def index():
    return refs.Index(refs.constitution_chunks(CONST) + refs.ncert_chunks(PAGES), min_score=0.5, min_coverage=0.2)


def prep_for(question, answer, found):
    return {"id": "t", "lang": "en", "question": question, "references": [], "refs": found,
            "sentences": text.sentences(answer), "parts": []}


class Building(unittest.TestCase):
    def test_labels_name_the_source(self):
        c = refs.constitution_chunks(CONST)
        self.assertEqual(c[0]["label"], "Constitution of India, Article 243K")
        n = refs.ncert_chunks(PAGES)
        self.assertEqual(n[0]["label"], "NCERT Class 11, Indian Constitution at Work, Ch. 8 (Local Governments)")
        self.assertNotIn("Chapter 8:", n[0]["text"])   # running heads and page numbers are dropped
        self.assertFalse(n[0]["text"].endswith("208"))


class Search(unittest.TestCase):
    def test_topic_question_gets_its_chapter_and_articles(self):
        found = index().pack(Q, A)
        labels = [r["label"] for r in found]
        self.assertTrue(any(lb.startswith("NCERT Class 11") for lb in labels))
        self.assertTrue(any("State Election Commission" in r["text"] for r in found))
        self.assertEqual([r["id"] for r in found], [f"r{i}" for i in range(1, len(found) + 1)])
        self.assertLessEqual(len(found), refs.MAX_PASSAGES)

    def test_named_article_comes_first(self):
        found = index().pack("Discuss the scope of Article 21 today.", "")
        self.assertEqual(found[0]["label"], "Constitution of India, Article 21")

    def test_off_topic_question_gets_nothing(self):
        self.assertEqual(index().pack("Discuss how urban wetlands reduce flood risk.", "Wetlands store water."), [])

    def test_missing_index_means_no_sources(self):
        self.assertIsNone(refs.load("/nonexistent/index.json.gz"))


class Citing(unittest.TestCase):
    def setUp(self):
        self.found = index().pack(Q, A)
        self.sec = next(r["id"] for r in self.found if "State Election Commission" in r["text"])
        self.prep = prep_for(Q, A, self.found)

    def raw(self, ref_id):
        c = concern("s1", "wrong", "major",
                    "Panchayat elections are run by the State Election Commission, not the national one.",
                    "Write that the State Election Commission conducts panchayat elections.")
        c["refId"] = ref_id
        return {"credit": [], "concerns": [c], "note": "One factual error about who runs the elections.",
                "verdict": "wrong", "rewrite": {"sourceId": "none", "text": "", "refId": "none"}}

    def test_a_cited_source_allows_its_facts(self):
        out = checks.bind(self.raw(self.sec), self.prep, "accuracy")
        self.assertEqual(out["concerns"][0]["ref"], self.sec)

    def test_naming_a_source_fact_without_citing_it_is_rejected(self):
        with self.assertRaises(Rejected) as e:
            checks.bind(self.raw("none"), self.prep, "accuracy")
        self.assertEqual(e.exception.code, "reason:unsourced_name")

    def test_a_citation_the_advice_does_not_use_is_dropped(self):
        raw = self.raw(self.sec)
        raw["concerns"][0].update(reason="This sentence is stated too briefly to show the reasoning.",
                                  fix="Explain the point in one more sentence.", kind="weak", priority="minor")
        raw["verdict"] = "partial"
        out = checks.bind(raw, self.prep, "accuracy")
        self.assertIsNone(out["concerns"][0]["ref"])   # no source label on advice that does not come from it


ART48A = [{"id": "r1", "label": "Constitution of India, Article 48A",
           "text": "Protection and improvement of environment and safeguarding of forests and wild life. The State "
                   "shall endeavour to protect and improve the environment and to safeguard the forests and wild life "
                   "of the country."}]


class InTheReport(unittest.TestCase):
    def run_all(self, script, found):
        with mock.patch.object(refs, "pack", return_value=found), contextlib.redirect_stdout(io.StringIO()):
            return pipeline.evaluate(Examiner(FakeGemma(script)), job("en"))

    def test_source_backed_point_shows_its_label(self):
        script = dict(EN)
        c = concern("none", "missing", "medium",
                    "The answer does not link conservation to Article 48A, which asks the State to protect the "
                    "environment.", "Add that Article 48A directs the State to protect and improve the environment.")
        c["refId"] = "r1"
        script["provisions"] = {"credit": [], "concerns": [c], "note": "One constitutional point is missing.",
                                "verdict": "partial", "rewrite": {"sourceId": "none", "text": "", "refId": "none"}}
        rep = self.run_all(script, ART48A)
        self.assertEqual(rep["sources"], [{"label": "Constitution of India, Article 48A", "used": True}])
        shown = rep["additions"] + rep["fixes"]
        self.assertTrue(any(x["source"] == "Constitution of India, Article 48A" for x in shown))
        self.assertEqual({d["id"]: d["verdict"] for d in rep["checks"]}["provisions"], "partial")

    def test_no_sources_skips_the_check_without_a_model_call(self):
        gemma = FakeGemma(EN)
        with mock.patch.object(refs, "pack", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            rep = pipeline.evaluate(Examiner(gemma), job("en"))
        self.assertEqual(rep["sources"], [])
        self.assertEqual({d["id"]: d["verdict"] for d in rep["checks"]}["provisions"], "not_applicable")
        self.assertFalse(any("TASK [provisions]" in m[1]["content"] for m in gemma.prompts))
        self.assertNotIn('"sources"', gemma.prompts[0][0]["content"])   # no empty sources block in the prompt


if __name__ == "__main__":
    unittest.main()
