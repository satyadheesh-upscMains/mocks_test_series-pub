"""Unit tests for the Constitution/NCERT reference extracts. No model and no network: a tiny fixture index.

Run: python -m unittest -v test_refs
"""
import contextlib
import io
import unittest
from unittest import mock

from mains import checks, pipeline, prompts, refs, text
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
    {"article": "324", "title": "Superintendence, direction and control of elections to be vested in an Election "
                               "Commission", "notes": [],
     "description": "The superintendence, direction and control of the preparation of the electoral rolls for, and the "
                    "conduct of, all elections to Parliament and to the Legislature of every State shall be vested in "
                    "a Commission (referred to in this Constitution as the Election Commission)."},
    {"article": "SCH6", "title": "Sixth Schedule", "notes": [],
     "description": "Elections to the District Councils of the autonomous districts shall be conducted by the "
                    "Election Commission, which shall hold such elections every five years."},
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
    def test_embedding_ranker_finds_governor_paraphrase(self):
        class FakeEmbedder:
            def embed(self, values):
                # e5 prefixes are part of the production contract.
                return [[1.0, 0.0] if "removed" in v or "term" in v else [0.0, 1.0] for v in values]
        chunks = refs.constitution_chunks(CONST + [{"article": "156", "title": "Term of office of Governor",
            "notes": [], "description": "The Governor shall hold office during the pleasure of the President."}])
        idx = refs.Index(chunks, min_score=0, min_coverage=0, embedder=FakeEmbedder())
        labels = [idx.chunks[i]["label"] for _, _, i in idx.search("A Governor cannot be removed before that")]
        self.assertIn("Constitution of India, Article 156", labels[:2])
    def test_topic_question_gets_its_chapter_and_articles(self):
        found = index().pack(Q, A)
        labels = [r["label"] for r in found]
        self.assertTrue(any(lb.startswith("NCERT Class 11") for lb in labels))
        self.assertTrue(any("State Election Commission" in r["text"] for r in found))
        self.assertEqual([r["id"] for r in found], [f"r{i}" for i in range(1, len(found) + 1)])
        self.assertLessEqual(len(found), refs.MAX_PASSAGES)

    def test_each_claim_gets_its_passage_within_the_topic(self):
        # Constitution only, as on the runners: the panchayat election claim must find Article 243K, not 324.
        idx = refs.Index(refs.constitution_chunks(CONST), min_score=0.5, min_coverage=0.2)
        found = idx.pack(Q + " Panchayats", [{"id": "s1", "text": "These elections are conducted by the Election "
                                                               "Commission of India."}])
        claimed = [r["label"] for r in found if "s1" in r.get("claims", [])]
        self.assertIn("Constitution of India, Article 243K", claimed)
        self.assertNotIn("Constitution of India, Article 324", claimed)
        self.assertNotIn("Constitution of India, Schedule 6", claimed)   # matches the claim's words, not the topic
        self.assertEqual(refs.family({"id": "c243K"}), "IX")              # Part IX: Panchayats
        self.assertEqual(refs.family({"id": "c243ZA"}), "IXA")            # Part IXA: Municipalities
        self.assertEqual(refs.family({"id": "c156"}), "VI.2")             # the Governor's chapter
        self.assertEqual(refs.family({"id": "cSCH11"}), "SCH11")
        self.assertEqual(refs.family({"id": "n208.4"}), "n208")

    def test_claim_only_extracts_stay_out_of_the_shared_prompt(self):
        found = index().pack(Q, [{"id": "s1", "text": A.split(". ")[0] + "."}])
        self.assertTrue(all(r["shared"] for r in found[:sum(r["shared"] for r in found)]))   # shared come first
        self.assertLessEqual(sum(r["shared"] for r in found), refs.MAX_PASSAGES)
        self.assertLessEqual(len(found), refs.MAX_REFS)

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

    def test_naming_a_source_fact_without_citing_it_is_dropped(self):
        out = checks.bind(self.raw("none"), self.prep, "accuracy")
        self.assertEqual(out["concerns"], [])

    def test_credit_naming_an_unknown_body_is_dropped_not_retried(self):
        raw = {"credit": [{"sourceId": "s2", "reason": "Women's reservation is right, as the Kothari Committee said."},
                          {"sourceId": "s2", "reason": "dup"}][:1],
               "concerns": [], "note": "Mostly accurate.", "verdict": "good",
               "rewrite": {"sourceId": "none", "text": "", "refId": "none"}}
        with self.assertRaises(Rejected):   # nothing left to back "good"
            checks.bind(raw, self.prep, "accuracy")
        raw["verdict"] = "not_applicable"
        self.assertEqual(checks.bind(raw, self.prep, "accuracy")["credit"], [])

    def test_a_citation_the_advice_does_not_use_is_dropped(self):
        raw = self.raw(self.sec)
        raw["concerns"][0].update(reason="This sentence is stated too briefly to show the reasoning.",
                                  fix="Explain the point in one more sentence.", kind="weak", priority="minor")
        raw["verdict"] = "partial"
        out = checks.bind(raw, self.prep, "accuracy")
        self.assertIsNone(out["concerns"][0]["ref"])   # no source label on advice that does not come from it


NOTHING = {"credit": [], "concerns": [], "note": "Nothing to add.", "verdict": "not_applicable",
           "rewrite": {"sourceId": "none", "text": "", "refId": "none"}}
ART48A = [{"id": "r1", "label": "Constitution of India, Article 48A",
           "text": "Protection and improvement of environment and safeguarding of forests and wild life. The State "
                   "shall endeavour to protect and improve the environment and to safeguard the forests and wild life "
                   "of the country."}]


class InTheReport(unittest.TestCase):
    def run_all(self, script, found):
        script = dict(script)
        script.setdefault("correct", {"sentence": "Panchayat elections are conducted by a State Election Commission."})
        script.setdefault("third", {"difference": "A State Election Commission, not the Election Commission of India.",
                                    "verdict": "different"})
        with mock.patch.object(refs, "pack", return_value=found), contextlib.redirect_stdout(io.StringIO()):
            return pipeline.evaluate(Examiner(FakeGemma(script)), job("en"))

    def test_source_backed_point_shows_its_label(self):
        script = dict(EN, gap=NOTHING)
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

    def test_a_wrong_claim_is_marked_on_its_own_sentence(self):
        # The claim check finds the error the combined check missed; it reaches the report as a red, cited fix.
        found = [dict(ART48A[0], shared=True),
                 {"id": "r2", "label": "Constitution of India, Article 243K", "shared": False, "claims": ["s4"],
                  "text": "Elections to the Panchayats. The conduct of all elections to the Panchayats shall be vested "
                          "in a State Election Commission."}]
        wrong = concern("s4", "wrong", "major",
                        "Article 243K vests panchayat elections in a State Election Commission, not this body.",
                        "Write that a State Election Commission conducts panchayat elections.")
        wrong["refId"] = "r2"
        script = dict(EN, gap=NOTHING, provisions=NOTHING)
        script["facts:s4"] = {"sourceWords": "vested in a State Election Commission", "wrongWords": "community monitoring",
                              "credit": [], "concerns": [wrong], "note": "The claim is contradicted.",
                              "verdict": "contradicted", "rewrite": {"sourceId": "none", "text": "", "refId": "none"}}
        rep = self.run_all(script, found)
        s4 = next(h for h in rep["highlights"] if h["sourceId"] == "s4")
        self.assertEqual(s4["tone"], "red")
        self.assertTrue(any(c["source"] == "Constitution of India, Article 243K" for c in s4["concerns"]))
        self.assertEqual({d["id"]: d["verdict"] for d in rep["checks"]}["provisions"], "wrong")
        self.assertIn({"label": "Constitution of India, Article 243K", "used": True}, rep["sources"])

    def test_a_claim_flagged_without_its_extract_is_dropped(self):
        found = [dict(ART48A[0], shared=True),
                 {"id": "r2", "label": "Constitution of India, Article 243K", "shared": False, "claims": ["s4"],
                  "text": "Elections to the Panchayats. A State Election Commission conducts them."}]
        loose = concern("s4", "wrong", "major", "This sentence is not right as written.", "Rewrite the sentence.")
        loose["refId"] = "none"
        script = dict(EN, gap=NOTHING, provisions=NOTHING)
        script["facts:s4"] = {"credit": [], "concerns": [loose], "note": "Contradicted.", "verdict": "contradicted",
                              "rewrite": {"sourceId": "none", "text": "", "refId": "none"}}
        base = self.run_all(dict(EN, gap=NOTHING, provisions=NOTHING, **{"facts:s4": NOTHING}), found)
        rep = self.run_all(script, found)
        tone = lambda r: {h["sourceId"]: h["tone"] for h in r["highlights"]}.get("s4")
        self.assertEqual(tone(rep), tone(base))   # no evidence from an extract: the claim check adds nothing

    def test_a_clash_not_in_the_claims_own_words_is_not_called_wrong(self):
        # Split run 45252594185: correct sentences were marked red with another sentence's error. A contradiction
        # must quote the wrong words from this claim and the clashing words from its extract.
        found = [dict(ART48A[0], shared=True),
                 {"id": "r2", "label": "Constitution of India, Article 243K", "shared": False, "claims": ["s1"],
                  "text": "Elections to the Panchayats. The conduct of all elections to the Panchayats shall be vested "
                          "in a State Election Commission."}]
        wrong = concern("s1", "wrong", "major", "Article 243K vests panchayat elections in a State Election Commission.",
                        "Write that a State Election Commission conducts panchayat elections.")
        wrong["refId"] = "r2"
        script = dict(EN, gap=NOTHING, provisions=NOTHING)
        script["facts:s1"] = {"sourceWords": "vested in a State Election Commission",
                              "wrongWords": "the Election Commission of India",   # not in s1
                              "credit": [], "concerns": [wrong], "note": "Contradicted.", "verdict": "contradicted",
                              "rewrite": {"sourceId": "none", "text": "", "refId": "none"}}
        base = self.run_all(dict(EN, gap=NOTHING, provisions=NOTHING, **{"facts:s1": NOTHING}), found)
        rep = self.run_all(script, found)
        tone = lambda r: {h["sourceId"]: h["tone"] for h in r["highlights"]}.get("s1")
        self.assertEqual(tone(rep), tone(base))

    def test_a_cleared_claim_is_not_called_wrong_by_another_check(self):
        found = [dict(ART48A[0], shared=True, claims=["s1"])]
        loud = concern("s1", "wrong", "major",
                       "Article 48A asks the State to protect the environment, which this sentence ignores.",
                       "Add that Article 48A directs the State to protect the environment.")
        loud["refId"] = "r1"
        provisions = {"credit": [], "concerns": [loud], "note": "One error.", "verdict": "wrong",
                      "rewrite": {"sourceId": "none", "text": "", "refId": "none"}}
        cleared = dict(NOTHING, sourceWords="", wrongWords="", verdict="not_covered")
        rep = self.run_all(dict(EN, gap=NOTHING, provisions=provisions, **{"facts:s1": cleared}), found)
        s1 = next(h for h in rep["highlights"] if h["sourceId"] == "s1")
        self.assertFalse(any(c["kind"] == "wrong" for c in s1["concerns"]))   # the claim check read it and cleared it

    def test_a_failed_question_part_gets_one_more_try(self):
        gemma = FakeGemma(dict(EN, gap=NOTHING))
        ex = Examiner(gemma)
        with mock.patch.object(refs, "pack", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            prep = pipeline.prepare(ex, job("en"))
            results = [r for shard in pipeline.SHARD_ORDER for r in pipeline.run_shard(ex, prep, shard)]
            results = [dict(r, status="unavailable", reason="runner") if r["id"] == "part:p2" else r for r in results]
            rep = pipeline.finish(ex, prep, results)
        self.assertEqual([p["status"] for p in rep["parts"]], ["met", "partial"])

    def test_no_sources_skips_the_check_without_a_model_call(self):
        gemma = FakeGemma(EN)
        with mock.patch.object(refs, "pack", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            rep = pipeline.evaluate(Examiner(gemma), job("en"))
        self.assertEqual(rep["sources"], [])
        self.assertEqual({d["id"]: d["verdict"] for d in rep["checks"]}["provisions"], "not_applicable")
        self.assertFalse(any("TASK [provisions]" in m[1]["content"] for m in gemma.prompts))
        self.assertNotIn('"sources"', gemma.prompts[0][0]["content"])   # no empty sources block in the prompt

GS_PAGES = [{"file": "kegy203", "page": 1, "text": (
    "Chapter 3: Drainage System\nThe Himalayan rivers are perennial because they are fed by glaciers and rainfall. "
    "The Peninsular rivers are seasonal and depend on the monsoon. The Ganga basin covers about 26 per cent of the "
    "country's area. Rivers of the Peninsular plateau flow mostly eastwards into the Bay of Bengal.\n21")}]
GS_MANIFEST = {"kegy2": {"title": "India Physical Environment", "class": 11,
                         "chapters": [{"file": "kegy203", "n": 3, "title": "Drainage System"}]}}
GQ = "Explain why the Himalayan rivers are perennial while the Peninsular rivers are seasonal."
GA = "Himalayan rivers are fed by melting snow and rain. Peninsular rivers dry up in summer."


def gs_index():
    return refs.Index(refs.constitution_chunks(CONST) + refs.ncert_chunks(PAGES)
                      + refs.subject_chunks(GS_PAGES, GS_MANIFEST), min_score=0.5, min_coverage=0.2)


class Background(unittest.TestCase):
    """v16 GS textbooks: support credit and suggestions, never a red mark, never a figure."""

    def test_label_and_flag(self):
        c = refs.subject_chunks(GS_PAGES, GS_MANIFEST)[0]
        self.assertEqual(c["label"], "NCERT Class 11, India Physical Environment, Ch. 3 (Drainage System)")
        self.assertTrue(c["subject"])

    def test_geography_question_gets_its_chapter_as_background_only(self):
        found = gs_index().pack(GQ, GA)
        gs = [r for r in found if r.get("background")]
        self.assertEqual(len(gs), 1)
        self.assertTrue(gs[0]["shared"])
        self.assertNotIn("claims", gs[0])   # never checks a claim

    def test_polity_extracts_do_not_change(self):
        self.assertEqual([r["label"] for r in index().pack(Q, A)],
                         [r["label"] for r in gs_index().pack(Q, A) if not r.get("background")])

    def prep(self):
        found = gs_index().pack(GQ, GA)
        return prep_for(GQ, GA, found), next(r["id"] for r in found if r.get("background"))

    def raw(self, ref_id, kind="wrong", reason=None, fix=None):
        c = concern("s2", kind, "major", reason or "Peninsular rivers depend on the monsoon, as the textbook explains.",
                    fix or "Say that Peninsular rivers depend on the monsoon rains.")
        c["refId"] = ref_id
        return {"credit": [], "concerns": [c], "note": "One point to sharpen.", "verdict": "wrong",
                "rewrite": {"sourceId": "none", "text": "", "refId": "none"}}

    def test_a_background_source_never_marks_a_sentence_wrong(self):
        prep, gid = self.prep()
        out = checks.bind(self.raw(gid), prep, "accuracy")
        self.assertEqual(out["concerns"][0]["kind"], "weak")
        self.assertEqual(out["concerns"][0]["priority"], "medium")

    def test_a_textbook_figure_is_never_quoted(self):
        prep, gid = self.prep()
        raw = self.raw(gid, "missing", "The Ganga basin covers about 26 per cent of the country's area.",
                       "Add that the Ganga basin covers 26 per cent of India.")
        raw["concerns"][0]["sourceId"] = "none"
        raw["verdict"] = "partial"
        out = checks.bind(raw, prep, "dimensions")
        self.assertEqual(out["concerns"], [])   # dropped, like an unsourced number

    def test_reference_books_are_background_one_passage_per_chapter(self):
        books = [{"book": "G.C. Leong, Certificate Physical and Human Geography", "chapter": 16,
                  "title": "The Tropical Monsoon and Tropical Marine Climate", "section": None, "page": 171,
                  "text": "Himalayan rivers are perennial because glaciers feed them through the dry season and "
                          "the monsoon adds to them in summer, while Peninsular rivers are seasonal."}] * 2
        chunks = refs.book_chunks(books)
        self.assertEqual(chunks[0]["label"],
                         "G.C. Leong, Certificate Physical and Human Geography, Ch. 16 "
                         "(The Tropical Monsoon and Tropical Marine Climate)")
        idx = refs.Index(refs.constitution_chunks(CONST) + chunks, min_score=0.5, min_coverage=0.2)
        found = [r for r in idx.pack(GQ, GA) if r.get("background")]
        self.assertEqual(len(found), 1)          # two passages of one chapter: one is shown
        self.assertTrue(checks.licensed(found[0]))  # findings must paraphrase it

    def test_the_prompt_marks_background_sources(self):
        prep, gid = self.prep()
        prep["stats"] = {"words": 17, "limit": 150}
        self.assertIn('"use": "background"', prompts.materials(prep))



if __name__ == "__main__":
    unittest.main()
