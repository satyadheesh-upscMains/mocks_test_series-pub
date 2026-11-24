"""v10 safety rules, each on a fixture shaped like the v9 split run 45324728302. No model."""
import json
import unittest

from mains import checks, pipeline, refs, report
from mains.model import Examiner
from mains.text import Rejected
# Unit tests never read the real reference index on disk: once it holds the GS textbooks (v16), every answer gets
# sources and checks the scripted fake model has no reply for. Tests that need sources build their own index.
refs._LOADED[refs.INDEX] = None



def concern(sid, kind="wrong", reason="The sentence is wrong.", fix="Correct the sentence.", ref=None,
            priority="major"):
    return {"sourceId": sid, "quote": f"text {sid}", "kind": kind, "priority": priority, "reason": reason,
            "fix": fix, "ref": ref}


def result(cid, concerns=(), credit=(), verdict="partial", kind="check", rewrite=None):
    return {"id": cid, "kind": kind, "status": "ok", "verdict": verdict, "credit": list(credit),
            "concerns": list(concerns), "rewrite": rewrite, "note": "Note."}


GOV = ["The Governor is the constitutional head of a State.",
       "The Governor is elected by the State Legislative Assembly for a fixed term of five years.",
       "The executive power of the State is vested in the Governor and is exercised through the Council of Ministers.",
       "Because the Governor is linked to the Centre, the office has often been at the centre of disputes."]


def prep_of(sentences, refs_=()):
    return {"id": "t", "lang": "en", "question": "Analyse the position of the Governor in India's federal structure.",
            "references": [], "marks": 15,
            "sentences": [{"id": f"s{n}", "text": t, "para": 1} for n, t in enumerate(sentences, 1)],
            "stats": {"words": 60, "sentences": len(sentences), "paragraphs": 1, "limit": 250},
            "parts": [{"id": "p1", "quote": "the position of the Governor in India's federal structure"}],
            "directives": [], "refs": list(refs_)}


A155 = {"id": "r1", "label": "Constitution of India, Article 155", "shared": True, "claims": ["s2"],
        "text": "Appointment of Governor. The Governor of a State shall be appointed by the President by warrant under "
                "his hand and seal."}
A154 = {"id": "r2", "label": "Constitution of India, Article 154", "shared": True, "claims": ["s3"],
        "text": "Executive power of State. The executive power of the State shall be vested in the Governor and shall "
                "be exercised by him either directly or through officers subordinate to him."}


class ClearedStaysCleared(unittest.TestCase):
    def test_provisions_cannot_overrule_a_cleared_claim_via_its_rewrite(self):
        # v9 Governor s3: the claim check found no clash, but "Constitution and textbook basics" marked it wrong
        # against Article 154's wording; its source came through the rewrite, so the old rule did not apply.
        prov = result("provisions", [concern("s3", reason="The extract says officers subordinate to him.")],
                      rewrite={"sourceId": "s3", "text": "The executive power is exercised by officers.", "ref": "r2"})
        facts3 = result("facts:s3", verdict="not_applicable", kind="fact")
        out = pipeline.merge_facts([prov, facts3])
        p = next(r for r in out if r["id"] == "provisions")
        self.assertFalse([c for c in p["concerns"] if c["sourceId"] == "s3"])

    def test_claim_finding_replaces_an_uncited_one_and_keeps_its_article(self):
        # v9: four planted errors were red with no Article, because the uncited finding was kept.
        prov = result("provisions", [concern("s2", reason="The Governor is not elected.")])
        facts2 = result("facts:s2", [concern("s2", reason="Article 155: appointed by the President.", ref="r1")],
                        verdict="wrong", kind="fact")
        out = pipeline.merge_facts([prov, facts2])
        p = next(r for r in out if r["id"] == "provisions")
        s2 = [c for c in p["concerns"] if c["sourceId"] == "s2"]
        self.assertEqual(len(s2), 1)
        self.assertEqual(s2[0]["ref"], "r1")


class WrongMeansWhatItSays(unittest.TestCase):
    def test_an_omission_is_amber_not_red(self):
        mb = ["A Money Bill can be introduced in either House of Parliament.",
              "After the Lok Sabha passes it, the Bill is sent to the Rajya Sabha for its recommendations."]
        prep = prep_of(mb)
        part = result("part:p1", credit=[{"sourceId": "s2", "quote": mb[1], "reason": "Right."}], kind="part")
        sol = result("solutions", [concern("s2", reason="It fails to mention the specific time limit.",
                                           fix="Specify the fourteen days.")])
        rep = report.assemble(prep, [part, sol])
        tone = {h["sourceId"]: h["tone"] for h in rep["highlights"]}
        self.assertEqual(tone["s2"], "amber")

    def test_equivalent_wording_is_not_a_problem(self):
        prep = prep_of(GOV)
        part = result("part:p1", credit=[{"sourceId": "s1", "quote": GOV[0], "reason": "Right."}], kind="part")
        acc = result("accuracy", [concern("s4", kind="weak", priority="minor",
                                          reason="The phrase is slightly less precise than the standard phrasing.")])
        rep = report.assemble(prep, [part, acc])
        self.assertFalse(any(c for h in rep["highlights"] for c in h["concerns"]))


class OneCorrectionOnce(unittest.TestCase):
    def test_two_fixes_on_the_same_number_and_source_are_one(self):
        a = {"fix": "Change the time limit from three months to fourteen days.", "source": "Constitution of India, Article 109"}
        b = {"fix": "Specify that the bill must be returned within fourteen days.", "source": "Constitution of India, Article 109"}
        self.assertTrue(report.repeats(b, [a]))
        self.assertFalse(report.repeats(dict(b, source=None), [dict(a, source=None)]))


class ScriptedExaminer:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def ask(self, *, validate, attempts=3, **_):
        last = None
        for _ in range(attempts):
            if not self.replies:
                break
            try:
                return validate(self.replies.pop(0))
            except Rejected as e:
                last = e
        raise last or Rejected("failed")


class CorrectionsTeachNoNewError(unittest.TestCase):
    def setUp(self):
        self.prep = prep_of(GOV, [A155])
        self.rep = {"fixes": [{"n": 1, "kind": "wrong", "title": "State the appointment.", "where": "The question",
                               "why": "Appointed, not elected.", "tryWriting": None,
                               "source": "Constitution of India, Article 155",
                               "yourLine": {"sourceId": "s2", "text": GOV[1]}}]}

    def test_a_correction_from_the_extract_is_kept(self):
        ex = ScriptedExaminer([{"sentence": "The Governor of a State is appointed by the President by warrant."}])
        report.correct(ex, self.prep, "sys", self.rep)
        self.assertIn("appointed by the President", self.rep["fixes"][0]["tryWriting"])

    def test_a_correction_with_a_new_number_or_name_is_dropped(self):
        ex = ScriptedExaminer([{"sentence": "The Governor is appointed by the President for six months at a time."},
                               {"sentence": "The Governor is appointed by the Prime Minister of India."}])
        report.correct(ex, self.prep, "sys", self.rep)
        self.assertIsNone(self.rep["fixes"][0]["tryWriting"])

    def test_an_unsourced_rewrite_of_a_wrong_fact_is_not_shown(self):
        # v9 Article 356: "remains in force until the situation is resolved", a new error with no source.
        prep = prep_of(GOV)
        part = result("part:p1", credit=[{"sourceId": "s1", "quote": GOV[0], "reason": "Right."}], kind="part")
        acc = result("accuracy", [concern("s2", reason="The Governor is not elected by the Assembly.")],
                     rewrite={"sourceId": "s2", "text": "The Governor is chosen for life.", "ref": None})
        rep = report.assemble(prep, [part, acc])
        self.assertIsNone(rep["fixes"][0]["tryWriting"])


class LicensedText(unittest.TestCase):
    def test_long_copying_is_detected(self):
        passage = " ".join(f"w{i}" for i in range(40))
        self.assertTrue(checks.copies(" ".join(f"w{i}" for i in range(5, 18)), passage))
        self.assertFalse(checks.copies(" ".join(f"w{i}" for i in range(5, 15)), passage))
        self.assertTrue(checks.licensed({"label": "Laxmikanth, Indian Polity, Ch. 30 (Governor)"}))
        self.assertFalse(checks.licensed({"label": "Constitution of India, Article 155"}))


class UnprovenClaims(unittest.TestCase):
    def test_contradicted_without_the_words_is_not_covered(self):
        prep = prep_of(GOV, [A155])
        bad = {"sourceWords": "nothing like this", "wrongWords": "not in the claim", "credit": [],
               "concerns": [{"sourceId": "s2", "kind": "wrong", "priority": "major", "reason": "Wrong body.",
                             "fix": "Say appointed by the President.", "refId": "r1"}],
               "note": "Wrong.", "verdict": "contradicted", "rewrite": {"sourceId": "none", "text": "", "refId": "none"}}

        class Fake:
            def create_chat_completion(self, **_):
                return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(bad)}}], "usage": {}}

        out = checks.run_fact(Examiner(Fake()), prep, "sys", "facts:s2")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["verdict"], "not_applicable")
        self.assertFalse(out["concerns"])


class CasesAndBook(unittest.TestCase):
    def test_a_governor_question_gets_governor_cases_and_cases_never_check_claims(self):
        chunks = refs.constitution_chunks(json.load(open(refs.Path(__file__).resolve().parent /
                                                         "refs/constitution_of_india_2026.json", encoding="utf-8")))
        chunks += refs.case_chunks(json.load(open(refs.Path(__file__).resolve().parent / "refs/cases.json",
                                                  encoding="utf-8")))
        idx = refs.Index(chunks)
        sents = [{"id": f"s{n}", "text": t} for n, t in enumerate(GOV, 1)]
        got = idx.pack("Analyse the position of the Governor in India's federal structure.", sents)
        cases = [r for r in got if r["label"].startswith("Case:")]
        self.assertTrue(cases)
        self.assertLessEqual(len(cases), refs.MAX_CASES)
        self.assertTrue(all(r["shared"] and not r.get("claims") for r in cases))

    def test_off_topic_question_gets_no_cases(self):
        chunks = refs.case_chunks(json.load(open(refs.Path(__file__).resolve().parent / "refs/cases.json",
                                                 encoding="utf-8")))
        chunks += refs.constitution_chunks(json.load(open(refs.Path(__file__).resolve().parent /
                                                          "refs/constitution_of_india_2026.json", encoding="utf-8")))
        idx = refs.Index(chunks)
        got = idx.pack("Discuss how urban wetlands reduce flood risk and suggest ways to conserve them.",
                       "Urban wetlands hold stormwater after heavy rain.")
        self.assertEqual(got, [])

    def test_a_textbook_only_question_still_gets_its_articles_from_the_claims(self):
        # v10 retrieval check: the 73rd Amendment question matched only NCERT, so no Part IX Article checked a claim.
        root = refs.Path(__file__).resolve().parent
        chunks = refs.constitution_chunks(json.load(open(root / "refs/constitution_of_india_2026.json",
                                                         encoding="utf-8")))
        chunks.append({"id": "n108.1", "label": "NCERT Class 11, Indian Constitution at Work, Ch. 8 (Local Governments)",
                       "title": "Local Governments", "text": "The 73rd Amendment strengthened rural local "
                       "self-government through panchayats in villages, blocks and districts. " * 4})
        idx = refs.Index(chunks)
        ans = ["The 73rd Amendment of 1992 gave constitutional status to panchayats and created a three-tier system "
               "of village, block and district panchayats.",
               "It made elections every five years compulsory, and these elections are conducted by the Election "
               "Commission of India.", "One-third of the seats are reserved for women in panchayats.",
               "Panchayats now control all the subjects in the State List."]
        got = idx.pack("Discuss how the 73rd Constitutional Amendment strengthened rural local self-government in "
                       "India. What challenges remain?", [{"id": f"s{n}", "text": t} for n, t in enumerate(ans, 1)])
        arts = [r["label"] for r in got if r["label"].startswith("Constitution of India, Article 243")]
        self.assertTrue(arts)

    def test_book_labels_name_chapter_and_section(self):
        c = refs.laxmikanth_chunks([{"chapter": 30, "title": "Governor", "section": "Appointment of Governor",
                                     "page": 677, "text": "words " * 40}])[0]
        self.assertEqual(c["label"], "Laxmikanth, Indian Polity, Ch. 30 (Governor), Appointment of Governor")
        self.assertTrue(c["private"])


if __name__ == "__main__":
    unittest.main()
