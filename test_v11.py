"""v11 rules, each on a fixture from the v10 split run 45344215712. No model: a scripted stand-in plays Gemma."""
import json
import unittest
from pathlib import Path

from mains import checks, pipeline, refs, report
from mains.model import Examiner
from mains.text import Rejected, sentences
# Unit tests never read the real reference index on disk: once it holds the GS textbooks (v16), every answer gets
# sources and checks the scripted fake model has no reply for. Tests that need sources build their own index.
refs._LOADED[refs.INDEX] = None


ROOT = Path(__file__).resolve().parent
CONST = json.loads((ROOT / "refs/constitution_of_india_2026.json").read_text(encoding="utf-8"))
CASES = json.loads((ROOT / "refs/cases.json").read_text(encoding="utf-8"))
IDX = refs.Index(refs.constitution_chunks(CONST) + refs.case_chunks(CASES))

ANSWERS = {
    "art21": ("Examine how the Supreme Court has widened the scope of Article 21 of the Constitution.",
              "Article 21 says that no one shall be deprived of life or personal liberty except according to procedure "
              "established by law. This protection is available only to citizens of India, not to foreigners. Over the "
              "years the Supreme Court has read rights such as livelihood, a clean environment and privacy into the right "
              "to life. A citizen whose Article 21 right is violated can approach the High Court directly under Article 32. "
              "This expansion has made Article 21 the heart of the Fundamental Rights."),
    "governor": ("Analyse the position of the Governor in India's federal structure.",
                 "The Governor is the constitutional head of a State. The Governor is elected by the State Legislative "
                 "Assembly for a fixed term of five years and cannot be removed before that. The executive power of the "
                 "State is vested in the Governor and is exercised through the Council of Ministers. Because the Governor "
                 "is linked to the Centre, the office has often been at the centre of Centre-State disputes."),
    "moneybill": ("Discuss the role of the Rajya Sabha in the passage of Money Bills.",
                  "A Money Bill can be introduced in either House of Parliament. After the Lok Sabha passes it, the Bill is "
                  "sent to the Rajya Sabha for its recommendations. The Rajya Sabha can keep a Money Bill for up to three "
                  "months before returning it. The Lok Sabha may accept or reject the Rajya Sabha's recommendations. This "
                  "shows the financial supremacy of the directly elected House."),
    "356": ("Examine the use of Article 356 and its impact on Indian federalism.",
            "Article 356 lets the President take over the government of a State when its government cannot be carried on "
            "in accordance with the Constitution. The proclamation must be approved by both Houses of Parliament. Once "
            "approved, President's Rule can continue indefinitely until the Centre decides to end it. Its frequent use "
            "in the past weakened the autonomy of States."),
}


def prep_for(name):
    q, a = ANSWERS[name]
    sents = [dict(s, para=1) for s in sentences(a)]
    return {"id": name, "lang": "en", "question": q, "references": [], "marks": 15, "sentences": sents,
            "stats": {"words": 80, "sentences": len(sents), "paragraphs": 1, "limit": 250},
            "parts": [{"id": "p1", "quote": q}], "directives": [], "refs": IDX.pack(q, sents)}


def sid_of(prep, phrase):
    return next(s["id"] for s in prep["sentences"] if phrase in s["text"])


def claim_refs(prep, sid):
    return [r for r in prep["refs"] if sid in r.get("claims", [])]


class Script:
    def __init__(self, *replies):
        self.replies = list(replies)

    def create_chat_completion(self, messages, **_):
        value = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}], "usage": {}}


def contradicted(sid, ref_id, source_words, wrong_words, both="no"):
    return {"bothTrue": both, "sourceWords": source_words, "wrongWords": wrong_words, "credit": [],
            "concerns": [{"sourceId": sid, "kind": "wrong", "priority": "major", "refId": ref_id,
                          "reason": "The extract says something different.", "fix": "Correct the sentence."}],
            "note": "Wrong.", "verdict": "contradicted", "rewrite": {"sourceId": "none", "text": "", "refId": "none"}}


def fact(prep, sid, reply):
    return checks.run_fact(Examiner(Script(reply)), prep, "sys", f"facts:{sid}")


def label_of(prep, ref_id):
    return next(r["label"] for r in prep["refs"] if r["id"] == ref_id)


class ClaimCheck(unittest.TestCase):
    def test_a_found_error_keeps_its_article_when_the_answer_names_it(self):
        # v10: "Supreme Court" and "Article 32" were already in the question and answer, so the Article was dropped.
        prep = prep_for("art21")
        sid = sid_of(prep, "High Court directly under Article 32")
        a32 = next(r for r in claim_refs(prep, sid) if r["label"].endswith("Article 32"))
        out = fact(prep, sid, contradicted(sid, a32["id"], "The right to move the Supreme Court",
                                           "approach the High Court directly"))
        self.assertEqual(out["verdict"], "wrong")
        self.assertEqual(label_of(prep, out["concerns"][0]["ref"]), "Constitution of India, Article 32")

    def test_both_true_is_not_a_contradiction(self):
        # v10 Governor: "exercised through the Council of Ministers" marked red against Article 154's wording.
        prep = prep_for("governor")
        sid = sid_of(prep, "Council of Ministers")
        a154 = next(r for r in claim_refs(prep, sid) if r["label"].endswith("Article 154"))
        out = fact(prep, sid, contradicted(sid, a154["id"], "either directly or through officers subordinate to him",
                                           "through the Council of Ministers", both="yes"))
        self.assertEqual(out["verdict"], "not_applicable")
        self.assertEqual(out["concerns"], [])

    def test_a_trimmed_quote_counts_and_cites_the_extract_it_comes_from(self):
        # v10 Governor: the quote was rejected twice and the planted error went unmarked.
        prep = prep_for("governor")
        sid = sid_of(prep, "elected by the State Legislative Assembly")
        out = fact(prep, sid, contradicted(sid, "none", "Governor shall be appointed by President",
                                           "elected by the State Legislative Assembly"))
        self.assertEqual(label_of(prep, out["concerns"][0]["ref"]), "Constitution of India, Article 155")

    def test_not_covered_carries_no_concern(self):
        # v10 Governor s4: an opinion marked red as "unsupported" by the claim check.
        prep = prep_for("governor")
        sid = sid_of(prep, "Centre-State disputes")
        reply = dict(contradicted(sid, "none", "", "", both="yes"), verdict="not_covered")
        reply["concerns"][0]["kind"] = "unsupported"
        out = fact(prep, sid, reply)
        self.assertEqual(out["concerns"], [])

    def test_an_unproven_claim_clears_nothing(self):
        prov = {"id": "provisions", "kind": "check", "status": "ok", "verdict": "wrong", "credit": [], "rewrite": None,
                "note": "N.", "concerns": [{"sourceId": "s2", "quote": "q", "kind": "wrong", "priority": "major",
                                            "reason": "Appointed by the President.", "fix": "Fix it.", "ref": "r2"}]}
        unproven = {"id": "facts:s2", "kind": "fact", "status": "ok", "verdict": "not_applicable", "credit": [],
                    "concerns": [], "rewrite": None, "note": None, "unproven": True}
        out = pipeline.merge_facts([prov, unproven])
        p = next(r for r in out if r["id"] == "provisions")
        self.assertTrue([c for c in p["concerns"] if c["sourceId"] == "s2"])


class Sources(unittest.TestCase):
    def test_a_long_article_gives_the_claim_its_deciding_clause(self):
        prep = prep_for("356")
        sid = sid_of(prep, "continue indefinitely")
        text = " ".join(r.get("full") or r["text"] for r in claim_refs(prep, sid))
        self.assertIn("six months", text)

    def test_a_union_answer_is_not_checked_against_the_state_mirror(self):
        prep = prep_for("moneybill")
        labels = {r["label"] for r in prep["refs"]}
        self.assertNotIn("Constitution of India, Article 198", labels)
        self.assertNotIn("Constitution of India, Article 243A", labels)
        sid = sid_of(prep, "up to three months")
        self.assertIn("Constitution of India, Article 109", {r["label"] for r in claim_refs(prep, sid)})

    def test_a_claim_meets_the_article_that_settles_it_without_sharing_a_word(self):
        prep = prep_for("governor")
        sid = sid_of(prep, "elected by the State Legislative Assembly")
        self.assertIn("Constitution of India, Article 155", {r["label"] for r in claim_refs(prep, sid)})

    def test_cases_come_only_through_the_articles_the_answer_is_checked_against(self):
        cases = [r["label"] for r in prep_for("moneybill")["refs"] if r["label"].startswith("Case:")]
        self.assertTrue(cases)
        self.assertTrue(all("Puttaswamy" in c or "Rojer Mathew" in c for c in cases), cases)

    def test_textbook_slices_never_crowd_out_the_constitution(self):
        # v11 source check with NCERT: five slices of one textbook chapter took every slot of the DPSP claim.
        filler = ("Directive Principles of State Policy cannot be enforced in courts like Fundamental Rights; the State "
                  "applies them in making laws and courts read the two parts in harmony. ")
        book = [{"id": f"n102.{n}", "label": "NCERT Class 11, Indian Constitution at Work, Ch. 2 (Rights)",
                 "title": "Rights in the Indian Constitution", "text": filler * 6} for n in range(1, 7)]
        idx = refs.Index(refs.constitution_chunks(CONST) + book)
        q = "Discuss the relationship between Fundamental Rights and Directive Principles of State Policy."
        sents = [{"id": "s1", "text": "Fundamental Rights are in Part III and Directive Principles are in Part IV."},
                 {"id": "s2", "text": "Directive Principles can be enforced in courts in the same way as Fundamental "
                                      "Rights."}]
        mine = [r["label"] for r in idx.pack(q, sents) if "s2" in r.get("claims", [])]
        self.assertLessEqual(sum(1 for m in mine if m.startswith("NCERT")), 1, mine)
        self.assertIn("Constitution of India, Article 37", mine)

    def test_common_names_map_to_the_constitutions_words(self):
        self.assertEqual(refs.vocabulary("Rajya Sabha and Lok Sabha"), "council of states and house of the people")


def finding(cid, sid, kind, reason, rewrite=None, ref=None, kind_of="check"):
    return {"id": cid, "kind": kind_of, "status": "ok", "verdict": "partial", "credit": [], "note": "N.",
            "concerns": [{"sourceId": sid, "quote": "q", "kind": kind, "priority": "major", "reason": reason,
                          "fix": "Change the sentence.", "ref": ref}],
            "rewrite": {"sourceId": sid, "text": rewrite, "ref": None} if rewrite else None}


class Corrections(unittest.TestCase):
    def setUp(self):
        self.prep = prep_for("356")
        self.part = {"id": "part:p1", "kind": "part", "status": "ok", "verdict": "partial", "note": "N.",
                     "credit": [{"sourceId": "s1", "quote": "q", "reason": "Right."}], "concerns": [], "rewrite": None}

    def test_a_dropped_rewrite_of_a_wrong_fact_does_not_come_back_through_a_merge(self):
        # v10 Article 356: "continues until the situation is resolved" reached the student this way.
        rep = report.assemble(self.prep, [
            self.part,
            finding("concepts", "s3", "wrong", "It cannot continue indefinitely."),
            finding("causality", "s3", "wrong", "It cannot continue indefinitely.",
                    rewrite="Once approved, President's Rule continues until the situation is resolved.")])
        fix = next(f for f in rep["fixes"] if f["yourLine"] and f["yourLine"]["sourceId"] == "s3")
        self.assertIsNone(fix["tryWriting"])

    def test_an_opinion_checks_rewrite_stays(self):
        rep = report.assemble(self.prep, [
            self.part, finding("solutions", "s4", "wrong", "This works against the goal.",
                               rewrite="Its frequent use in the past weakened the autonomy of States, which courts now review.")])
        fix = next(f for f in rep["fixes"] if f["yourLine"] and f["yourLine"]["sourceId"] == "s4")
        self.assertTrue(fix["tryWriting"])

    def test_a_copied_extract_is_not_a_correction(self):
        ref = next(r for r in self.prep["refs"] if r["label"].endswith("Article 356"))
        sentence = next(s["text"] for s in self.prep["sentences"] if "indefinitely" in s["text"])
        rep = {"fixes": [{"n": 1, "kind": "wrong", "title": "t", "where": "w", "why": "It lapses.", "tryWriting": None,
                          "source": ref["label"], "sourceRef": ref["id"],
                          "yourLine": {"sourceId": "s3", "text": sentence}}]}

        class Ask:
            def ask(self, *, validate, attempts=3, **_):
                for value in ({"sentence": ref["text"][:400]},):
                    try:
                        return validate(value)
                    except Rejected:
                        pass
                raise Rejected("failed")

        report.correct(Ask(), self.prep, "sys", rep)
        self.assertIsNone(rep["fixes"][0]["tryWriting"])


class SecondOpinion(unittest.TestCase):
    """v12, from split run 45361966648: the Panchayat report's false contradiction was never re-checked."""

    def test_claim_contradictions_are_rechecked_first_and_outside_the_budget(self):
        from mains import verify
        prep = prep_for("governor")
        busy = [{"id": d, "kind": "check", "status": "ok", "verdict": "partial", "credit": [], "rewrite": None,
                 "note": "N.", "concerns": [{"sourceId": "s1", "quote": "q", "kind": "unsupported", "priority": "major",
                                             "reason": "R.", "fix": "F.", "ref": None}]}
                for d in ("directive", "scope", "dimensions", "concepts", "accuracy", "evidence", "causality",
                          "balance", "solutions")]
        prov = {"id": "provisions", "kind": "check", "status": "ok", "verdict": "wrong", "credit": [], "rewrite": None,
                "note": "N.", "concerns": [{"sourceId": "s1", "quote": "q", "kind": "wrong", "priority": "major",
                                            "reason": "Block is not the intermediate level.", "fix": "F.",
                                            "ref": "r1", "fromClaim": True}]}
        results = busy + [prov]
        got = verify.targets(prep, results)
        self.assertEqual(results[got[0][1]]["id"], "provisions")
        self.assertEqual(len(got), verify.LIMIT + 1)


class CheckedCorrections(unittest.TestCase):
    """v12: a correction is checked against the clash itself, not against how much it looks like the Constitution."""

    def run_line(self, name, phrase, label, wrong, said, line):
        prep = prep_for(name)
        sid = sid_of(prep, phrase)
        ref = next(r for r in claim_refs(prep, sid) if r["label"] == "Constitution of India, " + label)
        sentence = next(s["text"] for s in prep["sentences"] if s["id"] == sid)
        rep = {"fixes": [{"n": 1, "kind": "wrong", "title": "t", "where": "w", "why": "w.", "tryWriting": None,
                          "source": ref["label"], "sourceRef": ref["id"],
                          "evidence": {"wrongWords": wrong, "sourceWords": said},
                          "yourLine": {"sourceId": sid, "text": sentence}}]}

        class Ask:
            def ask(self, *, validate, **_):
                return validate({"sentence": line})

        try:
            report.correct(Ask(), prep, "sys", rep)
        except Rejected:
            pass
        return rep["fixes"][0]["tryWriting"]

    def test_a_correct_line_worded_like_the_constitution_is_kept(self):
        # v11 refused this kind of line for Articles 356, 109 and 243G.
        self.assertTrue(self.run_line("moneybill", "up to three months", "Article 109", "up to three months",
                                      "within a period of fourteen days",
                                      "The Rajya Sabha must return a Money Bill within fourteen days."))

    def test_a_line_without_the_sources_fact_is_dropped(self):
        self.assertIsNone(self.run_line("356", "continue indefinitely", "Article 356", "continue indefinitely",
                                        "cease to operate on the expiration of a period of six months",
                                        "Once approved, President's Rule can continue until the situation is resolved."))

    def test_a_negated_correction_counts(self):
        self.assertTrue(self.run_line("governor", "elected by the State Legislative Assembly", "Article 155",
                                      "elected by the State Legislative Assembly",
                                      "appointed by the President by warrant under his hand and seal",
                                      "The Governor is not elected: the President appoints the Governor."))


class QuoteMustDiffer(unittest.TestCase):
    """v13, from split run 45382853367: a correct Money Bill sentence was marked wrong with a quote that agrees."""

    def test_a_source_quote_that_says_the_same_thing_is_not_a_contradiction(self):
        self.assertTrue(checks.quote_agrees(
            "After the Lok Sabha passes it, the Bill is sent to the Rajya Sabha for its recommendations.",
            "After a Money Bill has been passed by the House of the People it shall be transmitted to the Council "
            "of States for its recommendations"))

    def test_every_planted_error_still_counts(self):
        for wrong, said in [
                ("can be introduced in either House", "A Money Bill shall not be introduced in the Council of States."),
                ("up to three months", "shall within a period of fourteen days from the date of its receipt"),
                ("Election Commission of India", "vested in a State Election Commission"),
                ("elected by the State Legislative Assembly", "The Governor of a State shall be appointed by the President"),
                ("only to citizens of India", "No person shall be deprived of his life or personal liberty"),
                ("continue indefinitely", "cease to operate on the expiration of a period of six months")]:
            self.assertFalse(checks.quote_agrees(wrong, said), wrong)

    def test_the_claim_check_retries_then_marks_nothing(self):
        prep = prep_for("moneybill")
        sid = sid_of(prep, "sent to the Rajya Sabha")
        a109 = next(r for r in claim_refs(prep, sid) if r["label"].endswith("Article 109"))
        reply = contradicted(sid, a109["id"],
                             "After a Money Bill has been passed by the House of the People it shall be transmitted "
                             "to the Council of States for its recommendations",
                             "After the Lok Sabha passes it, the Bill is sent to the Rajya Sabha")
        out = fact(prep, sid, reply)
        self.assertEqual(out["concerns"], [])
        self.assertTrue(out.get("unproven"))


class ThirdView(unittest.TestCase):
    """v13.1: a narrow same-or-different look at each upheld claim contradiction, which can only remove a red mark."""

    def setUp(self):
        from mains import verify
        self.verify = verify

    def results(self, sid, wrong, said):
        return [{"id": "provisions", "kind": "check", "status": "ok", "verdict": "wrong", "credit": [], "rewrite": None,
                 "note": "N.", "concerns": [{"sourceId": sid, "quote": "q", "kind": "wrong", "priority": "major",
                                             "reason": "R.", "fix": "F.", "ref": None, "fromClaim": True,
                                             "evidence": {"wrongWords": wrong, "sourceWords": said}}]}]

    def run_third(self, prep, results, verdict):
        asked = []

        class Ask:
            def ask(self, *, validate, check, **_):
                asked.append(check)
                return validate({"difference": "x", "verdict": verdict})

        self.verify.third_view(Ask(), prep, "sys", results)
        self.verify.settle(results)
        return asked

    def test_same_meaning_removes_the_red_mark(self):
        prep = prep_for("moneybill")
        res = self.results("s2", "After the Lok Sabha passes it, the Bill is sent to the Rajya Sabha",
                           "After a Money Bill has been passed by the House of the People it shall be transmitted")
        self.assertEqual(self.run_third(prep, res, "same"), ["third:s2"])
        self.assertEqual(res[0]["concerns"], [])

    def test_a_real_difference_keeps_the_red_mark(self):
        prep = prep_for("governor")
        res = self.results("s2", "elected by the State Legislative Assembly",
                           "The Governor of a State shall be appointed by the President")
        self.run_third(prep, res, "different")
        self.assertEqual(len(res[0]["concerns"]), 1)

    def test_a_hard_difference_is_not_asked_and_cannot_be_removed(self):
        prep = prep_for("moneybill")
        res = self.results("s3", "up to three months", "shall within a period of fourteen days")
        self.assertEqual(self.run_third(prep, res, "same"), [])
        self.assertEqual(len(res[0]["concerns"]), 1)


if __name__ == "__main__":
    unittest.main()
