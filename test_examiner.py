"""Unit tests for the Mains examiner pipeline. No model: a scripted stand-in plays Gemma.

Run: python -m unittest -v
"""
import contextlib
import io
import json
import unittest

from mains import breakdown, checks, pipeline, refs, report, rubric, text, verify
from mains.model import Examiner
from mains.text import Rejected
# Unit tests never read the real reference index on disk: once it holds the GS textbooks (v16), every answer gets
# sources and checks the scripted fake model has no reply for. Tests that need sources build their own index.
refs._LOADED[refs.INDEX] = None


Q_EN = "Discuss how urban wetlands reduce flood risk and suggest ways to conserve them."
A_EN = ("Urban wetlands hold stormwater after heavy rain, so less water enters drains at once. This is why wetland "
        "restoration has reduced flooding by 80% in every Indian city where it has been attempted. Municipalities "
        "should concrete all wetland beds to stop seepage, build retaining walls and prohibit any human entry. "
        "Regular desilting and community monitoring can help protect the area. These measures will end urban floods.")
Q_HI = ("समझाइए कि हीट एक्शन प्लान शहरों में गर्मी का जोखिम कैसे कम करते हैं। भारतीय शहरों में इन्हें लागू करने की "
        "दो चुनौतियों पर चर्चा कीजिए।")
A_HI = ("हीट एक्शन प्लान में लोगों को गर्मी के खतरे के बारे में बताना चाहिए। नगर निगम मोबाइल पर संदेश भेज सकता है "
        "और स्कूलों में पोस्टर लगा सकता है। लोगों को खूब पानी पीना चाहिए और धूप में जाने से बचना चाहिए। अगर "
        "जागरूकता बढ़ेगी तो गर्मी से कम लोग बीमार होंगे। इस काम के लिए सभी को मिलकर प्रयास करना चाहिए।")
DIMS = ["directive", "scope", "dimensions", "concepts", "accuracy", "evidence", "causality", "balance", "solutions",
        "structure", "language", "presentation"]


def good(sid, reason, note):
    return {"credit": [{"sourceId": sid, "reason": reason}], "concerns": [], "note": note, "verdict": "good",
            "rewrite": {"sourceId": "none", "text": ""}}


def concern(sid, kind, priority, reason, fix):
    return {"sourceId": sid, "kind": kind, "priority": priority, "reason": reason, "fix": fix}


EN = {
    "breakdown": {"directives": ["Discuss", "suggest"],
                  "parts": [{"text": "Discuss how urban wetlands reduce flood risk", "count": 0},
                            {"text": "suggest ways to conserve them", "count": 0}]},
    "part:p1": {"credit": [{"sourceId": "s1", "reason": "It explains that wetlands store stormwater, so drains get less water at once."}],
                "concerns": [concern("s2", "unsupported", "major", "The 80% figure for every city has no source and overclaims.",
                                     "Remove the 80% claim or name a study you have read.")],
                "note": "The storage mechanism is right; the statistic weakens it.", "verdict": "met",
                "rewrite": {"sourceId": "s1", "text": "Urban wetlands store stormwater and release it slowly, which lowers the peak flow into drains."}},
    "part:p2": {"credit": [{"sourceId": "s4", "reason": "Community monitoring is a practical conservation step."}],
                "concerns": [concern("s3", "wrong", "major", "Concreting the bed stops water soaking in, which destroys the flood buffer.",
                                     "Replace concreting with protecting the natural inflow and outflow paths.")],
                "note": "One good measure, one that harms the wetland.", "verdict": "partial",
                "rewrite": {"sourceId": "s3", "text": "Municipalities should protect the wetland's natural inflow and outflow paths and stop encroachment."}},
    "accuracy": {"credit": [], "concerns": [concern("s2", "unsupported", "major", "The 80% figure for every Indian city cannot be verified.",
                                                    "Remove the 80% claim, or replace it with a source you can name.")],
                 "note": "One unsupported universal claim.", "verdict": "unsupported", "rewrite": {"sourceId": "none", "text": ""}},
    "solutions": {"credit": [{"sourceId": "s4", "reason": "Community monitoring names who acts."}],
                  "concerns": [concern("s3", "wrong", "major", "Concreting wetland beds works against conserving them.",
                                       "Replace concreting with steps that keep the wetland working.")],
                  "note": "One workable and one harmful suggestion.", "verdict": "wrong", "rewrite": {"sourceId": "none", "text": ""}},
    "evidence": {"credit": [], "concerns": [concern("none", "missing", "medium", "No example supports the points made.",
                                                    "Add one wetland or city plan you have studied.")],
                 "note": "No example is given.", "verdict": "missing", "rewrite": {"sourceId": "none", "text": ""}},
    "structure": {"credit": [{"sourceId": "s1", "reason": "The answer opens straight with the mechanism."}],
                  "concerns": [concern("s5", "unsupported", "minor", "Saying the measures will end urban floods overclaims.",
                                       "End with a careful claim such as reducing flood risk.")],
                  "note": "Clear opening; the conclusion overclaims.", "verdict": "partial", "rewrite": {"sourceId": "s5", "text": "These measures can reduce urban flood risk."}},
    "verify": {"coveredBy": [], "reason": "The criticism matches the sentence.", "decision": "upheld"},
    "summary": {"summary": "Your explanation of how wetlands store stormwater works. Concreting the wetland bed and the unsupported statistic cost the most. Fix the conservation part first."},
}
for d in DIMS:
    EN.setdefault(d, good("s1", "The sentence states the storage mechanism clearly and directly.", "This check is done well."))

HI = {
    "breakdown": {"directives": ["समझाइए", "चर्चा कीजिए"],
                  "parts": [{"text": "समझाइए कि हीट एक्शन प्लान शहरों में गर्मी का जोखिम कैसे कम करते हैं", "count": 0},
                            {"text": "भारतीय शहरों में इन्हें लागू करने की दो चुनौतियों पर चर्चा कीजिए", "count": 0}]},
    "part:p1": {"credit": [{"sourceId": "s2", "reason": "नगर निगम का मोबाइल संदेश एक ठोस कदम है।"}],
                "concerns": [concern("s1", "weak", "major", "जागरूकता प्लान का सिर्फ़ एक हिस्सा है, चेतावनी के बाद की कार्रवाई नहीं बताई।",
                                     "लिखिए कि चेतावनी के बाद नगर निगम और अस्पताल क्या करते हैं।")],
                "note": "जागरूकता की बात है, पर कार्रवाई नहीं।", "verdict": "partial",
                "rewrite": {"sourceId": "s1", "text": "तापमान तय सीमा पार करते ही प्लान चेतावनी को स्थानीय कार्रवाई से जोड़ता है।"}},
    "part:p2": {"credit": [], "concerns": [concern("none", "missing", "major", "सवाल दो चुनौतियाँ माँगता है, जवाब में एक भी नहीं है।",
                                                   "जवाब के अंत में दो चुनौतियाँ जोड़िए।")],
                "note": "यह हिस्सा छूट गया है।", "verdict": "missing", "rewrite": {"sourceId": "none", "text": ""}},
    "verify": {"coveredBy": [], "reason": "No sentence discusses implementation challenges.", "decision": "upheld"},
    "summary": {"summary": "नगर निगम के संदेश वाला सुझाव ठोस है। दो चुनौतियाँ न लिखने से सबसे ज़्यादा अंक कटते हैं। पहले दूसरा हिस्सा जोड़िए।"},
}
for d in DIMS:
    HI.setdefault(d, good("s2", "इस वाक्य में साफ़ बताया गया है कि नगर निगम क्या करेगा।", "यह जाँच ठीक है।"))


class FakeGemma:
    """Answers by the TASK marker. `bad` maps a task to a first reply that must be rejected."""

    def __init__(self, script, bad=None):
        self.script, self.bad, self.prompts = script, dict(bad or {}), []

    def create_chat_completion(self, messages, **_):
        user = messages[1]["content"]
        self.prompts.append(messages)
        key = user.split("TASK [", 1)[1].split("]", 1)[0]
        # A follow-up pass (e.g. "solutions:rest") falls back to its base check's scripted reply unless scripted.
        value = self.script[key] if key in self.script else self.script[key.split(":", 1)[0]]
        if key in self.bad and "rejected by an automatic quality check" not in user:
            value = self.bad[key]
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value, ensure_ascii=False)}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200}}


def job(lang="en", **kw):
    base = {"id": "t1", "lang": lang, "question": Q_EN if lang == "en" else Q_HI,
            "answer": A_EN if lang == "en" else A_HI, "references": ["Store rainfall and slow runoff"], "marks": 10}
    return dict(base, **kw)


class TextRules(unittest.TestCase):
    def test_sentences_keep_exact_wording(self):
        s = text.sentences(A_EN)
        self.assertEqual(len(s), 5)
        self.assertEqual(s[1]["text"], "This is why wetland restoration has reduced flooding by 80% in every Indian city where it has been attempted.")
        self.assertEqual(len(text.sentences(A_HI)), 5)
        self.assertEqual([x["para"] for x in text.sentences("One point here.\n\nSecond point here.")], [1, 2])

    def test_invented_numbers_are_caught(self):
        self.assertEqual(text.unsourced_numbers("Cut floods by 75% as in 2019.", A_EN), ["75", "2019"])
        self.assertEqual(text.unsourced_numbers("Remove the 80% claim from Part 2.", A_EN), [])
        self.assertEqual(text.unsourced_numbers("२०१९ में", A_EN), ["2019"])

    def test_script_rules(self):
        self.assertIsNone(text.script_problem("यह वाक्य साफ़ है और UPSC के लिए ठीक है।", "hi"))
        self.assertEqual(text.script_problem("यह slums की बात है और ठीक नहीं है।", "hi"), "roman_in_hindi")
        self.assertEqual(text.script_problem("This mentions बाढ़ for no reason.", "en"), "devanagari_in_english")

    def test_clean_rejects_gibberish(self):
        cases = {"good good good good good good good good good good good good good": "repetition",
                 "This deserves 7/10 for effort.": "marks", "Well done, keep up the good work.": "filler",
                 "Look at s3 again carefully.": "source_id"}
        for value, code in cases.items():
            with self.assertRaises(Rejected) as e:
                text.clean(value, "en", [A_EN])
            self.assertTrue(e.exception.code.endswith(code), e.exception.code)


class Stages(unittest.TestCase):
    def test_breakdown_must_be_verbatim(self):
        ok = breakdown.validate(EN["breakdown"], Q_EN)
        self.assertEqual([p["quote"] for p in ok["parts"]],
                         ["Discuss how urban wetlands reduce flood risk", "suggest ways to conserve them"])
        with self.assertRaises(Rejected):
            breakdown.validate({"parts": [{"text": "explain wetland benefits", "count": 0}]}, Q_EN)
        hi = breakdown.validate(HI["breakdown"], Q_HI)
        self.assertEqual(hi["parts"][1]["count"], 2)  # "दो चुनौतियों" sets the count even when the model says 0

    def test_breakdown_falls_back_to_sentences(self):
        ex = Examiner(FakeGemma({"breakdown": {"directives": [], "parts": [{"text": "made up text", "count": 0}]}}))
        out = breakdown.run(ex, {"question": Q_HI})
        self.assertEqual(out["source"], "fallback")
        self.assertEqual(len(out["parts"]), 2)

    def test_check_binding_rules(self):
        ex = Examiner(FakeGemma(EN))
        prep = pipeline.prepare(ex, job())
        bad = dict(EN["part:p2"], verdict="met", credit=[])
        with self.assertRaises(Rejected):
            checks.bind(bad, prep, "part:p2")
        with self.assertRaises(Rejected):
            checks.bind(dict(EN["part:p1"], credit=[{"sourceId": "s9", "reason": "Not a real sentence id at all."}]), prep, "part:p1")
        bound = checks.bind(EN["part:p1"], prep, "part:p1")
        self.assertEqual(bound["credit"][0]["quote"], text.sentences(A_EN)[0]["text"])

    def test_duplicate_concerns_are_deduped(self):
        # Two concerns about the same sentence with the same kind are redundant: keep the first, do not reject.
        ex = Examiner(FakeGemma(EN))
        prep = pipeline.prepare(ex, job())
        dup = {"credit": [{"sourceId": "s1", "reason": "It explains that wetlands store stormwater."}],
               "concerns": [concern("s2", "unsupported", "major", "The 80% figure is unsupported.", "Remove the 80% claim."),
                            concern("s2", "unsupported", "medium", "The 80% figure is unsupported.", "Remove the 80% claim.")],
               "note": "Two notes about the same claim.", "verdict": "partial",
               "rewrite": {"sourceId": "none", "text": ""}}
        bound = checks.bind(dup, prep, "part:p1")
        self.assertEqual(len(bound["concerns"]), 1)
        self.assertEqual(bound["concerns"][0]["priority"], "major")

    def test_rejected_output_gets_one_corrected_retry(self):
        invented = good("s1", "Like the 2019 Chennai plan, this cut floods by 75% in a year.", "Done well.")
        fake = FakeGemma(EN, bad={"directive": invented})
        ex = Examiner(fake)
        prep = pipeline.prepare(ex, job())
        out = checks.run(ex, prep, "sys", "directive")
        self.assertEqual(out["status"], "ok")
        self.assertEqual([c["status"] for c in ex.calls if c["check"] == "directive"], ["rejected", "ok"])

    def test_overturned_finding_is_dropped(self):
        ex = Examiner(FakeGemma(dict(EN, verify={"coveredBy": ["s1"], "reason": "Sentence s1 already covers this.", "decision": "overturned"})))
        prep = pipeline.prepare(ex, job())
        results = [r for s in pipeline.SHARD_ORDER for r in pipeline.run_shard(ex, prep, s)]
        verify.run(ex, prep, "sys", results)
        acc = next(r for r in results if r["id"] == "accuracy")
        self.assertEqual(acc["concerns"], [])
        self.assertEqual(acc["verdict"], "good" if acc["credit"] else "not_applicable")


class Report(unittest.TestCase):
    def run_all(self, script, lang, **kw):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rep = pipeline.evaluate(Examiner(FakeGemma(script)), job(lang, **kw))
        return rep, out.getvalue()

    def test_second_look_flags_a_lesser_suggestion(self):
        script = dict(EN)
        script["solutions:s4"] = {
            "credit": [], "note": "Desilting is offered as routine, not tied to a need.", "verdict": "partial",
            "concerns": [concern("s4", "weak", "medium", "Regular desilting everywhere is not tied to a surveyed need.",
                                 "Say desilting is done where a survey finds excess silt.")],
            "rewrite": {"sourceId": "none", "text": ""}}
        rep, _ = self.run_all(script, "en")
        tone = {h["sourceId"]: h["tone"] for h in rep["highlights"]}
        self.assertEqual(tone["s4"], "amber")                               # flagged, no longer green
        self.assertNotIn("s4", [s["sourceId"] for s in rep["strengths"]])   # and not praised
        self.assertEqual(tone["s3"], "red")                                 # the first finding still stands

    def test_second_look_ignores_claims_and_other_sentences(self):
        script = dict(EN)
        # A claim judged "not a measure" adds nothing, even if a concern slips into the reply.
        script["solutions:s5"] = {"credit": [], "note": "Not a measure.", "verdict": "not_applicable",
                                  "concerns": [concern("s5", "weak", "minor", "Overclaims.", "Soften the claim.")],
                                  "rewrite": {"sourceId": "none", "text": ""}}
        base, _ = self.run_all(EN, "en")
        rep, _ = self.run_all(script, "en")
        self.assertEqual([h["tone"] for h in rep["highlights"]], [h["tone"] for h in base["highlights"]])
        self.assertEqual(len(rep["fixes"]), len(base["fixes"]))

    def test_english_report(self):
        rep, logs = self.run_all(EN, "en")
        self.assertEqual([p["status"] for p in rep["parts"]], ["met", "partial"])
        self.assertEqual(rep["level"], "developing")
        self.assertEqual(rep["headline"], "Part 1 is answered. Part 2 needs the most work.")
        # Two checks flagged the 80% claim and two flagged concreting: each appears once among the fixes.
        lines = [f["yourLine"]["sourceId"] for f in rep["fixes"] if f["yourLine"]]
        self.assertEqual(len(lines), len(set(lines)))
        self.assertEqual({f["yourLine"]["sourceId"] for f in rep["fixes"][:2]}, {"s2", "s3"})
        tones = {h["sourceId"]: h["tone"] for h in rep["highlights"]}
        self.assertEqual((tones["s1"], tones["s2"], tones["s3"]), ("green", "red", "red"))
        # The same problem flagged by two checks is shown to the student once.
        s3 = next(h for h in rep["highlights"] if h["sourceId"] == "s3")
        self.assertEqual(len(s3["concerns"]), 1)
        self.assertEqual(rep["fixes"][2]["yourLine"], None)  # third fix: add an example
        self.assertEqual(rep["directives"][0]["word"], "Discuss")
        self.assertEqual(rep["directives"][1]["word"], "Suggest")
        self.assertTrue(rep["summary"])
        self.assertEqual(rep["stats"]["limit"], 150)
        self.assertGreater(rep["trust"]["secondOpinions"], 0)
        # Logs show the work, never the content.
        self.assertIn('"stage":"check"', logs)
        for s in text.sentences(A_EN):
            self.assertNotIn(s["text"][:30], logs)

    def test_hindi_report_with_missing_part(self):
        rep, _ = self.run_all(HI, "hi")
        self.assertEqual([p["status"] for p in rep["parts"]], ["partial", "missing"])
        self.assertEqual(rep["headline"], "भाग 2 छूट गया है। भाग 1 अधूरा है।")
        self.assertEqual(rep["missing"][0]["partId"], "p2")
        self.assertEqual(rep["fixes"][0]["yourLine"], None)  # the missing part is the first fix
        self.assertEqual(rep["levelLabel"], "सुधार चाहिए")
        self.assertEqual([d["word"] for d in rep["directives"]], ["समझाइए", "चर्चा कीजिए"])  # question order

    def test_shards_match_single_runner(self):
        single, _ = self.run_all(EN, "en")
        ex = Examiner(FakeGemma(EN))
        with contextlib.redirect_stdout(io.StringIO()):
            prep = pipeline.prepare(ex, job())
            results = []
            for shard in reversed(pipeline.SHARD_ORDER):  # order of arrival must not matter
                results += pipeline.run_shard(ex, json.loads(json.dumps(prep)), shard)
            split = pipeline.finish(ex, prep, results)
        self.assertEqual(json.dumps(single, sort_keys=True), json.dumps(split, sort_keys=True))

    def test_two_failed_checks_show_not_checked(self):
        # A check that cannot be judged adds no finding: it is shown as "Not checked", never as an invented gap.
        script = dict(EN, concepts={"verdict": "nonsense"}, language={"verdict": "nonsense"})
        rep, _ = self.run_all(script, "en")
        verdicts = {d["id"]: d["verdict"] for d in rep["checks"]}
        self.assertEqual((verdicts["concepts"], verdicts["language"]), ("unavailable", "unavailable"))
        self.assertEqual(rep["trust"]["unavailableChecks"], ["concepts", "language"])
        self.assertTrue(all(f["where"] not in ("Concepts explained", "Clear wording") for f in rep["fixes"]))

    def test_many_failed_checks_still_build_a_report(self):
        script = {k: v for k, v in EN.items() if k not in ("concepts", "accuracy", "evidence")}
        script.update({k: {"verdict": "nonsense"} for k in ("concepts", "accuracy", "evidence")})
        rep, _ = self.run_all(script, "en")
        unavailable = {d["id"] for d in rep["checks"] if d["verdict"] == "unavailable"}
        self.assertTrue({"concepts", "accuracy", "evidence"} <= unavailable)

    def test_same_advice_twice_is_one_fix(self):
        a = "दो विशिष्ट चुनौतियां जोड़ें, जैसे कि फंड की कमी या अलग-अलग विभागों के बीच तालमेल की कमी।"
        b = "भारतीय शहरों में फंड की कमी और विभागों के बीच तालमेल की कमी जैसी दो चुनौतियों को शामिल करें।"
        self.assertTrue(report.same_advice(a, b))
        self.assertFalse(report.same_advice("Replace concreting with protecting the natural inflow paths.",
                                            "Remove the 80% claim or name a study you have read."))

    def test_same_rewrite_twice_is_one_fix_and_repeated_sentences_are_trimmed(self):
        same_line = "Urban wetlands reduce flood risk by acting as natural sponges that store rainfall."
        a = {"fix": "Add a sentence explaining how wetlands absorb excess rainwater.", "rewrite": same_line}
        b = {"fix": "Explain the mechanism by which wetlands mitigate flooding.", "rewrite": same_line}
        self.assertTrue(report.repeats(b, [a]))
        c = {"fix": "Add hospital preparation and protection for outdoor workers.",
             "rewrite": "Plans protect outdoor workers and prepare hospitals. " + same_line}
        self.assertFalse(report.repeats(c, [a]))
        report.trim_rewrite(c, [a])
        self.assertEqual(c["rewrite"], "Plans protect outdoor workers and prepare hospitals.")

    def test_dimension_gap_matching_a_missing_part_is_folded(self):
        # The balance check also asking for the two challenges repeats Part 2's own finding: one fix, not two.
        hi = dict(HI, balance={"credit": [], "concerns": [concern("none", "missing", "major",
                  "भारतीय शहरों में लागू करने की चुनौतियाँ नहीं लिखीं।",
                  "भारतीय शहरों में इन योजनाओं को लागू करने में आने वाली दो मुख्य चुनौतियों को लिखें।")],
                  "note": "चुनौतियाँ नहीं लिखी गईं।", "verdict": "missing", "rewrite": {"sourceId": "none", "text": ""}})
        rep, _ = self.run_all(hi, "hi")
        self.assertEqual([f["where"] for f in rep["fixes"]].count("सीमाएँ और मुश्किलें"), 0)

    def test_extra_angle_on_a_complete_answer_is_not_a_big_problem(self):
        # Every part answered: a point a dimension check would like added is medium, so the level stays "Good".
        en = dict(EN, **{"part:p2": dict(EN["part:p2"], verdict="met", concerns=[])},
                  accuracy=good("s1", "The storage mechanism is stated accurately.", "No factual problem."),
                  solutions=good("s4", "Community monitoring names who acts.", "Workable."),
                  dimensions={"credit": [{"sourceId": "s1", "reason": "Covers the hydrological angle clearly."}],
                              "concerns": [concern("none", "missing", "major", "The economic angle is not covered.",
                                                   "Add one sentence on the economic cost of floods.")],
                              "note": "One angle is missing.", "verdict": "partial",
                              "rewrite": {"sourceId": "none", "text": ""}})
        # No word limit here, so only severity decides. The four-sentence fixture has no room for an introduction, a
        # conclusion and five points, so its shape keeps it Developing; the missing angle must not be one more reason.
        rep, _ = self.run_all(en, "en", marks=None)
        self.assertTrue(all(f["impact"] != "major" or f["where"].startswith("Part") for f in rep["fixes"]
                            if f["yourLine"] is None))
        self.assertFalse([w for w in rep["levelWhy"] if "serious" in w or "angle" in w], rep["levelWhy"])


class MarkingStandard(unittest.TestCase):
    """The level follows docs/marking-standard.md: length, shape and the command word are checked in code."""

    def form(self, **kw):
        return dict({"intro": "s1", "conclusion": "s9", "points": 6, "pointsNeeded": 5, "paragraphs": 3}, **kw)

    def level(self, words, limit=150, form=None, dims=None, majors=(), parts=("met",), L=0):
        ps = [{"status": s} for s in parts]
        ds = {d: {"verdict": "good"} for d in ("directive", "scope", "dimensions", "evidence", "structure")}
        ds.update(dims or {})
        length = {"words": words, "limit": limit, "ratio": round(words / limit, 2) if limit else None}
        return report.level_of(ps, ds, list(majors), length, self.form() if form is None else form, L)

    def test_a_complete_accurate_answer_far_under_the_limit_is_developing(self):
        lvl, why = self.level(69)          # the v14 Governor answer: every part met, no errors, called "Good"
        self.assertEqual(lvl, "developing")
        self.assertIn("69 words", why[0])

    def test_full_length_well_shaped_answer_can_be_strong(self):
        self.assertEqual(self.level(150), ("strong", []))

    def test_no_conclusion_caps_at_developing(self):
        lvl, why = self.level(150, form=self.form(conclusion=None))
        self.assertEqual(lvl, "developing")
        self.assertTrue(any("conclusion" in w for w in why))

    def test_too_few_points_caps_at_developing(self):
        self.assertEqual(self.level(150, form=self.form(points=3))[0], "developing")

    def test_directive_not_met_caps_at_developing(self):
        self.assertEqual(self.level(150, dims={"directive": {"verdict": "partial"}})[0], "developing")

    def test_missing_evidence_stops_strong_only(self):
        lvl, why = self.level(150, dims={"evidence": {"verdict": "partial"}})
        self.assertEqual(lvl, "good")
        self.assertTrue(why)

    def test_between_sixty_and_seventy_percent_needs_the_angles(self):
        self.assertEqual(self.level(100)[0], "good")
        self.assertEqual(self.level(100, dims={"dimensions": {"verdict": "partial"}})[0], "developing")

    def test_much_too_long_is_not_strong(self):
        self.assertEqual(self.level(220)[0], "good")

    def test_hindi_gets_an_allowance(self):
        prep = {"lang": "hi", "stats": {"words": 160, "limit": 150}}
        self.assertEqual(report.length_of(prep)["ratio"], round(160 / 180, 2))

    def test_printed_word_limit_wins(self):
        self.assertEqual(rubric.word_limit("Examine X. (Answer in 250 words)", "gs-10", 10), 250)
        self.assertEqual(rubric.word_limit("Examine X.", "ethics-case-20", None), 250)
        self.assertEqual(rubric.word_limit("Examine X.", None, 15), 250)
        self.assertIsNone(rubric.word_limit("Examine X.", None, None))
        self.assertEqual(rubric.word_limit("X की समीक्षा कीजिए। (उत्तर 150 शब्दों में)", None, None), 150)

    def test_form_needs_intro_at_the_start_and_conclusion_at_the_end(self):
        sents = [{"id": f"s{i}", "para": 1 if i < 3 else (2 if i < 8 else 3), "text": "x"} for i in range(1, 10)]
        prep = {"sentences": sents, "marks": 10}
        pts = [{"sourceId": f"s{i}", "point": f"point number {i}"} for i in range(2, 9)]
        f = checks.form_of(prep, {"introId": "s1", "conclusionId": "s9", "points": pts})
        self.assertEqual((f["intro"], f["conclusion"], f["points"]), ("s1", "s9", 7))
        f = checks.form_of(prep, {"introId": "s5", "conclusionId": "s4", "points": pts})
        self.assertEqual((f["intro"], f["conclusion"]), (None, None))
        one_para = [dict(s, para=1) for s in sents]
        f = checks.form_of(dict(prep, sentences=one_para), {"introId": "s1", "conclusionId": "s9", "points": pts})
        self.assertEqual(f["conclusion"], "s9")
        same = [{"sourceId": "s2", "point": "the same point"}] * 4
        self.assertEqual(checks.form_of(prep, {"introId": "none", "conclusionId": "none", "points": same})["points"], 1)

    def test_length_fix_is_first_and_uses_the_real_numbers(self):
        f = report.length_finding({"marks": 10}, {"words": 69, "limit": 150, "ratio": 0.46}, 0)
        self.assertEqual(f["fix"], "Write about 150 words: this answer has 69.")
        self.assertEqual(f["priority"], "major")
        self.assertIsNone(report.length_finding({"marks": 10}, {"words": 140, "limit": 150, "ratio": 0.93}, 0))

    def test_case_studies_say_what_is_not_checked_yet(self):
        prep = {"paper": "ethics_case", "stats": {"limit": 250}}
        self.assertIn("not available yet", report.format_note(prep, 0))
        self.assertIsNone(report.format_note({"paper": "gs", "stats": {"limit": 150}}, 0))

    def test_directive_task_names_the_failure_to_avoid(self):
        prep = {"question": "Critically examine the role of the Governor.", "parts": [], "sentences": []}
        self.assertIn("no reasoned verdict", checks.task(prep, "directive"))


class NoInventedContent(unittest.TestCase):
    def test_names_not_in_materials_are_rejected(self):
        for value in ("For example, the AMRUT scheme can serve as a model.",
                      "Cities should adopt frameworks like the National Water Policy.",
                      "For example, the rejuvenation of the Yamuna floodplains shows this."):
            with self.assertRaises(Rejected) as e:
                text.clean(value, "en", [Q_EN, A_EN])
            self.assertEqual(e.exception.code, "text:unsourced_name")
        # Names that the question or answer use, and sentence-initial capitals, are fine.
        self.assertTrue(text.clean("Municipalities should protect the inflow paths; Indian cities need this.",
                                   "en", [Q_EN, A_EN]))

    def test_evidence_check_gives_advice_only(self):
        ex = Examiner(FakeGemma(EN))
        prep = pipeline.prepare(ex, job())
        raw = {"credit": [], "concerns": [concern("none", "missing", "major", "No example supports the points made.",
                                                  "Add one wetland or city plan you have studied.")],
               "note": "No example is given.", "verdict": "missing",
               "rewrite": {"sourceId": "none", "text": "Cities such as the one you studied restored their wetlands."}}
        bound = checks.bind(raw, prep, "evidence")
        self.assertIsNone(bound["rewrite"])
        self.assertEqual(bound["concerns"][0]["priority"], "medium")

    def test_not_applicable_with_problems_is_partial(self):
        ex = Examiner(FakeGemma(EN))
        prep = pipeline.prepare(ex, job())
        raw = {"credit": [], "concerns": [concern("s3", "weak", "minor", "Banning all entry ignores the people nearby.",
                                                  "Say who uses the wetland and how they are involved.")],
               "note": "The answer does not weigh who is affected.", "verdict": "not_applicable",
               "rewrite": {"sourceId": "none", "text": ""}}
        self.assertEqual(checks.bind(raw, prep, "balance")["verdict"], "partial")


if __name__ == "__main__":
    unittest.main()
