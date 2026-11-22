import unittest
from grader import CRITERIA, validate


class GraderTests(unittest.TestCase):
    def test_off_topic_requires_no_strength_invention(self):
        for lang in ("en", "hi"):
            hi = lang == "hi"
            scores = dict.fromkeys(CRITERIA, 0)
            for short in ("", "None", "कोई नहीं"):
                report = {"scores": scores, "strength": short,
                          "gap": "उत्तर प्रश्न से जुड़ा नहीं है।" if hi else "The answer misses the question.",
                          "nextStep": "प्रश्न के पहले भाग को समझाएँ।" if hi else "Start by answering the first part of the question.",
                          "quote": ""}
                self.assertGreaterEqual(len(validate(report, "synthetic", lang)["strength"]), 8)

    def test_partial_answers_still_require_structured_feedback(self):
        scores = dict.fromkeys(CRITERIA, 1)
        report = {"scores": scores, "strength": "None", "gap": "The answer is missing a key point.",
                  "nextStep": "Explain why flood peaks fall.", "quote": ""}
        with self.assertRaises(ValueError):
            validate(report, "synthetic", "en")

    def test_off_topic_missing_gap_still_fails(self):
        report = {"scores": dict.fromkeys(CRITERIA, 0), "strength": "None", "gap": "",
                  "nextStep": "Explain why flood peaks fall.", "quote": ""}
        with self.assertRaises(ValueError):
            validate(report, "synthetic", "en")
