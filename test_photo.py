"""Unit tests for photo transcription. No model: a scripted reader plays Gemma, pages are drawn here.

Run: python -m unittest -v test_photo
"""
import io
import unittest

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from mains import photo

TRUTH = [
    "Urban wetlands hold stormwater after heavy rain.",
    "They release it slowly into the drains.",
    "Cities should map and protect wetland edges.",
    "Community monitoring can report blocked inlets.",
    "These steps lower the flood peak.",
]


def font(size=44):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def page(lines=TRUTH, ruled=True, dim=False, para_after=1, clip=False, tilt=0.0, smudge=None):
    w, h = 1240, 1600
    img = Image.new("RGB", (w, h), "#fffaf0")
    d = ImageDraw.Draw(img)
    if ruled:
        for y in range(120, h, 90):
            d.line((0, y, w, y), fill="#c9d6e0", width=2)
        d.line((90, 0, 90, h), fill="#e0a0a0", width=2)
    y = 80
    tops = []
    for i, t in enumerate(lines):
        d.text((120, y), t, fill="#1f2a44", font=font())
        tops.append(y)
        y += 270 if i == para_after else 90
    if smudge is not None:
        box = (100, tops[smudge] - 6, w - 40, tops[smudge] + 66)
        patch = img.crop(box).filter(ImageFilter.GaussianBlur(radius=6))
        img.paste(Image.blend(patch, Image.new("RGB", patch.size, "#ddd8c6"), 0.3), box)
    if clip:
        d.text((120, y), "This last line runs right off the edge of the photo and is cut", fill="#1f2a44", font=font())
    if dim:
        img = ImageEnhance.Brightness(Image.blend(img, Image.new("RGB", (w, h), "#908f8c"), 0.6)).enhance(0.7)
    if tilt:
        img = img.rotate(tilt, resample=Image.BICUBIC, fillcolor="#fffaf0")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return buf.getvalue()


class Script:
    """Hands out scripted reads per line, in call order; None means the model was unsure."""

    def __init__(self, reads, cut=()):
        self.reads = {k: list(v) for k, v in reads.items()}
        self.cut = set(cut)
        self.calls = 0

    def _next(self, i):
        v = self.reads.get(i, [])
        return v.pop(0) if v else None

    def sheet(self, image, n, lang, view, indices):
        self.calls += 1
        return [self._next(i) for _, i in indices]

    def line(self, image, lang, view, index):
        self.calls += 1
        return self._next(index[1])

    def cut_off(self, image, lang, index):
        self.calls += 1
        return index[1] in self.cut


def same(n=3, lines=TRUTH):
    return {i: [t] * n for i, t in enumerate(lines)}


class Detection(unittest.TestCase):
    def test_finds_each_line_on_ruled_paper(self):
        found = photo.detect_lines(photo.load_page(page()))
        self.assertEqual(len(found), 5)
        tops = [f["run"][0] for f in found]
        self.assertEqual(tops, sorted(tops))
        self.assertFalse(any(f["edge"] for f in found))

    def test_dim_photo_still_finds_lines_and_is_flagged(self):
        img = photo.load_page(page(dim=True))
        self.assertEqual(len(photo.detect_lines(img)), 5)
        self.assertIn("low_contrast", photo.quality(img))

    def test_tilted_photo_is_straightened(self):
        for tilt in (-3.0, 2.5):
            img = photo.load_page(page(tilt=tilt))
            self.assertEqual(len(photo.detect_lines(img)), 5, tilt)
        _, angle = photo.deskew(photo.load_page(page()))
        self.assertEqual(angle, 0.0)

    def test_line_running_off_the_edge_is_marked(self):
        found = photo.detect_lines(photo.load_page(page(clip=True)))
        self.assertEqual(len(found), 6)
        self.assertTrue(found[-1]["edge"])
        self.assertFalse(any(f["edge"] for f in found[:-1]))

    def test_blank_page_has_no_lines(self):
        buf = io.BytesIO()
        Image.new("RGB", (1240, 1600), "white").save(buf, "JPEG")
        self.assertEqual(photo.detect_lines(photo.load_page(buf.getvalue())), [])

    def test_smudged_line_is_flagged_not_dropped(self):
        img = photo.load_page(page(smudge=3))
        found = photo.detect_lines(img)
        self.assertEqual(len(found), 4)                     # the smudged line has no clear ink left
        self.assertEqual([f["after"] for f in photo.faint_regions(img, found)], [3])

    def test_paper_noise_and_paragraph_gaps_are_not_faint_writing(self):
        for raw in (page(), page(dim=True), page(tilt=2.5)):
            img = photo.load_page(raw)
            self.assertEqual(photo.faint_regions(img, photo.detect_lines(img)), [])


class Agreement(unittest.TestCase):
    def test_three_matching_reads_are_needed(self):
        self.assertEqual(photo.decide(["a b", "a b", "a b"]), "a b")
        self.assertIsNone(photo.decide(["a b", "a b", "a c"]))       # never a 2-of-3 vote
        self.assertEqual(photo.decide(["a b", "a b", "a c", "a b"]), "a b")
        self.assertIsNone(photo.decide(["a", "a", "a", "b", "b", "b"]))
        self.assertIsNone(photo.decide([None, None, None]))
        self.assertEqual(photo.decide(["a  b", "a b", "a b\n"]), "a b")   # spacing differences are not words

    def test_optional_nukta_is_the_same_word_but_flap_letters_are_not(self):
        with_n, without = "बस्तियों की ज़्यादा गर्मी", "बस्तियों की ज्यादा गर्मी"
        self.assertEqual(photo.decide([with_n, without, without]), without)
        self.assertEqual(photo.decide([with_n, with_n, without]), with_n)
        self.assertIsNone(photo.decide(["और पेड़ गर्मी", "और पेड़ गर्मी", "और पेड गर्मी"]))   # ड़ is its own letter


class Transcription(unittest.TestCase):
    def test_clear_page_reads_every_line(self):
        reader = Script(same())
        tr = photo.transcribe([page()], "en", reader)
        self.assertFalse(tr["blocked"])
        self.assertEqual([x["status"] for x in tr["lines"]], ["unanimous"] * 5)
        self.assertEqual(reader.calls, 4)   # three views, five lines: 1 + 1 + 2 sheet images, no close-ups
        self.assertEqual(tr["text"].count("\n\n"), 1)   # the wider gap after line 2 starts a paragraph
        self.assertEqual(photo.locate([{"id": "s1", "text": TRUTH[0]}, {"id": "s3", "text": TRUTH[2]}], tr),
                         {"s1": ["p1-l1"], "s3": ["p1-l3"]})

    def test_same_misreading_twice_is_corrected_by_close_ups(self):
        # Two views make the same mistake; the close-ups read it right, so the right text wins 3 to 2.
        reads = same()
        wrong = "They release it slowly into the drain."
        reads[1] = [wrong, wrong, TRUTH[1], TRUTH[1], TRUTH[1]]
        tr = photo.transcribe([page()], "en", Script(reads))
        self.assertFalse(tr["blocked"])
        self.assertEqual(tr["lines"][1]["text"], TRUTH[1])
        self.assertEqual(tr["lines"][1]["status"], "confirmed")

    def test_unreadable_line_blocks_grading(self):
        reads = same()
        reads[3] = [None] * 6
        tr = photo.transcribe([page()], "en", Script(reads))
        self.assertTrue(tr["blocked"])
        self.assertEqual(tr["retake"], [{"page": 1, "line": 4, "lineId": "p1-l4", "reason": "unclear"}])
        self.assertNotIn("Community", tr["text"])
        msg = photo.retake_message(tr, "hi")
        self.assertEqual(msg[0]["lines"], [4])
        self.assertTrue(all("\u0900" <= ch <= "\u097f" or not ch.isalpha() for ch in msg[0]["reasons"][0]))

    def test_cut_off_line_blocks_even_when_reads_agree(self):
        lines = TRUTH + ["This last line runs right off the edge of the photo and is"]
        tr = photo.transcribe([page(clip=True)], "en", Script(same(lines=lines), cut={5}))
        self.assertTrue(tr["blocked"])
        self.assertEqual(tr["retake"][0]["reason"], "cut_off")

    def test_smudged_page_asks_for_retake(self):
        reads = {i: [t] * 3 for i, t in enumerate(TRUTH[:3] + TRUTH[4:])}
        tr = photo.transcribe([page(smudge=3)], "en", Script(reads))
        self.assertTrue(tr["blocked"])
        self.assertEqual(tr["retake"][0]["reason"], "faint")

    def test_blank_page_asks_for_retake(self):
        buf = io.BytesIO()
        Image.new("RGB", (1240, 1600), "white").save(buf, "JPEG")
        tr = photo.transcribe([buf.getvalue()], "en", Script({}))
        self.assertTrue(tr["blocked"])
        self.assertEqual(tr["retake"][0]["reason"], "no_text")

    def test_two_pages_continue_the_answer(self):
        tr = photo.transcribe([page(), page()], "en", PagesScript())
        self.assertFalse(tr["blocked"])
        self.assertEqual([x["id"] for x in tr["lines"]][5], "p2-l1")
        self.assertIn("\n\n", tr["text"])


class PagesScript(Script):
    def __init__(self):
        super().__init__({})

    def _next(self, i):
        return TRUTH[i]

    def sheet(self, image, n, lang, view, indices):
        self.calls += 1
        return [TRUTH[i] for _, i in indices]


if __name__ == "__main__":
    unittest.main()
