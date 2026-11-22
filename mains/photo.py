"""Stage: read handwritten answer pages with Gemma 4 12B vision. Nothing is guessed.

    load + quality check -> find the handwritten lines (image processing, no model) -> three reads of every line
    from three views of the page -> extra close-up reads only where the three disagree or were unsure ->
    a line is accepted when at least three reads agree exactly -> any line still unclear blocks grading and the
    student is asked to retake that page.

A line near the edge of the photo is also asked about directly ("is any writing cut off?"), because three reads of a
clipped line can agree on the clipped text. Logs carry stage, view, timing and status, never what the student wrote.
"""
import base64
import io
import json
import re
import time
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .model import log

MAX_SIDE = 2400          # long side after loading: enough for handwriting, bounded memory on the runner
MAX_LINES = 45           # per page
SHEET_VIEWS = (("original", 6, 80), ("contrast", 6, 80), ("zoom", 3, 140))  # view, lines per image, strip height
FOCUS_VIEWS = ("sharp", "binary", "large")   # extra close-up reads of one line, in this order
MAX_FOCUS_PER_PAGE = 30
VOTES = 3                # reads that must agree on the exact text


# ---------------------------------------------------------------- page

def load_page(raw):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    scale = MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return deskew(img)[0]


def _skew_score(mask, angle):
    rows = np.asarray(mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)).sum(axis=1).astype(np.float64)
    return float(rows.var())


def deskew(img):
    """Straighten a tilted photo: the angle whose ink rows are sharpest (writing and ruled lines line up)."""
    small = img.convert("L")
    scale = 800 / max(small.size)
    if scale < 1:
        small = small.resize((max(1, round(small.width * scale)), max(1, round(small.height * scale))))
    g = np.asarray(ImageOps.autocontrast(small, cutoff=1))
    mask = Image.fromarray(((g <= _otsu(g)) * 255).astype(np.uint8))
    best = max(np.arange(-6.0, 6.01, 0.5), key=lambda a: _skew_score(mask, a))
    best = max(np.arange(best - 0.4, best + 0.41, 0.1), key=lambda a: _skew_score(mask, a))
    angle = round(float(best), 1)
    if abs(angle) < 0.3:
        return img, 0.0
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255)), angle


def quality(img):
    """Page-level hints. Only too_small blocks on its own; the others make the retake message specific."""
    gray = img.convert("L")
    g = np.asarray(gray, dtype=np.float32)
    issues = []
    if min(img.size) < 500:
        issues.append("too_small")
    if g.mean() < 70:
        issues.append("too_dark")
    if g.std() < 22:
        issues.append("low_contrast")
    s = np.asarray(gray.resize((max(3, img.width // 2), max(3, img.height // 2))), dtype=np.float32)
    lap = 4 * s[1:-1, 1:-1] - s[:-2, 1:-1] - s[2:, 1:-1] - s[1:-1, :-2] - s[1:-1, 2:]
    if lap.var() < 40:
        issues.append("blurry")
    return issues


def _otsu(g):
    hist = np.bincount(np.asarray(g, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    total, levels = hist.sum(), np.arange(256)
    w0 = np.cumsum(hist)
    m0 = np.cumsum(hist * levels)
    w1 = total - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (m0[-1] * w0 - m0 * total) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = 0
    return int(np.argmax(between))


def _drop_rules(mask, thickness=4):
    """Remove ruled lines and the margin line: thin runs of rows (or columns) that are dark across most of the page.
    A thicker dark band is not ruling (a smudge, a shadow) and is kept."""
    for axis in (1, 0):
        full = mask.mean(axis=axis) > 0.5
        i, n = 0, len(full)
        while i < n:
            if full[i]:
                j = i
                while j < n and full[j]:
                    j += 1
                if j - i <= thickness:
                    if axis == 1:
                        mask[i:j, :] = False
                    else:
                        mask[:, i:j] = False
                i = j
            else:
                i += 1
    return mask


def ink_mask(img):
    g = np.asarray(ImageOps.autocontrast(img.convert("L"), cutoff=1))
    # Ruled lines and the margin line span most of the page; handwriting never does in a single row or column.
    return _drop_rules(g <= _otsu(g))


def detect_lines(img):
    """Handwritten lines as boxes, top to bottom, from the row profile of the ink."""
    ink = ink_mask(img)
    h, w = ink.shape
    rows = ink.sum(axis=1).astype(np.float32)
    k = max(3, h // 200)
    rows = np.convolve(rows, np.ones(k) / k, mode="same")
    if rows.max() <= 0:
        return []
    on = rows > max(2.0, 0.06 * float(rows.max()))
    runs, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            runs.append([start, y])
            start = None
    if start is not None:
        runs.append([start, h])
    if not runs:
        return []
    med = float(np.median([b - a for a, b in runs]))
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] < 0.35 * med:   # a gap smaller than a third of a line splits descenders, not lines
            merged[-1][1] = b
        else:
            merged.append([a, b])
    med = float(np.median([b - a for a, b in merged]))
    out = []
    for a, b in merged:
        if b - a < 0.35 * med:
            continue
        band = ink[a:b]
        if band.sum() < 0.003 * (b - a) * w:
            continue
        cols = np.where(band.sum(axis=0) > 0)[0]
        pad = round(0.25 * (b - a))
        x0, x1 = max(0, int(cols[0]) - pad), min(w, int(cols[-1]) + 1 + pad)
        y0, y1 = max(0, a - pad), min(h, b + pad)
        # Ink within about one letter-gap of a side, or a line touching the top or bottom, may be cut off.
        tol = max(6, round(0.012 * w))
        edge = cols[0] <= tol or cols[-1] >= w - 1 - tol or a <= 1 or b >= h - 1
        out.append({"px": (x0, y0, x1, y1), "run": (a, b), "edge": bool(edge),
                    "box": {"x": round(x0 / w, 4), "y": round(y0 / h, 4),
                            "w": round((x1 - x0) / w, 4), "h": round((y1 - y0) / h, 4)}})
        if len(out) == MAX_LINES:
            break
    return out


def faint_regions(img, lines):
    """Marks between two detected lines that are too faint to count as writing (a smudge, a pencil line, a wet
    patch). A line lost this way must not silently drop out of the answer, so each one asks for a retake."""
    if len(lines) < 2:
        return []
    # A median filter removes camera noise (single dark pixels) but keeps smudged strokes, which are smooth.
    g = np.asarray(ImageOps.autocontrast(img.convert("L"), cutoff=1).filter(ImageFilter.MedianFilter(5)),
                   dtype=np.int16)
    paper = int(np.median(g))
    light = g[g >= np.percentile(g, 25)]
    sigma = 1.4826 * float(np.median(np.abs(light - np.median(light))))
    dark = _drop_rules(g < paper - max(30.0, 5 * sigma))
    height = float(np.median([ln["run"][1] - ln["run"][0] for ln in lines]))
    out = []
    for i in range(len(lines) - 1):
        a, b = lines[i]["px"][3], lines[i + 1]["px"][1]
        if b - a < 0.6 * height:
            continue
        band = dark[a:b]
        rows = band.sum(axis=1)
        # a real lost line leaves a run of rows with scattered marks, not a few specks of noise
        busy = rows > 0.02 * band.shape[1]
        if busy.sum() >= 0.4 * height:
            out.append({"after": i + 1, "y": round(a / img.height, 4), "h": round((b - a) / img.height, 4)})
    return out


# ---------------------------------------------------------------- views

def jpeg(img, quality_=92):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality_)
    return buf.getvalue()


def _fit(img, height, max_w=1800):
    scale = height / max(1, img.height)
    w = max(1, round(img.width * scale))
    if w > max_w:
        scale = max_w / img.width
        w = max_w
    return img.resize((w, max(1, round(img.height * scale))), Image.LANCZOS)


def _view(crop, view):
    if view in ("contrast", "sharp"):
        crop = ImageOps.autocontrast(crop.convert("L"), cutoff=1)
        if view == "sharp":
            crop = crop.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
    elif view == "binary":
        g = crop.convert("L")
        t = _otsu(np.asarray(g))
        crop = g.point(lambda v: 0 if v <= t else 255)
    return crop.convert("RGB")


def line_image(img, line, view):
    x0, y0, x1, y1 = line["px"]
    if view == "large":   # some context above and below; the prompt says to read only the middle line
        pad = round(0.4 * (y1 - y0))
        y0, y1 = max(0, y0 - pad), min(img.height, y1 + pad)
    return _fit(_view(img.crop((x0, y0, x1, y1)), view), 200 if view == "large" else 140)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_sheet(img, lines, view, strip_h):
    """Stack the line strips into one image, labelled 1..n in a grey box at the left."""
    strips = [_fit(_view(img.crop(ln["px"]), view), strip_h) for ln in lines]
    label_w, gap = 70, 18
    width = label_w + max(s.width for s in strips) + 10
    height = gap + sum(s.height + gap for s in strips)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(max(18, strip_h // 3))
    y = gap
    for n, s in enumerate(strips, 1):
        draw.rectangle((6, y, label_w - 10, y + s.height), fill=(232, 232, 232))
        draw.text((18, y + s.height // 2 - strip_h // 6), str(n), fill=(110, 110, 110), font=font)
        sheet.paste(s, (label_w, y))
        y += s.height + gap
    return sheet


# ---------------------------------------------------------------- agreement

def canon(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


# A nukta under क ख ग ज फ is an optional spelling variant in Hindi (ज़्यादा / ज्यादा, फ़ैसला / फैसला): same word, same
# meaning, written either way. Reads that differ only there agree. ड़ and ढ़ are different letters and stay exact.
_OPTIONAL_NUKTA = re.compile("([\u0915\u0916\u0917\u091c\u092b])\u093c")


def same_words(text):
    return _OPTIONAL_NUKTA.sub(r"\1", unicodedata.normalize("NFD", canon(text)))


def decide(reads):
    """reads: texts, None for an unsure read. Accept a text read the same way at least VOTES times and more often
    than any other text. Never a 2-of-3 majority: disagreement means more reads, not a vote. The accepted text is
    the spelling most reads used."""
    groups = {}
    for r in reads:
        c = canon(r)
        if c:
            g = groups.setdefault(same_words(c), {})
            g[c] = g.get(c, 0) + 1
    if not groups:
        return None
    ranked = sorted(groups.values(), key=lambda g: -sum(g.values()))
    n = sum(ranked[0].values())
    if n >= VOTES and (len(ranked) == 1 or n > sum(ranked[1].values())):
        return max(ranked[0].items(), key=lambda kv: kv[1])[0]
    return None


# ---------------------------------------------------------------- the model

def _hi(lang):
    return " It is mostly Hindi in Devanagari script and may contain English words or acronyms." if lang == "hi" else ""


RULES = ("Copy the handwriting literally, in the script it is written in. Keep spelling, numbers, punctuation and "
         "every vowel sign, nasal mark and nukta exactly as written. Do not correct, translate, complete or guess. ")


class GemmaReader:
    """The three kinds of vision question the transcription asks. Returns plain values; logs no content."""

    def __init__(self, llm):
        self.llm = llm
        self.calls = []

    def _ask(self, image, instruction, schema, max_tokens, view):
        started = time.monotonic()
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg(image)).decode()}},
                {"type": "text", "text": instruction}]}],
            response_format={"type": "json_object", "schema": schema}, temperature=0, max_tokens=max_tokens)
        choice = response["choices"][0]
        out = None
        if choice.get("finish_reason") != "length":
            try:
                out = json.loads(choice["message"]["content"])
            except (TypeError, ValueError, KeyError):
                out = None
        meta = dict(stage="transcribe", check=view, seconds=round(time.monotonic() - started, 1),
                    status="ok" if isinstance(out, dict) else "rejected")
        self.calls.append(meta)
        log(**meta)
        return out if isinstance(out, dict) else None

    @staticmethod
    def _clean(text, unclear):
        t = canon(text) if isinstance(text, str) else ""
        if unclear is not False or not t or len(t) > 300 or re.fullmatch(r"[\d\s.)(]+", t):
            return None
        return t

    def sheet(self, image, n, lang, view, indices=None):
        schema = {"type": "object", "additionalProperties": False, "required": ["lines"], "properties": {"lines": {
            "type": "array", "minItems": n, "maxItems": n, "items": {
                "type": "object", "additionalProperties": False, "required": ["line", "text", "unclear"],
                "properties": {"line": {"type": "integer", "enum": list(range(1, n + 1))},
                               "text": {"type": "string"}, "unclear": {"type": "boolean"}}}}}}
        instruction = (f"This image shows {n} strips cut from a student's handwritten exam answer, one handwritten line "
                       f"per strip, labelled 1 to {n} in a grey box at the left.{_hi(lang)} For each label, give the "
                       "handwritten text of that strip. " + RULES + "Do not copy the grey label. If a strip is blank, "
                       "cut off or smudged, or you are not sure of any word in it, return unclear=true and empty text.")
        out = self._ask(image, instruction, schema, 120 + 90 * n, view)
        got = {}
        for row in (out or {}).get("lines", []):
            if isinstance(row, dict) and row.get("line") in range(1, n + 1):
                got[row["line"]] = self._clean(row.get("text"), row.get("unclear"))
        return [got.get(i) for i in range(1, n + 1)]

    def line(self, image, lang, view, index=None):
        schema = {"type": "object", "additionalProperties": False, "required": ["text", "unclear"],
                  "properties": {"text": {"type": "string"}, "unclear": {"type": "boolean"}}}
        where = (" Read only the middle line; ignore partial lines above and below it." if view == "large" else "")
        instruction = (f"This close-up shows one handwritten line from a student's exam answer.{where}{_hi(lang)} "
                       + RULES + "Check the last word letter by letter. If you are not sure of any word, or the line "
                       "is cut off, return unclear=true and empty text.")
        out = self._ask(image, instruction, schema, 220, view) or {}
        return self._clean(out.get("text"), out.get("unclear"))

    def cut_off(self, image, lang, index=None):
        schema = {"type": "object", "additionalProperties": False, "required": ["cutOff"],
                  "properties": {"cutOff": {"type": "boolean"}}}
        out = self._ask(image, "This close-up shows one handwritten line that reaches the edge of the photo. Is any "
                        "part of the writing cut off by the edge, so that a letter, a word or the end of the line is "
                        "missing? Answer cutOff true or false.", schema, 40, "edge")
        return True if out is None else bool(out.get("cutOff"))


# ---------------------------------------------------------------- transcription

def _join(prev, nxt):
    if prev.endswith("-") and nxt[:1].islower():
        return prev[:-1], ""
    return prev, " "


def transcribe(pages, lang, reader):
    """pages: raw image bytes. Returns the transcript, the line boxes and what blocks grading."""
    page_info, lines, retake, evidence = [], [], [], []
    text, offsets = "", []
    for pno, raw in enumerate(pages, 1):
        started = time.monotonic()
        img = load_page(raw)
        q = quality(img)
        found = [] if "too_small" in q else detect_lines(img)
        faint = faint_regions(img, found)
        page_info.append({"page": pno, "width": img.width, "height": img.height, "issues": q, "lines": len(found),
                          "faint": faint})
        if not found:
            retake.append({"page": pno, "line": None, "reason": "too_small" if "too_small" in q else "no_text"})
            continue
        for f in faint:
            retake.append({"page": pno, "line": None, "after": f["after"], "reason": "faint"})
        reads = [[] for _ in found]
        for view, per, strip_h in SHEET_VIEWS:
            for at in range(0, len(found), per):
                chunk = list(range(at, min(at + per, len(found))))
                got = reader.sheet(make_sheet(img, [found[i] for i in chunk], view, strip_h), len(chunk), lang, view,
                                   [(pno, i) for i in chunk])
                for i, value in zip(chunk, got):
                    reads[i].append(value)
        budget = MAX_FOCUS_PER_PAGE
        gaps = [found[i + 1]["run"][0] - found[i]["run"][1] for i in range(len(found) - 1)]
        para_gap = 1.6 * float(np.median(gaps)) if gaps else None
        for i, ln in enumerate(found):
            result = decide(reads[i])
            status = "unanimous" if result is not None else None
            for view in FOCUS_VIEWS:
                if result is not None or budget <= 0:
                    break
                reads[i].append(reader.line(line_image(img, ln, view), lang, view, (pno, i)))
                budget -= 1
                result = decide(reads[i])
                status = "confirmed" if result is not None else None
            reason = None if result is not None else "unclear"
            if result is not None and ln["edge"] and reader.cut_off(line_image(img, ln, "large"), lang, (pno, i)):
                result, status, reason = None, None, "cut_off"
            lid = f"p{pno}-l{i + 1}"
            lines.append({"id": lid, "page": pno, "box": ln["box"], "text": result or "",
                          "status": status or "unclear", "reads": len(reads[i]), "edge": ln["edge"]})
            evidence.append({"lineId": lid, "reads": [r if r is not None else None for r in reads[i]]})
            if reason:
                retake.append({"page": pno, "line": i + 1, "lineId": lid, "reason": reason})
                continue
            if text:
                new_para = (i == 0) or (para_gap is not None and i > 0 and gaps[i - 1] > para_gap)
                if new_para:
                    text += "\n\n"
                else:
                    text, sep = _join(text, result)
                    text += sep
            offsets.append({"lineId": lid, "start": len(text), "end": len(text) + len(result)})
            text += result
        log(stage="transcribe", check=f"page{pno}", lines=len(found),
            seconds=round(time.monotonic() - started, 1), status="ok")
    return {"text": text, "pages": page_info, "lines": lines, "offsets": offsets,
            "blocked": bool(retake), "retake": retake, "evidence": evidence}


def locate(sentences, transcript):
    """Which handwritten lines each answer sentence came from, for the marked page in the report."""
    text, offsets = transcript["text"], transcript["offsets"]
    where, cursor = {}, 0
    for s in sentences:
        at = text.find(s["text"], cursor)
        if at < 0:
            continue
        end = at + len(s["text"])
        cursor = end
        where[s["id"]] = [o["lineId"] for o in offsets if o["start"] < end and at < o["end"]]
    return where


RETAKE = {
    "en": {"unclear": "We could not read this line with certainty.",
           "cut_off": "Part of this line is cut off at the edge of the photo.",
           "faint": "Some writing on this page is too faint or smudged to read.",
           "no_text": "We could not find handwriting on this page.",
           "too_small": "This photo is too small to read.",
           "too_dark": "take the photo in better light", "low_contrast": "use a darker pen or brighter light",
           "blurry": "hold the phone steady and let it focus"},
    "hi": {"unclear": "यह पंक्ति हम पक्के तौर पर नहीं पढ़ पाए।",
           "cut_off": "इस पंक्ति का कुछ हिस्सा फ़ोटो के किनारे पर कट गया है।",
           "faint": "इस पेज पर कुछ लिखावट बहुत हल्की या धुँधली है, पढ़ी नहीं जा सकी।",
           "no_text": "इस पेज पर लिखावट नहीं मिली।",
           "too_small": "यह फ़ोटो पढ़ने के लिए बहुत छोटी है।",
           "too_dark": "अच्छी रोशनी में फ़ोटो लीजिए", "low_contrast": "गहरे रंग का पेन या ज़्यादा रोशनी इस्तेमाल कीजिए",
           "blurry": "फ़ोन स्थिर रखिए और फ़ोकस होने दीजिए"},
}


def retake_message(transcript, lang):
    """Plain-language request for the student, page by page. Fixed text only."""
    t = RETAKE[lang]
    by_page = {}
    for r in transcript["retake"]:
        by_page.setdefault(r["page"], []).append(r)
    out = []
    for page, items in sorted(by_page.items()):
        info = next(p for p in transcript["pages"] if p["page"] == page)
        tips = [t[i] for i in info["issues"] if i in ("too_dark", "low_contrast", "blurry")]
        out.append({"page": page, "lines": [r["line"] for r in items if r["line"]],
                    "reasons": sorted({t[r["reason"]] for r in items}), "tips": tips})
    return out
