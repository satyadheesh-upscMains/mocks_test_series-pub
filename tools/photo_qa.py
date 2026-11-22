"""Synthetic handwritten pages on the GitHub runner, read by real Gemma 4 12B vision through mains.photo.

    python tools/photo_qa.py --case clear-en --dir photo-out

Each page is drawn here with a handwriting font from a synthetic student answer, so the true text of every line is
known and every line the pipeline accepts is checked against it. No real student data.
"""
import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mains import photo  # noqa: E402

STUDENTS = {s["id"]: s for s in json.loads((ROOT / "qa" / "students.json").read_text())}


def wrap(text, words):
    w = text.split()
    return [" ".join(w[i:i + words]) for i in range(0, len(w), words)]


EN = wrap(STUDENTS["wetlands-overclaim"]["answer"], 8)
HI = wrap(STUDENTS["heat-strong-hi"]["answer"], 7)
CASES = {
    "clear-en": dict(lines=EN, lang="en", expect="read"),
    "clear-hi": dict(lines=HI, lang="hi", expect="read"),
    "tilted-en": dict(lines=EN, lang="en", tilt=2.5, expect="read"),
    "dim-hi": dict(lines=HI, lang="hi", dim=True, expect="safe"),
    "smudged-en": dict(lines=EN, lang="en", smudge=3, expect="safe"),
    "clipped-en": dict(lines=EN, lang="en", clip=True, expect="block-last"),
}
FONT = "Kalam-Regular.ttf"
PAPER = "#fbf7ee"


def draw(name):
    c, rnd = CASES[name], random.Random(name)
    w, h = 1240, 1754
    img = Image.new("RGB", (w, h), PAPER)
    d = ImageDraw.Draw(img)
    for y in range(150, h, 96):
        d.line((0, y, w, y), fill="#c8d5df", width=2)
    d.line((96, 0, 96, h), fill="#e3a3a3", width=2)
    # Every line must fit the page (except the deliberately clipped one): shrink the pen until it does.
    size = 46
    font = ImageFont.truetype(FONT, size)
    while size > 30 and max(d.textlength(t, font=font) for t in c["lines"]) > w - 190:
        size -= 2
        font = ImageFont.truetype(FONT, size)
    tops = []
    for i, line in enumerate(c["lines"]):
        x, y = 118 + rnd.randint(-6, 6), 88 + 96 * i + rnd.randint(-4, 4)
        if c.get("clip") and i == len(c["lines"]) - 1:
            x = w - round(0.55 * d.textlength(line, font=font))   # the last line runs off the right edge
        d.text((x, y), line, fill=rnd.choice(["#1d2a4a", "#22305a", "#1a2440"]), font=font)
        tops.append(y)
    if c.get("smudge") is not None:
        y = tops[c["smudge"]]
        box = (100, y - 6, w - 40, y + 70)
        patch = img.crop(box).filter(ImageFilter.GaussianBlur(radius=6))
        img.paste(Image.blend(patch, Image.new("RGB", patch.size, "#ddd8c6"), 0.3), box)
    if c.get("dim"):
        img = ImageEnhance.Brightness(Image.blend(img, Image.new("RGB", img.size, "#908f8c"), 0.6)).enhance(0.72)
    if c.get("tilt"):
        img = img.rotate(c["tilt"], resample=Image.BICUBIC, fillcolor=PAPER)
    noise = np.random.default_rng(len(name)).normal(0, 5, (h, w, 1))
    img = Image.fromarray(np.clip(np.asarray(img, dtype=np.float32) + noise, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def gates(name, tr):
    c = CASES[name]
    truth = [photo.canon(t) for t in c["lines"]]
    lines = tr["lines"]
    faint = sum(len(p.get("faint", [])) for p in tr["pages"])
    out = {"every line found or flagged": len(lines) + faint >= len(truth)}
    # An accepted line must be one of the true lines, word for word.
    accepted = [photo.canon(x["text"]) for x in lines if x["text"]]
    wrong = [x["id"] for x in lines if x["text"] and photo.canon(x["text"]) not in truth]
    out["no wrong line accepted"] = not wrong
    exact = sum(1 for t in truth if t in accepted)
    if c["expect"] == "read":
        out["page read without a retake"] = not tr["blocked"] and exact == len(truth)
    elif "smudge" in c:
        out["smudged line read exactly or blocks grading"] = truth[c["smudge"]] in accepted or tr["blocked"]
    elif c["expect"] == "block-last":
        last = lines[-1]["id"] if lines else None
        out["cut-off line blocks grading"] = tr["blocked"] and any(r.get("lineId") == last for r in tr["retake"])
    return out, wrong, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--dir", default="photo-out")
    a = ap.parse_args()
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = draw(a.case)
    (out / f"photo-{a.case}.jpg").write_bytes(raw)
    from mains.model import load_model
    reader = photo.GemmaReader(load_model(vision=True))
    started = time.monotonic()
    tr = photo.transcribe([raw], CASES[a.case]["lang"], reader)
    g, wrong, exact = gates(a.case, tr)
    record = {"case": a.case, "lang": CASES[a.case]["lang"], "truth": CASES[a.case]["lines"], "gates": g,
              "wrongLines": wrong, "exactLines": exact, "transcript": tr,
              "retakeMessage": photo.retake_message(tr, CASES[a.case]["lang"]), "calls": reader.calls,
              "seconds": round(time.monotonic() - started, 1)}
    (out / f"photo-{a.case}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    statuses = {}
    for x in tr["lines"]:
        statuses[x["status"]] = statuses.get(x["status"], 0) + 1
    print(f"{a.case}: {len(tr['lines'])} lines, {exact} exact, statuses {statuses}, blocked={tr['blocked']}, "
          f"{len(reader.calls)} vision calls, {record['seconds']} s", flush=True)
    for k, v in g.items():
        print(f"  {'PASS' if v else 'FAIL'} {k}", flush=True)


if __name__ == "__main__":
    main()
