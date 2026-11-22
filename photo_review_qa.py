"""Synthetic scans only. Exercises real Gemma vision, a text review and source-line anchors."""
import base64
import io
import json
import os
from pathlib import Path
import re
import time
import unicodedata

from huggingface_hub import hf_hub_download
from llama_cpp.llama_chat_format import Gemma4ChatHandler
from PIL import Image, ImageDraw, ImageFont

from grader import MODEL_REPO, MODEL_FILE, evaluate, load_model

PROJECTOR = "mmproj-F32.gguf"
CASES = {
    "en": {
        "question": "Discuss how urban wetlands reduce flood risk and suggest ways to conserve them.",
        "references": ["Store rainfall and slow runoff", "Restoring inlets and preserving vegetation"],
        "lines": [
            "Urban wetlands store stormwater.",
            "They slow runoff into city drains.",
            "Restoration reduced floods by 80% in every city.",
            "Cities should protect wetland boundaries.",
            "Blocked inlets should be restored.",
        ],
    },
    "hi": {
        "question": "समझाइए कि शहरी आर्द्रभूमि बाढ़ का जोखिम कैसे कम करती हैं। इनके संरक्षण के उपाय बताइए।",
        "references": ["बारिश का पानी जमा करना", "जलमार्ग बहाल करना और अतिक्रमण रोकना"],
        "lines": [
            "शहरी आर्द्रभूमि बारिश का पानी जमा करती हैं।",
            "इससे नालों में पानी धीरे पहुँचता है।",
            "हर शहर में बाढ़ 80% कम हो गई है।",
            "नगर निकाय अतिक्रमण रोक सकते हैं।",
            "बंद जलमार्ग फिर खोलने चाहिए।",
        ],
    },
}
FORMAT = {
    "type": "object", "additionalProperties": False, "required": ["lines"],
    "properties": {"lines": {
        "type": "array", "minItems": 5, "maxItems": 5,
        "items": {"type": "object", "additionalProperties": False, "required": ["line", "text", "unclear"],
                  "properties": {"line": {"type": "integer"}, "text": {"type": "string"},
                                 "unclear": {"type": "boolean"}}}
    }},
}


def scan(case, obscured=False):
    # Ground-truth layout for this generated fixture only; not a general handwriting locator.
    w, h = 1240, 750
    image = Image.new("RGB", (w, h), "#fffaf0")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Kalam-Regular.ttf", 48)
    label = ImageFont.truetype("Kalam-Regular.ttf", 36)
    boxes = {}
    for i, line in enumerate(case["lines"], 1):
        y = 28 + i * 122
        draw.line((60, y + 70, w - 45, y + 70), fill="#d9e3e8", width=2)
        draw.text((65, y), f"{i}.", fill="#404352", font=label)
        draw.text((150, y - 4), line, fill="#202f4a", font=font)
        boxes[i] = {"x": 150/w, "y": (y-8)/h, "w": 990/w, "h": 76/h}
        if obscured and i == 3:
            draw.rectangle((145, y - 6, w - 45, y + 68), fill="#191b20")
    raw = io.BytesIO()
    image.save(raw, format="JPEG", quality=90, optimize=True)
    return raw.getvalue(), boxes


def canonical(s):
    return re.sub(r"[^\w\s%\u0900-\u097f]", "", unicodedata.normalize("NFC", s).casefold()).split()


def transcribe(model, photo, instruction, schema, max_tokens):
    choice = model.create_chat_completion(
        messages=[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(photo).decode()}},
            {"type":"text","text":instruction},
        ]}], response_format={"type":"json_object","schema":schema}, temperature=0, max_tokens=max_tokens,
    )["choices"][0]
    assert choice["finish_reason"] != "length"
    return json.loads(choice["message"]["content"])


def main():
    name = os.environ["QA_PHOTO"]
    obscured = name == "unreadable"
    lang = "en" if obscured else name
    case = CASES[lang]
    photo, boxes = scan(case, obscured)
    Path(f"sample-{name}.jpg").write_bytes(photo)
    proj = hf_hub_download(repo_id=MODEL_REPO, filename=PROJECTOR, local_dir="models")
    model = load_model(Gemma4ChatHandler(clip_model_path=proj, verbose=False, use_gpu=False))
    assert model.metadata.get("general.architecture", "").startswith("gemma")
    print(f"Loaded Gemma 4 12B vision: {MODEL_FILE} + {PROJECTOR}", flush=True)
    start = time.monotonic()
    instruction = (
        "Transcribe exactly the five numbered lines in this synthetic answer page. Preserve words, numbers, and language. "
        "Do not correct or translate the student. Omit the printed line numbers from the text field. "
        "For any obscured line return its number, text as an empty string, and unclear true. Never guess illegible text."
    )
    raw_lines = transcribe(model, photo, instruction, FORMAT, 850)["lines"]
    assert sorted(r["line"] for r in raw_lines) == list(boxes)
    lines = [{"id":f"p1-l{r['line']}", "page":1, "box":boxes[r["line"]], "text":r["text"].strip(), "unclear":r["unclear"]}
             for r in sorted(raw_lines,key=lambda r:r["line"])]
    out = {"model": MODEL_FILE, "projector": PROJECTOR, "case": name, "lang": lang, "question": case["question"],
           "expectedLines": case["lines"], "lines": lines, "ocrSeconds": round(time.monotonic()-start, 1)}
    report_path = Path(f"photo-report-{name}.json")
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    if obscured:
        assert lines[2]["unclear"] and not lines[2]["text"], "Guessed the opaque line"
        out["blockedBeforeGrading"] = True
    else:
        if name == "hi":
            # A fixture-only probe, not a production OCR correction rule. Inspect the known troublesome region
            # blind: give the crop and instructions, never the reference transcription or prior response.
            image = Image.open(io.BytesIO(photo))
            b = boxes[3]
            crop = image.crop((145,int(b["y"]*image.height)-8, image.width-40, int((b["y"]+b["h"])*image.height)+8))
            raw = io.BytesIO()
            crop.save(raw, "JPEG", quality=95)
            precise = transcribe(model, raw.getvalue(),
                "Copy the single Devanagari line exactly, including the number. Inspect the final word letter by letter. Do not fix spelling, do not add or remove vowel or nasal marks. If uncertain return unclear=true and empty text.",
                {"type":"object","additionalProperties":False,"required":["text","unclear"],
                 "properties":{"text":{"type":"string"},"unclear":{"type":"boolean"}}}, 220)
            out["wholePageLine3"] = dict(lines[2])
            lines[2] = {**lines[2], "text":precise["text"].strip(), "unclear":precise["unclear"]}
        exact = sum(not x["unclear"] and canonical(x["text"]) == canonical(t) for x,t in zip(lines,case["lines"]))
        out["exactLines"] = exact
        report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        assert exact == 5, "Visible lines did not all match the known fixture text"
        start = time.monotonic()
        answer = "\n".join(x["text"] for x in lines)
        report = evaluate(model, {**case, "lang":lang, "answer":answer})
        critique = report["gap"]+" "+report["nextStep"]
        # This single visual anchor is mechanically tied to the numerical claim, never a generated coordinate.
        out["feedback"] = report
        out["gradeSeconds"] = round(time.monotonic()-start, 1)
        message = next((report[k] for k in ("gap", "nextStep") if "80" in report[k]), None)
        if message and "80%" in lines[2]["text"]:
            out["marks"] = [{"lineId":lines[2]["id"],"message":message}]
        report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        assert out.get("marks"), "Feedback missed the unsupported numerical claim"
    report_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"{name}: real transcription{' + feedback and annotation' if not obscured else '; correctly blocked before grading'} passed.", flush=True)


if __name__ == "__main__":
    main()
