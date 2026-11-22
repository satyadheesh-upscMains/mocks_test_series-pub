"""Three stateless vision reads per synthetic page, then a separate line-crop retry if needed."""
import io
import json
import os
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from huggingface_hub import hf_hub_download
from llama_cpp.llama_chat_format import Gemma4ChatHandler

from grader import MODEL_FILE, MODEL_REPO, load_model
from ocr_agreement import agreement, normal
from photo_review_qa import CASES, FORMAT, PROJECTOR, scan, transcribe

TRIALS = {"clear-en":"en", "dim-hi":"hi", "smudged-en":"en", "clipped-en":"en"}
VIEW_IDS = ["original", "contrast", "zoom"]
LINE_SCHEMA={"type":"object","additionalProperties":False,"required":["text","unclear"],
             "properties":{"text":{"type":"string"},"unclear":{"type":"boolean"}}}


def jpeg(pic):
    f=io.BytesIO()
    pic.convert("RGB").save(f,"JPEG",quality=92)
    return f.getvalue()


def region(pic, box, labels=False):
    w,h=pic.size
    return (30 if labels else max(0,int(box["x"]*w)-5),
            max(0,int(box["y"]*h)-3), w-24, min(h,int((box["y"]+box["h"])*h)+4))


def make_trial(name, case):
    raw, boxes=scan(case)
    pic=Image.open(io.BytesIO(raw)).convert("RGB")
    if name == "dim-hi":
        pic=Image.blend(pic,Image.new("RGB",pic.size,"#908f8c"),.61)
        pic=ImageEnhance.Brightness(pic).enhance(.72)
    if name == "smudged-en":
        area=region(pic,boxes[3])
        patch=pic.crop(area).filter(ImageFilter.GaussianBlur(radius=6))
        patch=Image.blend(patch,Image.new("RGB",patch.size,"#dddac8"),.31)
        pic.paste(patch,area)
    if name == "clipped-en":
        # Simulate a page corner clipped out of the shot. No source pixels remain for line 3.
        a=region(pic,boxes[3]); ImageDraw.Draw(pic).rectangle((a[0]-12,a[1],pic.width,a[3]),fill="#fffaf0")
    return pic,boxes


def zoom_fixture(pic, boxes):
    # This QA fixture supplies line bounds; no claim of production layout recognition.
    strips=[pic.crop(region(pic,b,labels=True)) for b in boxes.values()]
    out=Image.new("RGB",(strips[0].width,sum(p.height for p in strips)+24*len(strips)),"#fffaf0")
    y=12
    for strip in strips:
        out.paste(strip,(0,y));y+=strip.height+24
    return out


def read(row):
    value=row["text"]
    return value.strip() if row["unclear"] is False and isinstance(value,str) and 1 <= len(value.strip()) <= 300 else None


def main():
    name=os.environ["QA_PHOTO"]
    lang=TRIALS[name]
    case=CASES[lang]
    pic,boxes=make_trial(name,case)
    Path(f"consensus-sample-{name}.jpg").write_bytes(jpeg(pic))
    projector=hf_hub_download(repo_id=MODEL_REPO,filename=PROJECTOR,local_dir="models")
    model=load_model(Gemma4ChatHandler(clip_model_path=projector,verbose=False,use_gpu=False))
    assert Path(model.model_path).name==MODEL_FILE and model.metadata["general.architecture"].startswith("gemma")
    print(f"Gemma vision: {MODEL_FILE} + {PROJECTOR}",flush=True)
    t=time.monotonic()
    contrast=ImageOps.autocontrast(ImageOps.grayscale(pic),cutoff=1).convert("RGB")
    images=[pic,contrast,zoom_fixture(pic,boxes)]
    votes=[]
    for method,image in zip(VIEW_IDS,images):
        s=time.monotonic()
        rows=transcribe(model,jpeg(image),
          "Copy the five numbered handwritten lines literally. Do not copy the printed line numbers into text. "
          "Inspect every word including the last. Preserve the exact script, punctuation, numbers, spelling and vowel or nasal marks. "
          "Do not correct, translate, infer, or complete missing text. For a partly cropped, blank, smudged or uncertain line, "
          "return its numbered slot with unclear=true and text=''. Never guess.",FORMAT,900)["lines"]
        assert sorted(x["line"] for x in rows)==list(boxes), "Missing OCR line slots"
        votes.append({x["line"]:read(x) for x in rows})
        print(f"{method}: {time.monotonic()-s:.1f}s",flush=True)

    lines=[]; evidence=[]; statuses={}; retry_count=0
    for n,b in boxes.items():
        reads=[v[n] for v in votes]
        text,status=agreement(reads)
        crop=None
        if not text and retry_count<3 and any(reads):
            focused=contrast.crop(region(pic,b))
            focused=ImageOps.expand(focused,border=(6,24),fill="#fffaf0")
            crop=read(transcribe(model,jpeg(focused),
                "Copy the single line from this close-up. Preserve each glyph, exact words, numbers and punctuation, "
                "and inspect the final word letter by letter. Do not correct spelling. If incomplete or unreadable, return unclear=true and empty text.",LINE_SCHEMA,240))
            retry_count+=1
            text,status=agreement(reads,crop)
        key=f"p1-l{n}"
        lines.append({"id":key,"page":1,"box":b,"text":text,"unclear":not text})
        evidence.append({"lineId":key,"reads":reads,"crop":crop})
        statuses[key]=status

    incorrect=[i+1 for i,(r,expected) in enumerate(zip(lines,case["lines"])) if r["text"] and normal(r["text"])!=normal(expected)]
    blocked=any(x["unclear"] for x in lines)
    proof={"model":MODEL_FILE,"case":name,"lang":lang,"question":case["question"],"expectedLines":case["lines"],
           "lines":lines,"ocr":{"views":VIEW_IDS,"evidence":evidence},"decisions":statuses,"cropRetries":retry_count,
           "blockedBeforeGrading":blocked,"incorrectAcceptedLines":incorrect,"ocrSeconds":round(time.monotonic()-t,1)}
    output=Path(f"consensus-{name}.json")
    output.write_text(json.dumps(proof,ensure_ascii=False,indent=2))
    assert not incorrect, "Agreement accepted a wrong line"
    if name=="clear-en": assert not blocked, "Clear English control was blocked"
    if name=="clipped-en": assert lines[2]["unclear"] and blocked, "Missing line was allowed through"
    proof["fixtureGatePassed"]=True
    output.write_text(json.dumps(proof,ensure_ascii=False,indent=2))
    print(f"{name}: {'needs clearer photo' if blocked else 'ready for grading'}; {retry_count} crop retries",flush=True)


if __name__=="__main__":
    main()
