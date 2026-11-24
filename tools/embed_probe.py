"""Embedding probe: which embedding setup actually works, measured on the Constitution and the Polity test set.

    python tools/embed_probe.py --out probe-out

Measurement only; nothing in grading uses it. v11 found the pinned e5 GGUF giving every passage a cosine of 0.91-0.96,
with Article 44 on top for nearly every query, Hindi included. Three setups are compared on the same passages:

  gguf-current   the v10/v11 file (keisuke-miyako, Q8_0) through llama-cpp-python, mean pooling
  gguf-cstr      another converter's Q8_0 of the same model (cstr), through llama-cpp-python, mean pooling
  onnx-int8      the same model's ONNX export (Xenova, int8) through onnxruntime, with the model's own HF tokenizer:
                 the reference way to run it

For each setup: whether its token ids match the HF tokenizer's, the rank of the settling Article for the 11 planted
errors (English) and 8 Hindi claims and questions, the top cosine for off-syllabus questions against Polity ones, how
often one passage is the top hit (hubness), and seconds to embed every passage. Prints and writes counts and labels
only, never passage text.
"""
import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mains import refs  # noqa: E402
import e2e_qa  # noqa: E402

HF_TOKENIZER = ("Xenova/multilingual-e5-small", "tokenizer.json")
SETUPS = {
    "gguf-current": ("gguf", "keisuke-miyako/multilingual-e5-small-gguf-q8_0", "multilingual-e5-small-Q8_0.gguf",
                     "0d5a5a0b0ad84faad6357a6145e769b0661f0efbf53acf74598afc34dab454f4"),
    "gguf-cstr": ("gguf", "cstr/multilingual-e5-small-GGUF", "multilingual-e5-small-q8_0.gguf",
                  "0a34067a40f25d3149b36885faa62bee0e5284d0f9edc102acfc00e115d953e8"),
    "onnx-int8": ("onnx", "Xenova/multilingual-e5-small", "onnx/model_int8.onnx",
                  "4d24e2bc01a447951524466ef533e52944bf48509e6552810bcee1a2711cb02c"),
}
HINDI = [  # (query, Articles that settle or answer it)
    ("राज्यपाल का चुनाव राज्य की विधानसभा पाँच वर्ष के लिए करती है।", ["155", "156"]),
    ("धन विधेयक संसद के किसी भी सदन में पेश किया जा सकता है।", ["109", "110"]),
    ("पंचायतों के चुनाव भारत का निर्वाचन आयोग कराता है।", ["243K"]),
    ("राष्ट्रपति शासन अनिश्चित काल तक जारी रह सकता है।", ["356"]),
    ("नीति निदेशक तत्वों को मौलिक अधिकारों की तरह अदालत में लागू कराया जा सकता है।", ["37"]),
    ("मुख्य निर्वाचन आयुक्त को प्रधानमंत्री की सलाह पर राष्ट्रपति हटा सकते हैं।", ["324"]),
    ("राज्यपाल की नियुक्ति कैसे होती है? भारत की संघीय व्यवस्था में राज्यपाल की स्थिति का विश्लेषण कीजिए।", ["153", "155", "156"]),
    ("धन विधेयक के पारित होने में राज्यसभा की भूमिका की चर्चा कीजिए।", ["109", "110"]),
]
OFF_TOPIC = [e2e_qa.BASE["question"],
             "Explain the causes of the monsoon in India and its effect on agriculture.",
             "Discuss the role of microfinance in poverty reduction.",
             "शहरी बाढ़ के कारणों की चर्चा कीजिए और उनसे बचाव के उपाय सुझाइए।"]
SAMPLES = ["The Governor of a State shall be appointed by the President.", "राज्यपाल की नियुक्ति राष्ट्रपति करता है।",
           "Panchayats at the village, intermediate and district levels."]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(repo, name, digest):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo, filename=name, local_dir="models/probe")
    return path, sha256(path) == digest


class Gguf:
    def __init__(self, path):
        import llama_cpp
        self.llm = llama_cpp.Llama(model_path=path, embedding=True, n_ctx=512, n_batch=512, n_ubatch=512,
                                   pooling_type=llama_cpp.LLAMA_POOLING_TYPE_MEAN, verbose=False)

    def ids(self, text):
        return self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=True)

    def embed(self, texts):
        return np.asarray([self.llm.embed(t, normalize=True, truncate=True) for t in texts], dtype="float32")


class Onnx:
    def __init__(self, path, tok):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}
        self.tok = tok

    def ids(self, text):
        return self.tok.encode(text).ids

    def embed(self, texts, batch=32):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            n = max(len(e.ids) for e in enc)
            ids = np.zeros((len(enc), n), dtype="int64")
            mask = np.zeros((len(enc), n), dtype="int64")
            for k, e in enumerate(enc):
                ids[k, :len(e.ids)] = e.ids
                mask[k, :len(e.ids)] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.names:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feed)[0]
            pooled = (hidden * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
            out.append(pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9))
        return np.concatenate(out).astype("float32")


def article_of(chunk):
    import re
    m = re.match(r"c(\w+?)(?:\.\d+)?$", chunk["id"])
    return m.group(1).upper() if m else None


def measure(model, chunks, tok):
    texts = ["passage: " + c["title"] + " " + c["text"] for c in chunks]
    started = time.monotonic()
    P = model.embed(texts)
    seconds = round(time.monotonic() - started, 1)
    arts = [article_of(c) for c in chunks]
    const = np.array([a is not None for a in arts])

    def rank(query, wanted):
        q = model.embed(["query: " + query])[0]
        sims = P @ q
        sims_c = np.where(const, sims, -9)
        order = np.argsort(-sims_c)
        best = next((n for n, i in enumerate(order, 1) if arts[i] in wanted), None)
        return {"rank": best, "top": chunks[order[0]]["label"].replace("Constitution of India, ", ""),
                "topCos": round(float(sims_c[order[0]]), 3), "spread": round(float(sims_c[order[0]] - np.median(sims_c[const])), 3)}

    english = []
    for case, v in e2e_qa.POLITY_SET.items():
        sents = {p: next(s for s in v["answer"].split(". ") if p in s) for p, _ in v["planted"]}
        for phrase, wanted in v["planted"]:
            english.append(dict(rank(sents[phrase], {w.upper() for w in wanted}), case=case, phrase=phrase))
    hindi = [dict(rank(q, {w.upper() for w in wanted}), query=q[:40]) for q, wanted in HINDI]
    polity_q = [rank(v["question"], set())["topCos"] for v in e2e_qa.POLITY_SET.values()]
    off = [rank(q, set())["topCos"] for q in OFF_TOPIC]
    tops = collections.Counter(r["top"] for r in english + hindi)
    hf = [tok.encode(s).ids for s in SAMPLES]
    mine = [list(model.ids(s)) for s in SAMPLES]
    return {
        "secondsToEmbedAll": seconds, "passages": len(chunks), "dim": int(P.shape[1]),
        "tokenizerMatchesHF": [a == b for a, b in zip(mine, hf)],
        "englishRanks": [r["rank"] for r in english], "englishTop10": sum(1 for r in english if r["rank"] and r["rank"] <= 10),
        "hindiRanks": [r["rank"] for r in hindi], "hindiTop10": sum(1 for r in hindi if r["rank"] and r["rank"] <= 10),
        "polityQuestionTopCos": polity_q, "offTopicTopCos": off,
        "separation": round(min(polity_q) - max(off), 3),
        "mostCommonTop": tops.most_common(3), "english": english, "hindi": hindi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probe-out")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(hf_hub_download(repo_id=HF_TOKENIZER[0], filename=HF_TOKENIZER[1],
                                              local_dir="models/probe"))
    tok.enable_truncation(512)
    const = json.loads((ROOT / "refs" / "constitution_of_india_2026.json").read_text(encoding="utf-8"))
    chunks = refs.constitution_chunks(const)
    out = {}
    for name, (kind, repo, fname, digest) in SETUPS.items():
        try:
            path, ok = fetch(repo, fname, digest)
            model = Gguf(path) if kind == "gguf" else Onnx(path, tok)
            out[name] = dict(measure(model, chunks, tok), checksumOk=ok)
        except Exception as e:  # noqa: BLE001 - a setup that cannot run is a result too
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        r = out[name]
        print(name, json.dumps({k: r.get(k) for k in ("error", "checksumOk", "tokenizerMatchesHF", "englishTop10",
                                                       "hindiTop10", "separation", "mostCommonTop",
                                                       "secondsToEmbedAll")}, ensure_ascii=False), flush=True)
    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out) / "embed-probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
