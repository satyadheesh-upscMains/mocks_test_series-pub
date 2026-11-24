"""Build refs/index.json.gz for the examiner: Constitution of India (committed JSON) + NCERT Class 11 Political Science.

    python tools/build_refs.py

NCERT chapters come from refs/ncert_pages.json.gz in the private core repo; elsewhere (the public QA repo) they are
downloaded from ncert.nic.in into refs/cache/, never committed. If a
chapter cannot be downloaded, the index is built without it and the run says so: the examiner then simply has fewer
sources, it never invents one. Prints counts only, never passage text.
"""
import json
import gzip
import http.client
import sys
import time
from pathlib import Path
from urllib import request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mains import refs  # noqa: E402

REFS = ROOT / "refs"
CONSTITUTION = REFS / "constitution_of_india_2026.json"
CASES = REFS / "cases.json"
LAXMIKANTH = REFS / "laxmikanth_pages.json.gz"
CACHE = REFS / "cache"
URL = "https://ncert.nic.in/textbook/pdf/{code}.pdf"


def fetch(code):
    path = CACHE / f"{code}.pdf"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    req = request.Request(URL.format(code=code), headers={"User-Agent": "Mozilla/5.0 (mains-examiner refs build)"})
    for attempt in range(3):
        try:
            with request.urlopen(req, timeout=60) as r:
                data = r.read()
            if data[:4] == b"%PDF":
                CACHE.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                return path
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(3 * (attempt + 1))
    return None


LOCAL = REFS / "ncert_pages.json.gz"   # text extracted from the NCERT PDFs; kept in the private core repo only


def ncert_pages():
    # ncert.nic.in refused every download from GitHub's runners (run 45223096235), so the private core repo carries
    # the extracted page text. The public QA repo never has it and falls back to downloading.
    if LOCAL.exists():
        pages = json.loads(gzip.decompress(LOCAL.read_bytes()))
        have = {p["file"] for p in pages}
        return pages, [c for c in refs.NCERT_FILES if c not in have]
    from pypdf import PdfReader
    pages, missing = [], []
    for code in refs.NCERT_FILES:
        path = None if (missing and not pages) else fetch(code)   # the first refusal means the site blocks us
        if not path:
            missing.append(code)
            continue
        for i, page in enumerate(PdfReader(str(path)).pages, 1):
            pages.append({"file": code, "page": i, "text": page.extract_text() or ""})
    return pages, missing


def main():
    chunks = refs.constitution_chunks(json.loads(CONSTITUTION.read_text(encoding="utf-8")))
    if CASES.exists():
        chunks += refs.case_chunks(json.loads(CASES.read_text(encoding="utf-8")))
    if LAXMIKANTH.exists():
        chunks += refs.laxmikanth_chunks(json.loads(gzip.decompress(LAXMIKANTH.read_bytes())))
    n_const = len(chunks)
    try:
        pages, missing = ncert_pages()
    except ImportError:
        pages, missing = [], list(refs.NCERT_FILES)
    chunks += refs.ncert_chunks(pages)
    REFS.mkdir(exist_ok=True)
    out = REFS / "index.json.gz"
    out.write_bytes(gzip.compress(json.dumps(chunks, ensure_ascii=False).encode("utf-8"), 9))
    print(json.dumps({"refs": "built", "constitution_passages": n_const, "ncert_passages": len(chunks) - n_const,
                      "ncert_missing": missing}), flush=True)


if __name__ == "__main__":
    main()
