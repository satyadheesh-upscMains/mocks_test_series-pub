"""Conservative exact agreement; a disagreement is not silently majority-voted."""
import re
import unicodedata


def normal(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip() if isinstance(text, str) else ""


def agreement(reads, crop=None):
    if len(reads) != 3:
        raise ValueError("Exactly three independent reads are required")
    pairs = [(s, normal(s)) for s in reads if normal(s)]
    for original, text in pairs:
        votes = sum(v == text for _, v in pairs)
        if votes == 3:
            return original.strip(), "unanimous"
        if votes == 2 and normal(crop) == text:
            return original.strip(), "crop-confirmed"
    return "", "needs-clearer-photo"
