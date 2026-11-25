"""Stage: build the student report from validated findings. Code, not Gemma, decides what goes where.

Gemma's words appear only in fields that passed validation (reasons, fixes, rewrites, notes, the optional summary).
Everything structural is computed: which part is answered, the overall level, the order of fixes, the colour of each
highlight, the headline, word counts and the command-word guide. Duplicates across checks are merged.
"""
import json
import re

from .model import MODEL_FILE
from .rubric import (DIM, DIMENSIONS, GROUPS, HINDI_ALLOWANCE, IMPACT_LABEL, LEVEL_LABEL, LEVELS, PART_LABEL, VERDICT_LABEL,
                     directive_guide)
from .text import Rejected, clean, sentence_count, similar, word_count
from . import prompts

REPORT_VERSION = "mains-report-v15"
MAX_UNJUDGED = 2  # checks that may fail to be judged; beyond this the report is not trustworthy and is retried
PRI = {"major": 0, "medium": 1, "minor": 2}
KIND = {"wrong": 0, "missing": 1, "unsupported": 2, "weak": 3}
KIND_LABEL = {"wrong": ("Wrong", "गलत"), "unsupported": ("Unsupported", "बिना आधार"),
              "weak": ("Needs work", "सुधार चाहिए"), "missing": ("Missing", "छूटा")}


# A sentence is "wrong" only for what it says. Something it leaves out is a gap: amber, never red (v9 Money Bill:
# "After the Lok Sabha passes it, the Bill is sent to the Rajya Sabha" was red for "fails to mention" the 14 days).
OMISSION = re.compile(r"\b(fails? to (mention|state|specify|include|note|explain)|does(n't| not) (mention|state|specify|"
                      r"include|explain|say)|did not (mention|state|specify)|without (mentioning|specifying|stating)|"
                      r"omits?|leaves? out|should (also )?(mention|specify|state|include)|is silent on|"
                      r"no mention of|not mention(ed)?)\b", re.I)
# Wording that differs from the Constitution's but means the same is not a problem (v9: "at least half" was flagged
# against Article 368's "not less than one-half").
PHRASING = re.compile(r"\b(less precise than|more precise|standard (constitutional )?(phrasing|wording)|exact (phrasing|"
                      r"wording)|phrased (slightly )?differently)\b", re.I)
# Checks about the student's own suggestions and reasoning: their "try writing" lines rest on the answer, not on a
# fact, so they may stay without a source. Any other wrong-sentence rewrite needs a checked source (v9 Article 356:
# an unsourced rewrite replaced "indefinitely" with another wrong rule).
OPINION_CHECKS = ("solutions", "balance", "causality", "language", "structure", "presentation", "directive")
NUMBER_WORDS = set("""one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand half third quarter
fourth twice thrice एक दो तीन चार पाँच पांच छह छः सात आठ नौ दस चौदह पंद्रह बीस तीस साठ सौ आधा तिहाई""".split())


NEGATION = {"not", "no", "never", "cannot", "can't", "only", "except", "नहीं", "केवल"}


def stems(value):
    """Content words cut to five letters, numbers and negations: "enforced"/"enforceable" and "court"/"courts" match,
    and "cannot" counts as "not", so a correction that says the source's fact in the student's words still matches."""
    from .text import tokens
    out = set()
    for t in tokens(value):
        if t in NEGATION:
            out.add("not" if t not in ("only", "except", "केवल") else "only")
        elif t.isdigit() or t in NUMBER_WORDS:
            out.add(t)
        elif len(t) > 3:
            out.add(t[:5])
    return out


def number_terms(value):
    """Numbers in a text, as digits or words: what makes two fixes the same correction, or a rewrite a new fact."""
    from .text import tokens
    return {t for t in tokens(value) if t.isdigit() or t in NUMBER_WORDS}


class Incomplete(RuntimeError):
    """Too many checks failed for a trustworthy report."""


def _part_label(prep, pid, L):
    if len(prep["parts"]) == 1:
        return ("The question", "सवाल")[L]
    return (f"Part {pid[1:]}", f"भाग {pid[1:]}")[L]


def _where(prep, r, L):
    return _part_label(prep, r["id"][5:], L) if r["kind"] == "part" else DIM[r["id"]][("en", "hi")[L]]


def _join(items, L):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + (" and ", " और ")[L] + items[-1]


def same_problem(g, f):
    """Two checks flagging the same sentence with the same kind of problem describe one problem: the student sees
    it once, with the wording from the higher-ranked check. A missing point is merged only if the texts agree."""
    if g["sourceId"] != f["sourceId"]:
        return False
    if f["sourceId"] is not None and g["kind"] == f["kind"]:
        return True
    return similar(g["fix"], f["fix"], 0.45) or similar(g["reason"], f["reason"], 0.45)


STOP = {"the", "a", "an", "and", "or", "of", "to", "with", "in", "on", "for", "by", "is", "are", "be", "this", "that",
        "these", "your", "you", "it", "as", "how", "such", "like", "more", "add", "include",
        "के", "की", "का", "में", "और", "को", "से", "है", "हैं", "जैसे", "जैसी", "कि", "या", "पर", "एक", "इस", "यह"}


HI_ENDINGS = ("ियों", "ियां", "ियाँ", "ओं", "ों", "ें", "ां", "ाँ", "ी", "ा", "े")


def _stem(t):
    """Fold common Hindi plural and oblique endings, so चुनौतियों / चुनौतियां / चुनौती compare as one word."""
    for e in HI_ENDINGS:
        if len(t) > len(e) + 2 and t.endswith(e):
            return t[:-len(e)]
    return t


def _content(value):
    from .text import tokens
    return {_stem(t) for t in tokens(value) if t not in STOP and len(t) > 1}


def same_advice(a, b, threshold=0.5):
    """Two fixes asking for the same change: most of the shorter one's content words appear in the other."""
    ta, tb = _content(a), _content(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= threshold


def repeats(f, chosen):
    """Same change asked twice: the fixes say the same thing, or the example rewrites are the same sentence (two
    checks can word the advice differently and still hand the student the same line to write). Two fixes from the
    same source that turn on the same number are one correction (v9 Money Bill: "fourteen days" twice)."""
    for g in chosen:
        if same_advice(f["fix"], g["fix"]):
            return True
        if f.get("source") and f.get("source") == g.get("source") and \
                number_terms(f["fix"]) & number_terms(g["fix"]):
            return True
        if f.get("rewrite") and g.get("rewrite") and similar(f["rewrite"], g["rewrite"], 0.85):
            return True
    return False


def trim_rewrite(f, chosen):
    """Drop sentences of a rewrite that repeat a sentence another chosen fix already gives the student."""
    from .text import SENTENCE_END
    if not f.get("rewrite"):
        return
    given = [s for g in chosen if g is not f and g.get("rewrite") for s in SENTENCE_END.split(g["rewrite"]) if s.strip()]
    keep = [s for s in SENTENCE_END.split(f["rewrite"]) if s.strip()
            and not any(same_advice(s, t, 0.7) for t in given)]
    f["rewrite"] = " ".join(keep) if keep else None


def headline(parts, level, majors, L, length=None):
    ratio = (length or {}).get("ratio")
    if level != "off_track" and ratio is not None and ratio < 0.6 and all(p["status"] == "met" for p in parts):
        return (f"At {length['words']} words, this answer is far shorter than the {length['limit']} the question expects.",
                f"{length['words']} शब्दों में यह उत्तर सवाल की अपेक्षित {length['limit']} शब्द-सीमा से बहुत छोटा है।")[L]
    miss = [p["label"] for p in parts if p["status"] == "missing"]
    part = [p["label"] for p in parts if p["status"] == "partial"]
    met = [p["label"] for p in parts if p["status"] == "met"]
    single = len(parts) == 1
    if level == "off_track":
        return ("This answer does not yet address the question.", "यह जवाब अभी सवाल का जवाब नहीं देता।")[L]
    if not miss and not part:
        if level == "strong":
            return ("Every part is answered, and the answer is strong.", "हर हिस्से का जवाब है, और जवाब मज़बूत है।")[L]
        if majors:
            n = len(majors)
            return ((f"Every part is answered, but {n} serious problem{'s' if n > 1 else ''} cost{'' if n > 1 else 's'} you credit."),
                    f"हर हिस्से का जवाब है, पर {n} गंभीर {'कमियाँ' if n > 1 else 'कमी'} अंक घटा {'रही हैं' if n > 1 else 'रही है'}।")[L]
        return ("Every part is answered. The fixes below will make it stronger.",
                "हर हिस्से का जवाब है। नीचे के सुधार इसे और मज़बूत बनाएँगे।")[L]
    if single:
        return ("You answer the question only in part.", "आपने सवाल का जवाब अधूरा दिया है।")[L]
    out = []
    if miss:
        many = len(miss) > 1
        out.append((f"{_join(miss, 0)} {'are' if many else 'is'} missing.",
                    f"{_join(miss, 1)} {'छूट गए हैं' if many else 'छूट गया है'}।")[L])
    if part:
        many = len(part) > 1
        out.append((f"{_join(part, 0)} {'need' if many else 'needs'} the most work.",
                    f"{_join(part, 1)} में सबसे ज़्यादा सुधार चाहिए।")[L] if not miss else
                   (f"{_join(part, 0)} {'are' if many else 'is'} incomplete.",
                    f"{_join(part, 1)} {'अधूरे हैं' if many else 'अधूरा है'}।")[L])
    if met:
        many = len(met) > 1
        out.insert(0, (f"{_join(met, 0)} {'are' if many else 'is'} answered.",
                       f"{_join(met, 1)} का जवाब दिया गया है।")[L])
    return " ".join(out)


def length_of(prep):
    """Words written against the word limit. Hindi gets an allowance on the limit: it writes postpositions and
    auxiliaries as separate words (an inference, to be calibrated on real scripts; docs/marking-standard.md)."""
    words, limit = prep["stats"]["words"], prep["stats"].get("limit")
    if not limit:
        return {"words": words, "limit": None, "ratio": None}
    allowed = limit * (HINDI_ALLOWANCE if prep["lang"] == "hi" else 1)
    return {"words": words, "limit": limit, "ratio": round(words / allowed, 2)}


# What a level needs, in code, so the model cannot talk its way past it. Candidates report about 4/10 as an average
# 10-mark score and 5-6 as good; "Good" here is an exam-ready answer, "Strong" topper level (docs/marking-standard.md).
def level_of(parts, dims, majors, length=None, form=None, L=0):
    """(level, reasons): the level, and in plain words what stops the next one up."""
    statuses = [p["status"] for p in parts]
    if all(s == "missing" for s in statuses) or dims.get("scope", {}).get("verdict") == "missing":
        return "off_track", [("The answer does not yet address the question.", "उत्तर अभी सवाल का जवाब नहीं देता।")[L]]
    verdict = lambda d: dims.get(d, {}).get("verdict")
    ok = lambda d: verdict(d) in ("good", "not_applicable", None)
    ratio = (length or {}).get("ratio")
    words, limit = (length or {}).get("words"), (length or {}).get("limit")

    def short(threshold):
        if ratio is None or ratio >= threshold:
            return []
        return [(f"At {words} words it is too short: a question with this word limit expects about {limit}.",
                 f"{words} शब्दों में यह बहुत छोटा है: इस शब्द-सीमा वाले सवाल में लगभग {limit} शब्द अपेक्षित हैं।")[L]]

    def shape():
        out = []
        if form:
            if not form.get("intro"):
                out.append(("It has no introduction that sets the context.", "इसमें संदर्भ बताने वाली भूमिका नहीं है।")[L])
            if not form.get("conclusion"):
                out.append(("It has no conclusion with a way forward.", "इसमें आगे की राह बताने वाला निष्कर्ष नहीं है।")[L])
            if form.get("points", 0) < form.get("pointsNeeded", 0):
                out.append((f"It makes {form['points']} distinct points; about {form['pointsNeeded']} are expected.",
                            f"इसमें {form['points']} अलग बिंदु हैं; लगभग {form['pointsNeeded']} अपेक्षित हैं।")[L])
        if not out and not ok("structure"):
            out.append(("Its structure needs work (see the Structure check).", "इसके ढाँचे में सुधार चाहिए (ढाँचा जाँच देखें)।")[L])
        return out

    good_needs = []
    if not all(s == "met" for s in statuses):
        good_needs.append(("Not every part of the question is answered.", "सवाल का हर हिस्सा पूरा नहीं है।")[L])
    good_needs += short(0.7 if not ok("dimensions") else 0.6)
    good_needs += shape()
    # The command word stops Good only when the answer commits its failure (a major finding, or a verdict worse than
    # partial). v15 run 45459767588 held a full, well-built Governor answer at Developing on a "partial" whose only
    # notes were ways to deepen it. Strong still needs the command word fully met.
    d = dims.get("directive") or {}
    directive_failed = verdict("directive") in ("missing", "wrong", "unsupported") or (
        verdict("directive") == "partial" and any(c.get("priority") == "major" for c in d.get("concerns") or []))
    if directive_failed:
        good_needs.append(("It does not fully do what the question's command word asks.",
                           "यह सवाल के निर्देश शब्द की माँग पूरी तरह नहीं करता।")[L])
    if len(majors) > 1:
        good_needs.append((f"It has {len(majors)} serious problems (see the fixes).",
                           f"इसमें {len(majors)} गंभीर कमियाँ हैं (सुधार देखें)।")[L])
    if good_needs:
        return "developing", good_needs
    strong_needs = short(0.85)
    if ratio is not None and ratio > 1.3:
        strong_needs.append((f"At {words} words it is well over the limit of about {limit}.",
                             f"{words} शब्दों में यह लगभग {limit} की सीमा से काफ़ी ज़्यादा है।")[L])
    if majors:
        strong_needs.append(("It has a serious problem (see the fixes).", "इसमें एक गंभीर कमी है (सुधार देखें)।")[L])
    if not ok("directive") and not directive_failed:
        strong_needs.append(("It could do more of what the question's command word asks.",
                             "यह सवाल के निर्देश शब्द की माँग को और गहराई से पूरा कर सकता है।")[L])
    if not ok("dimensions"):
        strong_needs.append(("It leaves out an important angle of the issue.", "इसमें मुद्दे का एक ज़रूरी पहलू छूटा है।")[L])
    if not ok("evidence"):
        strong_needs.append(("Its points need specific examples, data or cases.", "इसके बिंदुओं को ठोस उदाहरण, आँकड़े या केस चाहिए।")[L])
    return ("good", strong_needs) if strong_needs else ("strong", [])


def apply_form(dims, form, L):
    """Structure can be good only with an introduction, a conclusion and enough distinct body points."""
    r = dims.get("structure")
    if not r or not form:
        return
    gaps = []
    if not form.get("intro"):
        gaps.append(("Add a one- or two-sentence introduction that sets the context before your first point.",
                     "पहले बिंदु से पहले एक-दो वाक्य की भूमिका जोड़ें जो संदर्भ बताए।")[L])
    if not form.get("conclusion"):
        gaps.append(("End with a one- or two-sentence conclusion that gives a way forward, not a summary.",
                     "अंत में एक-दो वाक्य का निष्कर्ष लिखें जो सार नहीं, आगे की राह बताए।")[L])
    if form.get("points", 0) < form.get("pointsNeeded", 0):
        gaps.append((f"Make about {form['pointsNeeded']} distinct points in the body; it has {form['points']}.",
                     f"मुख्य भाग में लगभग {form['pointsNeeded']} अलग बिंदु रखें; अभी {form['points']} हैं।")[L])
    if not gaps:
        return
    if r["verdict"] == "good":
        r["verdict"] = "partial"
        r["note"] = gaps[0]
    have = " ".join(c["fix"] for c in r["concerns"])
    for g in gaps:
        if not same_advice(g, have):
            r["concerns"].append({"sourceId": None, "quote": None, "kind": "missing", "priority": "medium",
                                  "reason": g, "fix": g, "ref": None, "structural": True})


def format_note(prep, L):
    """What this review does not yet do for the chosen format, said plainly (docs/marking-standard.md)."""
    paper, limit = prep.get("paper"), prep["stats"].get("limit")
    if paper == "ethics_case":
        return ("Case studies are checked here with the same checks as GS answers. A dedicated case-study review "
                "(stakeholders, the value conflict, options and the course of action you choose) is not available yet.",
                "केस स्टडी यहाँ जीएस उत्तरों वाली जाँचों से ही जाँची जाती है। अलग केस-स्टडी समीक्षा (हितधारक, मूल्यों का "
                "टकराव, विकल्प और आपका चुना हुआ रास्ता) अभी उपलब्ध नहीं है।")[L]
    if paper == "essay":
        return ("Essays are checked here with the same checks as GS answers; a dedicated essay review is not available yet.",
                "निबंध यहाँ जीएस उत्तरों वाली जाँचों से ही जाँचा जाता है; अलग निबंध समीक्षा अभी उपलब्ध नहीं है।")[L]
    if paper == "optional" and limit:
        return (f"Optional papers print no word limit; length here is measured against a suggested {limit} words.",
                f"वैकल्पिक विषय के प्रश्नपत्र में शब्द-सीमा नहीं छपती; यहाँ लंबाई सुझाई गई {limit} शब्द-सीमा से मापी गई है।")[L]
    return None


def length_finding(prep, length, L):
    """The length fix, written by code from the word count: the model never supplies these numbers."""
    ratio, words, limit = length.get("ratio"), length.get("words"), length.get("limit")
    if ratio is None or 0.6 <= ratio <= 1.3:
        return None
    marks = prep.get("marks")
    of = (f"a {marks}-mark answer" if marks else "this question", f"{marks} अंक के उत्तर" if marks else "इस सवाल")
    if ratio < 0.6:
        title = (f"Write about {limit} words: this answer has {words}.", f"लगभग {limit} शब्द लिखें: इस उत्तर में {words} हैं।")[L]
        why = (f"The word limit tells you how much {of[0]} needs. At {words} words there is room for only a few points, "
               "so the answer cannot reach the level of an average exam answer however accurate it is.",
               f"शब्द-सीमा बताती है कि {of[1]} में कितना चाहिए। {words} शब्दों में कुछ ही बिंदु आ पाते हैं, इसलिए उत्तर "
               "कितना भी सही हो, औसत परीक्षा-उत्तर के स्तर तक नहीं पहुँचता।")[L]
        priority = "major"
    else:
        title = (f"Cut to about {limit} words: this answer has {words}.", f"लगभग {limit} शब्दों तक छोटा करें: इस उत्तर में {words} हैं।")[L]
        why = ("UPSC asks candidates to keep to the word limit. A much longer answer uses time meant for other questions.",
               "यूपीएससी शब्द-सीमा का पालन करने को कहता है। बहुत लंबा उत्तर दूसरे सवालों का समय ले लेता है।")[L]
        priority = "medium"
    return {"sourceId": None, "quote": None, "kind": "missing", "priority": priority, "reason": why, "fix": title,
            "origin": "presentation", "fromPart": False, "where": DIM["presentation"][("en", "hi")[L]], "rewrite": None,
            "rewriteRef": None, "source": None, "ref": None, "length": True}


def assemble(prep, results):
    L = 1 if prep["lang"] == "hi" else 0
    labels = {r["id"]: r["label"] for r in prep.get("refs") or []}
    spans = {s["id"]: s["text"] for s in prep["sentences"]}
    order = {s["id"]: i for i, s in enumerate(prep["sentences"])}
    ok = [r for r in results if r["status"] == "ok"]
    part_res = {r["id"][5:]: r for r in ok if r["kind"] == "part"}
    dims = {r["id"]: r for r in ok if r["kind"] == "check"}
    # A report is more useful than a retry loop.  Shards are remote work and may
    # exhaust their retries; represent that honestly instead of losing the whole
    # student's report (v9 eci/local).
    for p in prep["parts"]:
        if p["id"] not in part_res:
            part_res[p["id"]] = {"id": "part:" + p["id"], "kind": "part", "status": "unavailable",
                                  "verdict": "partial", "credit": [], "concerns": [], "rewrite": None,
                                  "note": "This part could not be judged automatically.", "notJudged": True}
    # A check skipped because it does not apply (no Constitution/NCERT extract fits the question) still returns a
    # result, so it never counts as a failure here.
    length = length_of(prep)
    form = (dims.get("structure") or {}).get("form")
    apply_form(dims, form, L)
    if length["ratio"] is not None and length["ratio"] < 0.6 and dims.get("presentation", {}).get("verdict") == "good":
        dims["presentation"]["verdict"] = "partial"  # an answer far under the limit is not "economical"
    ordered = [part_res[p["id"]] for p in prep["parts"]] + [dims[d["id"]] for d in DIMENSIONS if d["id"] in dims]

    # 1. Every finding, ranked: priority, then question parts before dimensions, then kind, then answer order.
    all_met = all(part_res[p["id"]]["verdict"] == "met" for p in prep["parts"])
    gaps = [p for p in prep["parts"] if part_res[p["id"]]["verdict"] != "met"]
    findings = []
    for r in ordered:
        rw = r.get("rewrite")
        for c in r["concerns"]:
            matched = bool(rw) and rw["sourceId"] == c["sourceId"]
            f = dict(c, origin=r["id"], fromPart=r["kind"] == "part", where=_where(prep, r, L),
                     rewrite=rw["text"] if matched else None, rewriteRef=rw.get("ref") if matched else None)
            f["source"] = labels.get(f.get("ref")) or labels.get(f.get("rewriteRef"))
            if f["kind"] == "wrong" and f["sourceId"] and OMISSION.search(f["reason"]):
                f["kind"] = "weak"
                if f["priority"] == "major":
                    f["priority"] = "medium"
            if f["kind"] == "weak" and PHRASING.search(f["reason"]):
                continue
            if not f["fromPart"] and f["sourceId"] is None and f["kind"] == "missing":
                # A dimension check asking for something the question part already asks for repeats the part's
                # own finding in other words: fold it into that part.
                if any(same_advice(f["fix"], p["quote"]) or same_advice(f["reason"], p["quote"]) for p in gaps):
                    continue
                # When every part is answered, a point a dimension check would like added is an improvement,
                # not a lost requirement: it cannot rank as a big problem or pull the level down.
                if all_met and f["priority"] == "major":
                    f["priority"] = "medium"
            findings.append(f)
    findings.sort(key=lambda f: (PRI[f["priority"]], 0 if f["fromPart"] else 1, KIND[f["kind"]],
                                 order.get(f["sourceId"], 10 ** 4)))
    kept = []
    for f in findings:
        dup = next((g for g in kept if same_problem(g, f)), None)
        if dup:
            dup.setdefault("alsoFrom", []).append(f["origin"])
            dup["rewrite"] = dup["rewrite"] or f["rewrite"]
            if not dup["source"] and f["source"]:
                dup["source"], dup["ref"] = f["source"], f.get("ref") or f.get("rewriteRef")
            if f.get("evidence") and not dup.get("evidence") and f["source"] == dup["source"]:
                dup["evidence"] = f["evidence"]
        else:
            kept.append(f)
    for f in kept:
        # A wrong fact with no source gets no "try writing" line, whichever merged check supplied it: v10 dropped the
        # unsourced Article 356 rewrite from one check and the merge brought it back from another. Suggestions judged
        # by the opinion checks (concreting a wetland) keep theirs: they rest on the answer, not on a fact.
        origins = [f["origin"]] + f.get("alsoFrom", [])
        if f["kind"] == "wrong" and f["sourceId"] and not f["source"] and \
                not all(o in OPINION_CHECKS for o in origins):
            f["rewrite"] = None
    lf = length_finding(prep, length, L)
    if lf:
        # Too short: first, after any missing or weak question part (answering that part is itself the way to add
        # length). Too long: after the serious problems.
        at = sum(1 for f in kept if f["priority"] == "major" and (f["fromPart"] or lf["priority"] != "major"))
        kept.insert(at, lf)
    majors = [f for f in kept if f["priority"] == "major" and not f.get("length")]

    # 2. Up to three fixes, preferring different sentences so the student is not told three things about one line.
    # A fix that asks for the same change as one already chosen (for example two checks both asking for the two
    # missing challenges, in different words) is skipped: three fixes must be three different things to do.
    fixes, used = [], set()
    for f in kept:
        if len(fixes) < 3 and (f["sourceId"] is None or f["sourceId"] not in used) and not repeats(f, fixes):
            fixes.append(f)
            used.add(f["sourceId"])
    for f in kept:
        if len(fixes) < 3 and f not in fixes and not repeats(f, fixes):
            fixes.append(f)
    for i, f in enumerate(fixes):
        trim_rewrite(f, fixes[:i])
    fix_no = {id(f): i for i, f in enumerate(fixes, 1)}

    # 3. Question parts.
    parts = []
    for p in prep["parts"]:
        r = part_res[p["id"]]
        parts.append({"id": p["id"], "label": _part_label(prep, p["id"], L), "text": p["quote"],
                      "status": r["verdict"], "statusLabel": PART_LABEL[r["verdict"]][L], "note": r["note"],
                      "checkedTwice": r.get("verification") in ("upheld", "softened", "overturned"),
                      "sentences": sorted({c["sourceId"] for c in r["credit"]}, key=order.get)})

    # 4. Highlights on the student's own sentences, coloured by the most serious finding on each.
    marks = {}

    def mark(sid):
        return marks.setdefault(sid, {"sourceId": sid, "text": spans[sid], "credit": [], "concerns": []})

    for r in ordered:
        for c in r["credit"]:
            m = mark(c["sourceId"])
            if len(m["credit"]) < 3 and not any(similar(x["reason"], c["reason"], 0.5) for x in m["credit"]):
                m["credit"].append({"where": _where(prep, r, L), "reason": c["reason"]})
    for f in kept:
        if f["sourceId"]:
            mark(f["sourceId"])["concerns"].append({
                "where": f["where"], "kind": f["kind"], "kindLabel": KIND_LABEL[f["kind"]][L],
                "reason": f["reason"], "fix": f["fix"], "fixNumber": fix_no.get(id(f)), "source": f["source"]})
    for m in marks.values():
        kinds = {c["kind"] for c in m["concerns"]}
        m["tone"] = "red" if kinds & {"wrong", "unsupported"} else "amber" if kinds else "green"
    highlights = sorted(marks.values(), key=lambda m: order[m["sourceId"]])
    missing = [{"partId": f["origin"][5:], "where": f["where"],
                "partText": next(p["quote"] for p in prep["parts"] if p["id"] == f["origin"][5:]),
                "reason": f["reason"], "fix": f["fix"], "fixNumber": fix_no.get(id(f)), "source": f["source"]}
               for f in kept if f["sourceId"] is None and f["fromPart"]]
    additions = [{"where": f["where"], "reason": f["reason"], "fix": f["fix"], "fixNumber": fix_no.get(id(f)),
                  "source": f["source"]}
                 for f in kept if f["sourceId"] is None and not f["fromPart"] and not f.get("length")][:4]

    # 5. What worked: credit from the question parts first, then from checks that went well; one per sentence.
    # A sentence the student is also told to fix is never listed as a strength: mixed praise confuses the student.
    strengths, seen = [], set()
    for r in ordered:
        if r["kind"] == "check" and r["verdict"] != "good":
            continue
        for c in r["credit"]:
            if marks.get(c["sourceId"], {}).get("concerns"):
                continue
            if len(strengths) < 3 and c["sourceId"] not in seen and not any(
                    similar(s["reason"], c["reason"], 0.5) for s in strengths):
                seen.add(c["sourceId"])
                strengths.append({"where": _where(prep, r, L), "sourceId": c["sourceId"], "quote": c["quote"],
                                  "reason": c["reason"]})

    # 6. Full examiner notes, every check in a fixed order.
    checks = []
    for d in DIMENSIONS:
        r = dims.get(d["id"])
        verdict = r["verdict"] if r else "unavailable"
        checks.append({"id": d["id"], "group": d["group"], "groupLabel": GROUPS[d["group"]][L],
                       "label": d[("en", "hi")[L]], "verdict": verdict, "verdictLabel": VERDICT_LABEL[verdict][L],
                       "note": r.get("note") if r else "This check could not be judged automatically."})

    # 7. Numbers about the answer itself.
    served = {}
    for p in prep["parts"]:
        r = part_res[p["id"]]
        for c in r["credit"] + r["concerns"]:
            if c["sourceId"]:
                served.setdefault(c["sourceId"], set()).add(p["id"])
    cited = {c["sourceId"] for r in ordered for c in r["credit"] + r["concerns"] if c["sourceId"]}
    stats = dict(prep["stats"])
    stats["wordsByPart"] = [{"partId": p["id"], "label": _part_label(prep, p["id"], L),
                             "words": sum(word_count(spans[s]) for s, ps in served.items() if p["id"] in ps)}
                            for p in prep["parts"]]
    stats["uncreditedSentences"] = [{"sourceId": s["id"], "text": s["text"]} for s in prep["sentences"]
                                    if s["id"] not in cited]

    level, why = level_of(parts, dims, majors, length, form, L)
    stats.update(marks=prep.get("marks"), format=prep.get("format"), ratio=length["ratio"])
    verified = [d for r in ordered for d in r.get("opinions", [])]
    return {
        "version": REPORT_VERSION, "model": MODEL_FILE, "lang": prep["lang"],
        "level": level, "levelLabel": LEVEL_LABEL[level][L], "levels": [LEVEL_LABEL[x][L] for x in LEVELS],
        "headline": headline(parts, level, majors, L, length), "summary": None, "levelWhy": why,
        "formatNote": format_note(prep, L),
        "question": prep["question"], "directives": directive_guide(prep["question"], prep["lang"]),
        "sentences": prep["sentences"], "parts": parts,
        "fixes": [{"n": i, "title": f["fix"], "where": f["where"], "impact": f["priority"],
                   "impactLabel": IMPACT_LABEL[f["priority"]][L], "kind": f["kind"],
                   "kindLabel": KIND_LABEL[f["kind"]][L],
                   "yourLine": {"sourceId": f["sourceId"], "text": f["quote"]} if f["sourceId"] else None,
                   "why": f["reason"], "tryWriting": f["rewrite"],
                   "checkedTwice": f.get("verification") in ("upheld", "softened"), "source": f["source"],
                   "sourceRef": (f.get("ref") or f.get("rewriteRef")) if f["source"] else None,
                   # The checked clash behind a red fix: the student's wrong words and the source's own words.
                   "evidence": f.get("evidence") if f["kind"] == "wrong" else None}
                  for i, f in enumerate(fixes, 1)],
        "highlights": highlights, "missing": missing, "additions": additions, "strengths": strengths,
        "checks": checks, "stats": stats, "transcript": None,
        # Checked extracts the examiner compared the answer with: labels only, never the text. "used" when at least
        # one finding shown to the student relies on it.
        "sources": [{"label": r["label"], "used": any(f["source"] == r["label"] for f in kept)}
                    for r in prep.get("refs") or []
                    if r.get("shared", True) or any(f["source"] == r["label"] for f in kept)],
        "trust": {"secondOpinions": len(verified), "overturned": verified.count("overturned"),
                  "softened": verified.count("softened"),
                  "thirdOpinions": sum(len(r.get("thirdOpinions", [])) for r in ordered),
                  "thirdRemoved": sum(r.get("thirdOpinions", []).count("same") for r in ordered),
                  "unavailableChecks": [d["id"] for d in DIMENSIONS if d["id"] not in dims]},
    }


SUMMARY_TASK = """TASK [summary]
Write a short summary for the student using ONLY the findings below. Do not add any new point, fact or example.
FINDINGS
{findings}
Fill the JSON field:
- summary: 2 or 3 sentences: what works, what costs the most, and what to do first."""
SUMMARY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["summary"],
                  "properties": {"summary": {"type": "string"}}}


def summarize(examiner, prep, system, report):
    findings = {
        "headline": report["headline"],
        "parts": [{"part": p["label"], "status": p["status"]} for p in report["parts"]],
        "fixFirst": [{"fix": f["title"], "why": f["why"]} for f in report["fixes"]],
        "whatWorks": [s["reason"] for s in report["strengths"]],
    }

    def validate(raw):
        out = clean(raw.get("summary"), prep["lang"], prompts.sources(prep, True), lo=40, hi=520, field="summary")
        if not 2 <= sentence_count(out) <= 4:
            raise Rejected("summary:sentences")
        return out

    try:
        return examiner.ask(stage="summary", check="report", system=system,
                            task=SUMMARY_TASK.format(findings=json.dumps(findings, ensure_ascii=False, indent=1)),
                            schema=SUMMARY_SCHEMA, max_tokens=300, validate=validate, attempts=2)
    except Rejected:
        # Deterministic, source-free fallback: the model summary is optional.
        titles = [f["title"] for f in report["fixes"][:2]]
        return report["headline"] + (" First, " + " Then, ".join(titles) + "." if titles else "")


CORRECT_TASK = """TASK [correct]
This sentence of the student's answer is wrong. Rewrite it as one correct sentence, using ONLY what the extract below
states. Keep the student's topic and as many of the student's own words as stay true. Do not add any other fact.
SENTENCE: \u00ab{sentence}\u00bb
WHAT IS WRONG: {why}
EXTRACT ({label}): {extract}
Fill the JSON field:
- sentence: the corrected sentence."""
CORRECT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["sentence"],
                  "properties": {"sentence": {"type": "string"}}}
CHECKED = ("Constitution of India", "NCERT")


def correct(examiner, prep, system, report):
    """Write each red fix's "try writing" line in its own small call, from the one extract that shows the error, and
    keep it only when every number and name in it comes from that extract or the student's sentence. v9 handed a
    student a wrong rule for Article 356; a fix must never teach a new error."""
    from .checks import copies, licensed
    by_id = {r["id"]: r for r in prep.get("refs") or []}
    by_label = {r["label"]: r for r in prep.get("refs") or []}
    for f in report["fixes"]:
        if f["kind"] != "wrong" or not f.get("yourLine"):
            continue
        # The exact slice the finding cites (Article 356 has four): the label alone would pick the first.
        ref = by_id.get(f.get("sourceRef") or "") or by_label.get(f.get("source") or "")
        if not ref or not f["source"].startswith(CHECKED):
            continue  # `assemble` already dropped an unsourced rewrite of a wrong fact
        sentence = f["yourLine"]["text"]
        ref = dict(ref, text=ref.get("full") or ref["text"])

        def validate(raw, sentence=sentence, ref=ref, evidence=f.get("evidence")):
            from .text import tokens
            out = clean(raw.get("sentence") if isinstance(raw, dict) else None, prep["lang"],
                        [ref["text"], sentence, prep["question"]], lo=15, hi=450, field="correct")
            if number_terms(out) - number_terms(f"{ref['text']} {sentence}"):
                raise Rejected("correct:unsourced_number")
            if similar(out, sentence, 0.9):
                raise Rejected("correct:unchanged")
            if licensed(ref) and copies(out, ref["text"]):
                raise Rejected("correct:copied")
            words = tokens(out)
            if evidence and evidence.get("sourceWords"):
                # Checked against the clash itself: the line must carry something the source says that the student's
                # sentence does not (the corrected fact) and keep the student's topic. v11's overlap rule refused
                # three correct lines because they were worded like the Constitution.
                own = stems(sentence)
                new = stems(evidence["sourceWords"]) - own
                if new and not new & stems(out):
                    raise Rejected("correct:misses_source")
                wrong = stems(evidence.get("wrongWords") or "")
                if not (own - wrong) & stems(out):
                    raise Rejected("correct:off_topic")
            elif words and len(words & tokens(ref["text"])) >= 0.9 * len(words) and \
                    len(words & tokens(sentence)) < 0.35 * len(words):
                raise Rejected("correct:not_a_rewrite")  # no checked clash: an extract copied out is not a correction
            return out

        try:
            f["tryWriting"] = examiner.ask(
                stage="correct", check=f"fix:{f['n']}", system=system,
                task=CORRECT_TASK.format(sentence=prompts.escape(sentence), why=prompts.escape(f["why"]),
                                         label=ref["label"], extract=prompts.escape(ref["text"])),
                schema=CORRECT_SCHEMA, max_tokens=200, validate=validate, attempts=2)
        except Rejected:
            f["tryWriting"] = None
    return report


def attach_page(report, transcript):
    """Point every highlight at the handwritten lines it came from, so the report can mark the student's own page."""
    from .photo import locate
    where = locate(report["sentences"], transcript)
    boxes = {ln["id"]: {"lineId": ln["id"], "page": ln["page"], "box": ln["box"]}
             for ln in transcript["lines"] if ln["text"]}

    def on_page(sid):
        return [boxes[i] for i in where.get(sid, []) if i in boxes]

    for h in report["highlights"]:
        h["lines"] = on_page(h["sourceId"])
    for s in report["strengths"]:
        s["lines"] = on_page(s["sourceId"])
    for f in report["fixes"]:
        if f["yourLine"]:
            f["yourLine"]["lines"] = on_page(f["yourLine"]["sourceId"])
    report["input"] = "photos"
    report["page"] = {"pages": [{"page": p["page"], "width": p["width"], "height": p["height"]}
                                for p in transcript["pages"]],
                      "lines": list(boxes.values())}
    return report
