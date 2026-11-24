"""End-to-end on the GitHub runner with real Gemma 4 12B: a synthetic submission goes through the queue worker.

    python tools/e2e_qa.py --case photo-en --dir e2e-out

The site is simulated in memory (no production URL, no database). The worker claims the job, reads the photo,
grades it and posts the report, exactly as it will against the real site. Synthetic students only.
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import worker  # noqa: E402
import photo_qa  # noqa: E402

STUDENTS = {s["id"]: s for s in json.loads((ROOT / "qa" / "students.json").read_text())}
BASE = STUDENTS["wetlands-overclaim"]
# A GS2 Polity answer with two planted errors the Constitution/NCERT sources can catch: panchayat elections are run
# by the State Election Commission (Article 243K), not the Election Commission of India, and panchayats get the
# Eleventh Schedule matters a State law devolves (Article 243G), not "all the subjects in the State List".
POLITY = {
    "question": ("Discuss how the 73rd Constitutional Amendment strengthened rural local self-government in India. "
                 "What challenges remain?"),
    "answer": ("The 73rd Amendment of 1992 gave constitutional status to panchayats and created a three-tier system of "
               "village, block and district panchayats. It made elections every five years compulsory, and these "
               "elections are conducted by the Election Commission of India. One-third of the seats are reserved for "
               "women, which has brought many women into local politics. Panchayats now control all the subjects in "
               "the State List. However, many panchayats still depend on grants from the State government and lack "
               "trained staff."),
    "references": ["Panchayats got constitutional status through the 73rd Amendment",
                   "Devolution of funds, functions and staff to panchayats remains uneven across States"],
}
# Polity test set: eight GS2 answers, each with one or two planted errors that the Constitution itself settles.
# Every other sentence is correct. `planted`: a phrase of the wrong sentence and the Articles (or Schedules) that
# show it is wrong. Checked against refs/constitution_of_india_2026.json: each planted error reaches its Article.
POLITY_SET = {
    'polity-local': dict(
        question='Discuss how the 73rd Constitutional Amendment strengthened rural local self-government in India. What challenges remain?',
        answer=('The 73rd Amendment of 1992 gave constitutional status to panchayats and created a three-tier system of village, block and district panchayats. '
                'It made elections every five years compulsory, and these elections are conducted by the Election Commission of India. '
                'One-third of the seats are reserved for women, which has brought many women into local politics. '
                'Panchayats now control all the subjects in the State List. '
                'However, many panchayats still depend on grants from the State government and lack trained staff.'),
        planted=[('Election Commission of India', ['243K']), ('all the subjects in the State List', ['243G', 'SCH11'])]),
    'polity-art21': dict(
        question='Examine how the Supreme Court has widened the scope of Article 21 of the Constitution.',
        answer=('Article 21 says that no one shall be deprived of life or personal liberty except according to procedure established by law. '
                'This protection is available only to citizens of India, not to foreigners. '
                'Over the years the Supreme Court has read rights such as livelihood, a clean environment and privacy into the right to life. '
                'A citizen whose Article 21 right is violated can approach the High Court directly under Article 32. '
                'This expansion has made Article 21 the heart of the Fundamental Rights.'),
        planted=[('only to citizens', ['21']), ('High Court directly under Article 32', ['32'])]),
    'polity-eci': dict(
        question='Examine the constitutional safeguards that protect the independence of the Election Commission of India.',
        answer=('Article 324 vests the superintendence, direction and control of elections to Parliament and the State Legislatures in the Election Commission. '
                'The Chief Election Commissioner can be removed by the President on the advice of the Prime Minister, like any other senior officer. '
                'His conditions of service cannot be varied to his disadvantage after his appointment. '
                'The Commission also prepares the electoral rolls for these elections. '
                'These safeguards help the Commission act without fear of the government of the day.'),
        planted=[('advice of the Prime Minister', ['324'])]),
    'polity-governor': dict(
        question="Analyse the position of the Governor in India's federal structure.",
        answer=('The Governor is the constitutional head of a State. '
                'The Governor is elected by the State Legislative Assembly for a fixed term of five years and cannot be removed before that. '
                'The executive power of the State is vested in the Governor and is exercised through the Council of Ministers. '
                'Because the Governor is linked to the Centre, the office has often been at the centre of Centre-State disputes.'),
        planted=[('elected by the State Legislative Assembly', ['155', '156'])]),
    'polity-amend': dict(
        question='Critically examine the procedure for amending the Constitution under Article 368.',
        answer=('Under Article 368, an amendment can be initiated by introducing a Bill in either House of Parliament. '
                'It must be passed in each House by a majority of the total membership and two-thirds of the members present and voting. '
                'Every constitutional amendment must also be ratified by the legislatures of at least half of the States. '
                'This makes the Indian Constitution partly rigid and partly flexible.'),
        planted=[('Every constitutional amendment must also be ratified', ['368'])]),
    'polity-moneybill': dict(
        question='Discuss the role of the Rajya Sabha in the passage of Money Bills.',
        answer=('A Money Bill can be introduced in either House of Parliament. '
                'After the Lok Sabha passes it, the Bill is sent to the Rajya Sabha for its recommendations. '
                'The Rajya Sabha can keep a Money Bill for up to three months before returning it. '
                "The Lok Sabha may accept or reject the Rajya Sabha's recommendations. "
                'This shows the financial supremacy of the directly elected House.'),
        planted=[('introduced in either House', ['109']), ('up to three months', ['109'])]),
    'polity-356': dict(
        question='Examine the use of Article 356 and its impact on Indian federalism.',
        answer=('Article 356 lets the President take over the government of a State when its government cannot be carried on in accordance with the Constitution. '
                'The proclamation must be approved by both Houses of Parliament. '
                "Once approved, President's Rule can continue indefinitely until the Centre decides to end it. "
                'Its frequent use in the past weakened the autonomy of States.'),
        planted=[('continue indefinitely', ['356'])]),
    'polity-dpsp': dict(
        question='Discuss the relationship between Fundamental Rights and Directive Principles of State Policy.',
        answer=('Fundamental Rights are in Part III and Directive Principles are in Part IV of the Constitution. '
                'Directive Principles can be enforced in courts in the same way as Fundamental Rights. '
                'Directive Principles guide the State in making laws for social and economic justice. '
                'Courts have tried to read both parts together in harmony.'),
        planted=[('enforced in courts in the same way', ['37'])]),
}
# Full-length answers in quality bands, each with the levels an examiner-like standard allows (docs/marking-standard.md):
# a strong 10-marker and 15-marker, a well-shaped answer with two planted errors, an accurate answer with no
# introduction or conclusion, and an off-topic answer. v14 had only 50-80 word answers and called two of them "Good".
BAND_SET = {
    'band-strong-10': dict(
        format='gs-10', marks=10,
        question="Analyse the position of the Governor in India's federal structure. (Answer in 150 words)",
        answer=(            "The Governor is the constitutional head of a State, and Article 154 vests the State's executive power in the office. "
            'Because the Governor is appointed by the President, the office sits where the Union and the States meet.\n\n'
            'As constitutional head, the Governor acts on the aid and advice of the Council of Ministers under Article 163, as the Supreme Court held in Shamsher Singh (1974). '
            'As a link with the Union, the Governor holds office during the pleasure of the President under Article 156. '
            'The Governor may report a failure of constitutional machinery under Article 356 and may reserve State Bills for the President under Article 200. '
            'Discretion in inviting a party to form the government after a hung verdict has often caused Centre-State friction. '
            'The Sarkaria Commission therefore recommended that Governors be detached figures from outside the State.\n\n'
            'For federalism to work, the Governor must act as a constitutional guardian rather than an agent of the Union, within the limits the Court set in B.P. '
            'Singhal (2010) and Nabam Rebia (2016).'),
        expect=('good', 'strong'), planted=[]),
    'band-wrong-10': dict(
        format='gs-10', marks=10,
        question='Discuss the role of the Rajya Sabha in the passage of Money Bills. (Answer in 150 words)',
        answer=(            'Money Bills, defined in Article 110, deal with taxes, borrowing and the Consolidated Fund of India, and their passage shows how Parliament balances its two Houses.\n\n'
            'A Money Bill can be introduced in either House of Parliament on the recommendation of the President. '
            'After the Lok Sabha passes it, the Bill is sent to the Rajya Sabha for its recommendations. '
            'The Rajya Sabha can keep a Money Bill for up to three months before returning it. '
            'It cannot reject or amend the Bill and can only suggest changes, which the Lok Sabha may accept or reject. '
            "The Speaker's certificate that a Bill is a Money Bill is final under Article 110(3). "
            "The passage of the Aadhaar Act as a Money Bill showed how this certificate can narrow the Rajya Sabha's role.\n\n"
            'The arrangement reflects the financial primacy of the directly elected House, but the certification power needs restraint so that the Rajya Sabha is not bypassed.'),
        expect=('developing',), planted=[('introduced in either House', ['109']), ('up to three months', ['109'])]),
    'band-noform-10': dict(
        format='gs-10', marks=10,
        question='Discuss the relationship between Fundamental Rights and Directive Principles of State Policy. (Answer in 150 words)',
        answer=(            'Fundamental Rights in Part III can be enforced in courts, while Article 37 says the Directive Principles in Part IV cannot be enforced by any court. '
            'In Champakam Dorairajan (1951) the Supreme Court held that the Directive Principles must yield to the Fundamental Rights. '
            'The First Amendment then allowed special provisions for backward classes. '
            'In Golaknath (1967) the Court held that Parliament could not amend the Fundamental Rights. '
            'The Twenty-fifth Amendment added Article 31C to protect laws giving effect to Article 39(b) and (c). '
            'In Kesavananda Bharati (1973) the Court upheld the first part of Article 31C but struck down the clause that barred judicial review. '
            'The right to education in Article 21A began as a Directive Principle in Article 45. '
            'The Forty-second Amendment extended Article 31C to all the Directive Principles, and Minerva Mills (1980) struck down that extension.'),
        expect=('developing',), planted=[]),
    'band-strong-15': dict(
        format='gs-15', marks=15,
        question='Examine the role of the Election Commission of India in ensuring free and fair elections. (Answer in 250 words)',
        answer=(            'Article 324 vests the superintendence, direction and control of elections to Parliament, the State Legislatures and the offices of President and Vice-President in the Election Commission of India. '
            "Its independence and its wide powers are the base of India's electoral democracy.\n\n"
            "Constitutionally, the Chief Election Commissioner can be removed only in the manner and on the grounds of a Supreme Court judge, and the conditions of service cannot be varied to the CEC's disadvantage after appointment. "
            'The Supreme Court held in Mohinder Singh Gill (1978) that Article 324 gives the Commission plenary powers where the law is silent. '
            'Administratively, the Commission prepares the electoral rolls, schedules elections and deploys observers and central forces. '
            'Through the Model Code of Conduct it restrains the use of government machinery once elections are announced, although the code rests on consensus rather than a statute. '
            'Technologically, electronic voting machines with VVPAT slips have made counting faster and more verifiable.\n\n'
            'Challenges remain. '
            "Other Election Commissioners lack the CEC's protection, as they can be removed on the CEC's recommendation. "
            'Money power and criminal candidates still weaken the level playing field. '
            'In Anoop Baranwal (2023) the Court required a selection committee including the Chief Justice until Parliament legislated, but the 2023 Act replaced the Chief Justice with a Union Minister, which revived concerns about independence.\n\n'
            "Giving all Commissioners equal protection, a statutory basis for the Model Code and an independent selection process would make the Commission's role in free and fair elections more secure."),
        expect=('good', 'strong'), planted=[]),
    'band-offtopic-10': dict(
        format='gs-10', marks=10,
        question="Analyse the position of the Governor in India's federal structure. (Answer in 150 words)",
        answer=(            'Urban wetlands hold stormwater after heavy rain, so less water enters drains at once. '
            'They recharge groundwater and keep cities cooler in summer. '
            'Encroachment and dumping of waste have shrunk many of them. '
            'Municipalities should map their wetlands, stop construction on their beds and involve local communities in protecting them.'),
        expect=('off_track',), planted=[]),
}

CASES = {
    "typed-en": dict(input="typed"),
    "typed-polity": dict(input="typed", **POLITY),
    "photo-en": dict(input="photos", page="clear-en"),
    "photo-smudged": dict(input="photos", page="smudged-en"),
}
CASES.update({k: dict(input="typed", question=v["question"], answer=v["answer"], references=[])
              for k, v in POLITY_SET.items()})
CASES.update({k: dict(input="typed", question=v["question"], answer=v["answer"], references=[], marks=v["marks"],
                      format=v["format"]) for k, v in BAND_SET.items()})


class Site:
    def __init__(self, job):
        self.jobs, self.calls = [job], []

    def __call__(self, p):
        a = p["action"]
        self.calls.append({k: v for k, v in p.items() if k not in ("lease",)})
        if a == "claim":
            return {"job": self.jobs.pop(0) if self.jobs else None}
        if a == "transcribe":
            return {"ok": True, "blocked": p["transcript"]["blocked"]}
        if a == "deliver":
            return {"processed": 0}
        return {"ok": True}


def job_for(case):
    c = CASES[case]
    job = {"id": "e2e-" + case, "lease": "lease", "lang": "en", "question": c.get("question", BASE["question"]),
           "references": c.get("references", BASE["references"]), "marks": c.get("marks", 10),
           "format": c.get("format", "gs-10"), "input": c["input"]}
    if c["input"] == "typed":
        job["answer"] = c.get("answer", BASE["answer"])
    else:
        job["pages"] = [base64.b64encode(photo_qa.draw(c["page"])).decode()]
    return job


def level_gates(rep, expect=None):
    """The level must follow the standard: an answer under 60% of its word limit is never Good or Strong."""
    g = {}
    ratio = rep["stats"].get("ratio")
    if ratio is not None and ratio < 0.6:
        g["a short answer is not Good or Strong"] = rep["level"] in ("developing", "off_track")
    if expect:
        g[f"level is {' or '.join(expect)}"] = rep["level"] in expect
    g["the report says what stops the next level"] = rep["level"] == "strong" or bool(rep.get("levelWhy"))
    return g


def band_gates(case, rep):
    v = BAND_SET[case]
    g = level_gates(rep, v["expect"])
    wrong_ids = set()
    for phrase, arts in v["planted"]:
        want = {f"Constitution of India, Article {a}" for a in arts}
        h = next((h for h in rep["highlights"] if phrase in h["text"]), None)
        wrong_ids.add(next((s["id"] for s in rep["sentences"] if phrase in s["text"]), None))
        g[f"caught: {phrase}"] = bool(h) and h["tone"] == "red"
        g[f"cited: {phrase}"] = bool(h) and any(c.get("source") in want for c in h["concerns"])
    false_red = [h["text"] for h in rep["highlights"] if h["tone"] == "red" and h["sourceId"] not in wrong_ids]
    if case != "band-offtopic-10":
        g["no red marks on correct sentences"] = not false_red
    if case == "band-noform-10":
        g["structure not rated good without an introduction and a conclusion"] = next(
            (c["verdict"] for c in rep["checks"] if c["id"] == "structure"), None) != "good"
    rep.setdefault("qa", {}).update(falseRed=false_red, levelWhy=rep.get("levelWhy"))
    return g


def polity_gates(case, rep):
    """Per planted error: its sentence is marked red, and a finding on it cites one of the Articles that settle it.
    Also counts red marks on the correct sentences (false alarms); those are reported, not failed."""
    planted = POLITY_SET[case]["planted"]
    labels = {f"Constitution of India, {'Schedule ' + a[3:] if a.startswith('SCH') else 'Article ' + a}"
              for _, arts in planted for a in arts}
    g, wrong_ids = {}, set()
    for phrase, arts in planted:
        want = {f"Constitution of India, {'Schedule ' + a[3:] if a.startswith('SCH') else 'Article ' + a}" for a in arts}
        h = next((h for h in rep["highlights"] if phrase in h["text"]), None)
        sid = next((s["id"] for s in rep["sentences"] if phrase in s["text"]), None)
        wrong_ids.add(sid)
        g[f"caught: {phrase}"] = bool(h) and h["tone"] == "red"
        g[f"cited: {phrase}"] = bool(h) and any(c.get("source") in want for c in h["concerns"])
    rep.setdefault("qa", {})["falseRed"] = [h["text"] for h in rep["highlights"]
                                            if h["tone"] == "red" and h["sourceId"] not in wrong_ids]
    rep["qa"]["sourcesUsed"] = [s["label"] for s in rep["sources"] if s["used"]]
    rep["qa"]["expected"] = sorted(labels)
    # v10: a correct sentence marked red teaches the student a wrong fact about their own answer.
    g["no red marks on correct sentences"] = not rep["qa"]["falseRed"]
    # Landmark cases are measured, not gated: the public runs carry our own case summaries only.
    shown = rep["fixes"] + rep["additions"] + rep["missing"] + [c for h in rep["highlights"] for c in h["concerns"]]
    cited = sorted({x["source"] for x in shown if (x.get("source") or "").startswith("Case:")})
    want = EXPECTED_CASES.get(case, ())
    rep["qa"]["casesGiven"] = [s["label"] for s in rep["sources"] if s["label"].startswith("Case:")]
    rep["qa"]["casesCited"] = cited
    rep["qa"]["expectedCaseCited"] = any(w in c for c in cited for w in want) if want else None
    rep["qa"]["redFixesWithTryWriting"] = sum(1 for f in rep["fixes"] if f["kind"] == "wrong" and f["tryWriting"])
    return g


EXPECTED_CASES = {
    "polity-art21": ("Maneka Gandhi", "Olga Tellis", "Puttaswamy", "Francis Coralie Mullin"),
    "polity-356": ("S.R. Bommai", "Rameshwar Prasad"),
    "polity-dpsp": ("Minerva Mills", "Champakam Dorairajan", "Kesavananda Bharati"),
    "polity-amend": ("Kesavananda Bharati", "Minerva Mills", "Golaknath", "I.R. Coelho"),
    "polity-governor": ("B.P. Singhal", "Shamsher Singh", "Nabam Rebia", "Rameshwar Prasad"),
}


def gates(case, site, counts):
    actions = [c["action"] for c in site.calls]
    tr = next((c["transcript"] for c in site.calls if c["action"] == "transcribe"), None)
    rep = next((c["result"] for c in site.calls if c["action"] == "complete"), None)
    g = {"claimed and delivered": actions[0] == "claim" and actions[-1] == "deliver"}
    if case == "photo-smudged":
        g["photo blocked for a retake"] = bool(tr and tr["blocked"])
        g["not graded"] = rep is None
        g["retake message given"] = bool(tr and tr["retakeMessage"])
        return g
    g["report completed"] = counts["done"] == 1 and rep is not None
    if rep is None:
        return g
    g["report version v15"] = rep["version"] == "mains-report-v15"
    g["three or fewer fixes"] = len(rep["fixes"]) <= 3
    g["every check judged"] = not rep["trust"]["unavailableChecks"]
    if case in BAND_SET:
        return g | band_gates(case, rep)
    if case in POLITY_SET:
        return g | polity_gates(case, rep) | level_gates(rep)
    if case == "typed-polity":
        shown = rep["fixes"] + rep["additions"] + rep["missing"] + [c for h in rep["highlights"] for c in h["concerns"]]
        g["Constitution/NCERT sources given"] = bool(rep["sources"])
        g["a finding cites a source"] = any(x.get("source") for x in shown)
        eci = [h for h in rep["highlights"] if "Election Commission of India" in h["text"]]
        g["Election Commission error marked red"] = bool(eci) and eci[0]["tone"] == "red"
        return g
    # Background GS textbooks (v16) may support a wetlands answer; the Constitution must not.
    g["no Constitution sources for an off-syllabus topic"] = not any(
        s["label"].startswith("Constitution of India") for s in rep["sources"])
    des = [h for h in rep["highlights"] if "desilt" in h["text"].lower()]
    g["routine desilting flagged, not praised"] = bool(des) and all(h["tone"] != "green" for h in des) and not any(
        "desilt" in s["quote"].lower() for s in rep["strengths"])
    red = {h["text"] for h in rep["highlights"] if h["tone"] == "red"}
    g["80% claim marked red"] = any("80%" in t for t in red)
    g["concreting marked red"] = any("concrete" in t for t in red)
    if case == "photo-en":
        g["photo read without a retake"] = bool(tr) and not tr["blocked"]
        g["every highlight on the page"] = all(h.get("lines") for h in rep["highlights"])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--dir", default="e2e-out")
    a = ap.parse_args()
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    job = job_for(a.case)
    site = Site(job)
    started = time.monotonic()
    counts = worker.process(site)
    g = gates(a.case, site, counts)
    record = {"case": a.case, "counts": counts, "gates": g, "seconds": round(time.monotonic() - started, 1),
              "calls": site.calls}
    (out / f"e2e-{a.case}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
    print(f"{a.case}: {counts} in {record['seconds']} s", flush=True)
    for k, v in g.items():
        print(f"  {'PASS' if v else 'FAIL'} {k}", flush=True)
    sys.exit(0 if all(g.values()) else 1)


if __name__ == "__main__":
    main()
