"""What the examiner checks, and the fixed labels the student report uses in English and Hindi."""

GROUPS = {
    "question": ("The question", "सवाल"),
    "content": ("Content", "सामग्री"),
    "reasoning": ("Reasoning", "तर्क"),
    "writing": ("Writing", "लेखन"),
}

# Twelve focused checks. Each is one separate Gemma call that judges only its own dimension; the question's
# parts are checked separately (one call per part), so these never re-judge "did you answer part 2".
DIMENSIONS = [
    dict(id="directive", group="question", en="Did what the question asks", hi="सवाल की माँग पूरी की", task=(
        "Check the command words of the question (for example discuss, explain, critically examine, analyse, evaluate, "
        "comment, suggest). Does the answer do that kind of thinking, or only describe the topic? 'Critically examine' "
        "and 'evaluate' need a judgement backed by reasons; 'discuss' needs more than one angle; 'explain' needs the "
        "how and why; 'suggest' needs concrete measures.")),
    dict(id="scope", group="question", en="Stayed on topic", hi="विषय पर टिके रहे", task=(
        "Check relevance. Find sentences that drift from what is asked (another topic, place, period or sector) or "
        "that only fill space. Credit an answer that stays focused. If the whole answer is about something else, the "
        "verdict is missing.")),
    dict(id="dimensions", group="question", en="Covered the key angles", hi="ज़रूरी पहलू", task=(
        "UPSC rewards answers that look at an issue from the angles that matter for it (for example social, economic, "
        "political, environmental, administrative, legal, ethical, technological, federal). Credit the angles the "
        "answer covers and name the single most important relevant angle it leaves out. Do not demand angles that do "
        "not fit this question.")),
    dict(id="concepts", group="content", en="Concepts explained", hi="मूल बातें", task=(
        "Check whether the core concepts and terms the question depends on are explained correctly and with enough "
        "depth, not just named.")),
    dict(id="accuracy", group="content", en="Facts are accurate", hi="तथ्य सही", task=(
        "Find factual statements that are wrong, unsupported, exaggerated or stated as universal (every, always, "
        "never, will end). Judge against the reference points and well-established knowledge only. A claim you cannot "
        "confirm is 'unsupported', not 'wrong'. Never supply a corrected number, date or name unless it appears in "
        "MATERIALS; say what the student should check instead. If there is no factual problem, credit the most "
        "precise factual sentence.")),
    dict(id="evidence", group="content", en="Examples and evidence", hi="उदाहरण", task=(
        "Check whether points are supported by specific examples, cases, schemes, committee or report findings, data "
        "or court judgements, and whether each example really supports its point. If evidence is missing, describe "
        "the kind of evidence that would help most (for example 'one city flood plan you have studied'); never "
        "invent a specific example.")),
    dict(id="causality", group="reasoning", en="Cause and effect", hi="कारण और असर", task=(
        "Check the reasoning chain: does the answer show how one thing leads to another, up to the outcome the "
        "question asks about? Point to the weakest or missing link, or credit a clear chain.")),
    dict(id="balance", group="reasoning", en="Trade-offs and limits", hi="सीमाएँ और मुश्किलें", task=(
        "Check whether the answer recognises limits, costs, counter-arguments, affected groups or uncertainty where "
        "the question calls for it. Do not force two sides onto a question that does not need them; then the verdict "
        "is not_applicable.")),
    dict(id="solutions", group="reasoning", en="Suggestions are workable", hi="सुझाव व्यावहारिक हैं", task=(
        "If the answer makes suggestions, check whether each is workable: who acts, what exactly, where or when it "
        "applies, and whether it fits the problem. Flag a suggestion that would harm the goal as wrong. Do not credit "
        "a measure just because it sounds standard: a step that should be targeted, offered as a blanket routine (for "
        "example yearly dredging of every canal instead of where a survey shows it is needed), is a concern, and the "
        "fix says where or when it applies. Judge every suggestion on its own: after the most serious problem, a "
        "lesser flaw in a different suggestion still gets its own concern (kind weak). If the "
        "question asks for no suggestions and the answer makes none, the verdict is not_applicable.")),
    dict(id="structure", group="writing", en="Structure", hi="ढाँचा", task=(
        "Check the order: an opening that addresses the question directly, a body organised by the question's parts, "
        "and a conclusion that adds a balanced or forward-looking point instead of repeating or overclaiming. Do not "
        "demand a generic introduction if the answer already opens well.")),
    dict(id="language", group="writing", en="Clear wording", hi="साफ़ भाषा", task=(
        "Check clarity and precise subject terms. Point to one phrase that is vague or imprecise and show a sharper "
        "version, or credit precise wording. Never penalise writing in English or Hindi, or simple vocabulary.")),
    dict(id="presentation", group="writing", en="Use of space", hi="जगह का इस्तेमाल", task=(
        "Check how the answer uses its length: repetition, padding, sentences that earn nothing, and whether the "
        "space is shared sensibly across the question's parts. MATERIALS.facts gives the word count and the usual "
        "limit; use those figures and do not count words yourself.")),
]
DIM = {d["id"]: d for d in DIMENSIONS}

# Runner shards: each shard is one GitHub job that loads Gemma once and runs its checks back to back (the shared
# system prompt is reused from the model's cache between checks). "parts" holds one check per question part.
SHARDS = {
    "parts": None,
    "question": ("directive", "scope", "dimensions"),
    "content": ("concepts", "accuracy", "evidence"),
    "reasoning": ("causality", "balance", "solutions"),
    "writing": ("structure", "language", "presentation"),
}

VERDICT_LABEL = {
    "good": ("Good", "अच्छा"), "partial": ("Partly", "अधूरा"), "missing": ("Missing", "छूटा"),
    "unsupported": ("Unsupported", "बिना आधार"), "wrong": ("Wrong", "गलत"),
    "not_applicable": ("Not needed here", "यहाँ ज़रूरी नहीं"), "unavailable": ("Not checked", "जाँच नहीं हो सकी"),
}
PART_LABEL = {"met": ("Answered", "पूरा"), "partial": ("Partly", "अधूरा"), "missing": ("Missing", "छूटा")}
LEVELS = ("off_track", "developing", "good", "strong")
LEVEL_LABEL = {
    "off_track": ("Off track", "विषय से हटकर"), "developing": ("Developing", "सुधार चाहिए"),
    "good": ("Good", "अच्छा"), "strong": ("Strong", "बहुत अच्छा"),
}
IMPACT_LABEL = {
    "major": ("Big effect on your marks", "अंकों पर बड़ा असर"),
    "medium": ("Medium effect on your marks", "अंकों पर मध्यम असर"),
    "minor": ("Small polish", "छोटा सुधार"),
}

# What each command word asks for. Fixed text, written and checked by hand, shown with the report; never generated.
DIRECTIVES = [
    (("critically examine", "critically analyse", "critically evaluate", "समालोचनात्मक"),
     ("Critically examine", "Look at the issue closely, weigh its strengths and weaknesses with evidence, and end with your own reasoned judgement."),
     ("समालोचनात्मक परीक्षण", "मुद्दे को गहराई से देखिए, सबूत के साथ उसकी खूबियाँ और कमियाँ तौलिए, और अंत में तर्क के साथ अपना निष्कर्ष दीजिए।")),
    (("discuss", "चर्चा"),
     ("Discuss", "Look at the issue from more than one side, with reasons, and reach a balanced view."),
     ("चर्चा कीजिए", "मुद्दे को एक से ज़्यादा पहलुओं से देखिए, कारण दीजिए और संतुलित राय पर पहुँचिए।")),
    (("explain", "elucidate", "समझाइए", "स्पष्ट कीजिए"),
     ("Explain", "Make clear how and why it happens: causes, the steps in between, and the effects."),
     ("समझाइए", "साफ़ बताइए कि यह कैसे और क्यों होता है: कारण, बीच की कड़ियाँ और असर।")),
    (("analyse", "analyze", "विश्लेषण"),
     ("Analyse", "Break the issue into its parts and show how they connect and what follows from them."),
     ("विश्लेषण कीजिए", "मुद्दे को हिस्सों में बाँटिए और दिखाइए कि वे कैसे जुड़े हैं और उनसे क्या नतीजा निकलता है।")),
    (("evaluate", "assess", "मूल्यांकन", "आकलन"),
     ("Evaluate", "Weigh the merits and limits and give a clear judgement on how well it works."),
     ("मूल्यांकन कीजिए", "फ़ायदे और सीमाएँ तौलिए और साफ़ फ़ैसला दीजिए कि यह कितना कारगर है।")),
    (("examine", "परीक्षण"),
     ("Examine", "Look into the issue in detail: its causes, effects and implications."),
     ("परीक्षण कीजिए", "मुद्दे को विस्तार से जाँचिए: उसके कारण, असर और नतीजे।")),
    (("comment", "टिप्पणी"),
     ("Comment", "Give your informed opinion on the statement, backed by reasons."),
     ("टिप्पणी कीजिए", "कथन पर तर्क के साथ अपनी सोची-समझी राय दीजिए।")),
    (("suggest", "सुझाइए", "सुझाव"),
     ("Suggest", "Propose practical measures and say who should act and how."),
     ("सुझाइए", "व्यावहारिक उपाय बताइए और यह भी कि कौन और कैसे करेगा।")),
    (("illustrate", "उदाहरण सहित", "उदाहरण देकर"),
     ("Illustrate", "Make the point with a concrete example."),
     ("उदाहरण दीजिए", "बात को एक ठोस उदाहरण से समझाइए।")),
]


def usual_limit(marks):
    """UPSC GS Mains: 150 words for a 10-mark question, 250 for 15 marks."""
    return {10: 150, 15: 250}.get(marks)


def directive_guide(question, lang):
    """The command words in the order they appear in the question, each with its fixed meaning."""
    q = question.casefold()
    found, taken = [], set()
    for keys, en, hi in DIRECTIVES:
        at = min((q.find(k) for k in keys if k in q), default=-1)
        if at < 0 or en[0] in taken:
            continue
        # "critically examine" also contains "examine": keep only the stronger directive.
        if en[0] == "Examine" and "Critically examine" in taken:
            continue
        taken.add(en[0])
        word, meaning = hi if lang == "hi" else en
        found.append((at, {"word": word, "meaning": meaning}))
    return [d for _, d in sorted(found, key=lambda x: x[0])]
