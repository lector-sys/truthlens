"""OWNER C. Context text analysis: claim extraction via Luna (regex fallback offline).
Keep AI-text detection low weight. Emit location={"span": "..."}."""
import re
from .. import llm
from ..indicator import Indicator

CLAIMS_PROMPT = """Extract verifiable claims from the context text accompanying media evidence.
Return JSON: {"who": [], "what": [], "where": [], "when": [], "quotes": [],
"inconsistencies": [{"span": "<exact text>", "reason": "<why the text contradicts itself>"}]}.
Only list inconsistencies inside the text itself. Do not judge whether the media is real."""


def extract_claims(text):
    claims = llm.ask_json(CLAIMS_PROMPT, text)
    if claims is None:  # offline fallback
        claims = {"when": re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", text),
                  "quotes": re.findall(r"[\"“](.+?)[\"”]", text)}
    return claims


def analyze(text, case):
    claims = extract_claims(text)
    case.log("text_claims", model=llm.MODEL if llm.client() else "regex", **claims)
    return [Indicator("context", "text", "text_inconsistency", 0.4, "low", c["reason"], location={"span": c["span"]},
                      alt_explanations=["Ambiguous wording", "Model misreading"])
            for c in claims.get("inconsistencies", [])]
