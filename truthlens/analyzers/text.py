"""OWNER C. Text analysis: claim extraction (regex fallback offline) from the caption that came with the media and from
text evidence files (.txt / .md). No AI-authorship detection (out of scope, TASKS.md). Emit location={"span": "..."}."""
import re
from pathlib import Path
from .. import llm
from ..indicator import Indicator

CLAIMS_PROMPT = """Extract verifiable claims from the context text accompanying media evidence.
Return JSON: {"who": [], "what": [], "where": [], "when": [], "quotes": [],
"inconsistencies": [{"span": "<exact text>", "reason": "<why the text contradicts itself>"}]}.
Write every "when" item as an ISO date: YYYY-MM-DD, or YYYY-MM / YYYY if the text gives only that much.
Write every "where" item as a plain place name.
Only list inconsistencies inside the text itself. Do not judge whether the media was edited or generated."""
CLAIMS_VERSION = "v2"
_STRS = {"type": "array", "items": {"type": "string"}}
CLAIMS_SCHEMA = {"type": "object", "additionalProperties": False,
                 "required": ["who", "what", "where", "when", "quotes", "inconsistencies"],
                 "properties": {**{k: _STRS for k in ("who", "what", "where", "when", "quotes")},
                                "inconsistencies": {"type": "array", "items": {
                                    "type": "object", "additionalProperties": False, "required": ["span", "reason"],
                                    "properties": {"span": {"type": "string"}, "reason": {"type": "string"}}}}}}


def extract_claims(text, case=None):
    claims = llm.ask_json(CLAIMS_PROMPT, llm.untrusted("context", text), schema=CLAIMS_SCHEMA, name="claims",
                          version=CLAIMS_VERSION, effort="low", case=case)
    if claims is None:  # offline fallback
        claims = {"when": re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", text),
                  "quotes": re.findall(r"[\"“](.+?)[\"”]", text)}
    return claims


def analyze(text, case):
    """`text`: the caption string, or the Path of a text evidence file (its claims are attributed to that file)."""
    source = "context"
    if isinstance(text, Path):
        source, text = text.name, text.read_text(encoding="utf-8", errors="replace")
    claims = extract_claims(text, case)
    case.log("text_claims", source=source, model=llm.MODEL if llm.client() else "regex", **claims)
    return [Indicator(source, "text", "text_inconsistency", 0.4, "low", c["reason"], location={"span": c["span"]},
                      alt_explanations=["Ambiguous wording", "Model misreading"])
            for c in claims.get("inconsistencies", [])]
