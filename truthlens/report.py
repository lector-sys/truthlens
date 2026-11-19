"""OWNER D. HTML forensic report: self-contained (artifacts embedded), human checks, summary from indicators only."""
import base64, json, mimetypes
from pathlib import Path
from jinja2 import Template
from . import llm, fusion

SUMMARY_PROMPT = """You write the summary section of a forensic report for a human investigator.
Use ONLY the indicators and assessment given. Do not add facts, guesses about intent, or new findings.
Never use the words fake, real, authentic, genuine, proof. Say "indicators" and "may indicate".
Mention the strongest 2-3 indicators with their location, the strongest benign alternative, and that a human must review.
Return JSON: {"summary": "<3-5 sentences>"}"""
SUMMARY_VERSION = "v2"
SUMMARY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["summary"],
                  "properties": {"summary": {"type": "string"}}}

TEMPLATE = Template("""<!DOCTYPE html><html><head><meta charset="utf-8"><title>TruthLens report {{ case.id }}</title>
<style>body{font-family:sans-serif;max-width:1000px;margin:auto;padding:0 12px}td,th{border:1px solid #ccc;padding:4px;vertical-align:top}
table{border-collapse:collapse;width:100%}img{max-width:220px}.note{color:#555}</style>
</head><body>
<h1>TruthLens forensic report: {{ case.id }}</h1>
<p><b>Overall:</b> {{ a.case.label }} (indicator strength {{ '%.2f' % a.case.p }}).</p>
<p class="note">Automated indicators, not proof. All findings require human review.</p>
<p class="note">Fusion weights: {{ calib }}. Semantic checks: {{ semantic }}.</p>
<h2>Summary</h2><p>{{ summary }}</p>
<h2>Evidence & integrity</h2>
<table><tr><th>File</th><th>SHA-256</th><th>Unchanged since ingest</th><th>Assessment</th></tr>
{% for e in ingests %}<tr><td>{{ e.file }}</td><td><code>{{ e.sha256 }}</code></td><td>{{ 'yes' if integrity.get(e.file) else 'NO' }}</td>
<td>{{ a.files.get(e.file, {}).get('label', 'No significant indicators') }}</td></tr>{% endfor %}</table>
<h2>Flagged evidence</h2>
{% if not indicators %}<p>No indicators were raised.</p>{% else %}
<table><tr><th>File</th><th>Indicator</th><th>Location</th><th>Score / reliability</th><th>Why</th><th>Could also be</th><th>Evidence</th></tr>
{% for i, img in rows %}<tr><td>{{ i.file }}{% if i.related %} ↔ {{ i.related|join(', ') }}{% endif %}</td><td>{{ i.modality }} · {{ i.type }}</td>
<td>{{ i.location }}</td><td>{{ '%.2f' % i.score }} / {{ i.reliability }}</td><td>{{ i.explanation }}</td>
<td>{{ i.alt_explanations|join('; ') }}</td><td>{% if img %}<img src="{{ img }}" alt="{{ i.type }} artifact">{% endif %}</td></tr>{% endfor %}</table>{% endif %}
<h2>Recommended human checks</h2><ol>{% for c in checks %}<li>{{ c }}</li>{% else %}<li>Spot-check the original files; absence of indicators is not proof of integrity.</li>{% endfor %}</ol>
<h2>Limitations</h2><ul>
<li>Detectors are heuristics tuned on a small team-made test set; reliability labels reflect that, not a guarantee.</li>
<li>Re-compression, resizing and platform uploads can create or hide indicators.</li>
<li>The language model is used only to read text and compare captions with images, never to judge authenticity.</li>
<li>Absence of indicators does not prove the media is unaltered.</li></ul>
<h2>Analysis log</h2><pre>{% for e in log %}{{ e }}
{% endfor %}</pre></body></html>""")


def data_uri(path):
    p = Path(path) if path else None
    if not p or not p.exists() or p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        return None
    return f"data:{mimetypes.guess_type(p.name)[0]};base64,{base64.b64encode(p.read_bytes()).decode()}"


def human_check(i):
    loc = i.location
    if "bbox" in loc:
        x, y, w, h = loc["bbox"]
        return f"Inspect {i.file} region x={x}, y={y}, {w}×{h}px at full resolution ({i.type})."
    if "t" in loc:
        return f"Step through {i.file} frame by frame around {loc['t']:.1f}s ({i.type})."
    if "start" in loc:
        return f"Listen to {i.file} from {loc['start']:.1f}s to {loc['end']:.1f}s with headphones ({i.type})."
    if "span" in loc:
        return f"Verify \"{loc['span']}\" against the original source ({i.type})."
    if "meta" in loc:
        return f"Ask the source for the original, un-uploaded {i.file} to compare its {loc['meta']} metadata."
    return None


def human_checks(indicators, limit=10):
    """Deduplicated checks, strongest indicators first."""
    ordered = sorted(indicators, key=lambda i: -i.score)
    return list(dict.fromkeys(c for c in map(human_check, ordered) if c))[:limit]


def summary(result, case=None):
    """LLM summary from indicators only; cached on the result so Streamlit reruns don't re-bill the API."""
    if "summary" not in result:
        a, inds = result["assessment"], result["indicators"]
        data = json.dumps({"assessment": a, "indicators": [i.to_dict() for i in inds]}, default=str)
        r = llm.ask_json(SUMMARY_PROMPT, llm.untrusted("indicators", data), schema=SUMMARY_SCHEMA, name="summary",
                         version=SUMMARY_VERSION, effort="low", case=case)
        top = [k for k, v in sorted(fusion.contributions(inds).items(), key=lambda kv: -kv[1]) if v > 0][:3]
        fallback = (f"{a['case']['label']} across {len(a['files'])} file(s). Strongest: "
                    + ("; ".join(f"{t} in {f}" for f, t in top) or "none")
                    + ". Each finding has benign alternatives listed and needs human review.")
        result["summary"] = (r or {}).get("summary") or fallback
    return result["summary"]


def render(case, result):
    log = case.entries()
    inds = sorted(result["indicators"], key=lambda i: -i.score)
    checks = human_checks(inds)
    cal = fusion.load_calibration()
    calib = (f"{cal['dataset']} test set, n={cal['n']}, held-out n={cal['n_held_out']}, {cal['date']}, git {cal['git']}"
             if cal else "uncalibrated reliability defaults")
    calls = [e for e in log if e["action"] == "llm_call" and e["check"] != "transcribe"]
    skipped = sorted({e["check"] for e in log if e["action"] == "check_not_run"})
    semantic = ", ".join(sorted({f"{e['model']}, prompt {e.get('prompt_version', '?')}" for e in calls})) or "none run"
    if skipped:
        semantic += f" (not run: {', '.join(skipped)})"
    return TEMPLATE.render(calib=calib, semantic=semantic, case=case, a=result["assessment"], indicators=inds, rows=[(i, data_uri(i.artifact)) for i in inds],
                           integrity=result["integrity"], ingests=[e for e in log if e["action"] == "ingest"],
                           log=log, checks=checks, summary=summary(result, case))
