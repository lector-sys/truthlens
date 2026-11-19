"""OWNER D. HTML forensic report: self-contained (artifacts embedded), human checks, summary from indicators only."""
import base64, json, mimetypes, re
from pathlib import Path
from jinja2 import Template
from . import llm, fusion
from .analyzers import image

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
{% if missing %}<p style="background:#F97352;color:#000;padding:8px;font-weight:bold">{{ missing }}</p>{% endif %}
{% for c in credentials %}<p style="background:#EDE9FE;border-left:4px solid #7C3AED;padding:8px"><b>{{ c|e }}</b></p>{% endfor %}
{% if headline.startswith('Not analysed') %}<p style="color:#64748B"><b>Overall: {{ headline }}.</b> No analyzer ran, so no label is given.</p>
{% else %}<p><b>Overall:</b> {{ headline }} (indicator strength {{ '%.2f' % a.case.p }}).</p>{% endif %}
{% if credentials %}<p class="note">{{ credentials_note }}</p>{% endif %}
{% if coverage_line %}<p style="color:#64748B">{{ coverage_line }}</p>{% endif %}
<p class="note">Automated indicators, not proof. All findings require human review.</p>
<p class="note">Fusion weights: {{ calib }}. Semantic checks: {{ semantic }}.</p>
{% if provenance %}<p class="note">{{ provenance }} Model answers: {{ source }}.</p>{% endif %}
<h2>Summary</h2><p>{{ summary }}</p>
<p><b>Recommended human checks</b></p><ol>{% for c in checks[:5] %}<li>{{ c }}</li>{% else %}<li>Spot-check the original files; absence of indicators is not proof of integrity.</li>{% endfor %}</ol>
<h2>Evidence & integrity</h2>
<table><tr><th>File</th><th>SHA-256</th><th>Unchanged since ingest</th><th>Assessment</th></tr>
{% for e in ingests %}<tr><td>{{ e.file }}</td><td><code>{{ e.sha256 }}</code></td><td>{{ 'yes' if integrity.get(e.file) else 'NO' }}</td>
<td>{{ file_headline(e.file) }}</td></tr>{% endfor %}</table>
{% for e in conversions %}<p class="note">{{ e.file }} was analysed via {{ e.to }} (converted with {{ e.tool }}, sha256 {{ e.sha256 }}):
compression-based checks (ELA) see the conversion, not the original. The evidence hash is the original file's.</p>{% endfor %}
<h2>Flagged evidence</h2>
{% if not indicators %}<p>No indicators were raised.</p>{% endif %}
{% for g, what in groups.items() %}{% set grp = rows|selectattr('0.type', 'in', group_types[g])|list %}{% if grp %}
<h3>{{ g }}: {{ what }}</h3>
<table><tr><th>File</th><th>Indicator</th><th>Location</th><th>Score / reliability</th><th>Why</th><th>Could also be</th><th>Recommended check</th><th>Evidence</th></tr>
{% for i, img in grp %}<tr><td>{{ i.file }}{% if i.related %} ↔ {{ i.related|join(', ') }}{% endif %}</td><td>{{ i.modality }} · {{ i.type }}</td>
<td>{{ i.location }}</td><td>{{ '%.2f' % i.score }} / {{ i.reliability }}{% if not calibrated(i) %}<br><i>{{ uncal }}</i>{% endif %}</td><td>{{ i.explanation }}</td>
<td>{{ i.alt_explanations|join('; ') }}</td><td>{{ recommended(i)|join('; ') }}</td><td>{% if img %}<img src="{{ img }}" alt="{{ i.type }} artifact">{% endif %}</td></tr>{% endfor %}</table>{% endif %}{% endfor %}
<h2>All recommended human checks</h2><ol>{% for c in checks %}<li>{{ c }}</li>{% else %}<li>Spot-check the original files; absence of indicators is not proof of integrity.</li>{% endfor %}</ol>
<h2>Limitations</h2><ul>
<li>Detectors are heuristics tuned on a small team-made test set; reliability labels reflect that, not a guarantee.</li>
<li>Re-compression, resizing and platform uploads can create or hide indicators.</li>
<li>{{ align_rule }}</li>
<li>The language model reads text, compares captions with images and lists visible artifacts with their location; it is never asked to judge authenticity.</li>
<li>Fully generated media without provenance metadata or visible artifacts may show no indicators at all.</li>
<li>Absence of indicators does not prove the media is unaltered.</li></ul>
<h2>Analysis log</h2><pre>{% for e in log %}{{ e }}
{% endfor %}</pre></body></html>""")


def data_uri(path):
    p = Path(path) if path else None
    if not p or not p.exists() or p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        return None
    return f"data:{mimetypes.guess_type(p.name)[0]};base64,{base64.b64encode(p.read_bytes()).decode()}"


ORIGINAL = "Obtain the original file from the source device or account (not a messenger or social-media copy)"
REVERSE = "Reverse image search the image (or keyframes) to find where and when it first appeared"
VOICE = "Compare the voice with a verified recording of the same speaker (pitch, rhythm, breaths, room sound)"
CHECKS = {  # what a person should do next, per indicator type
    "exif_missing": ORIGINAL, "exif_edit_software": ORIGINAL, "reencoded": ORIGINAL, "no_creation_time": ORIGINAL,
    "ela_hotspot": REVERSE, "noise_inconsistency": REVERSE, "copy_move": REVERSE,
    "synthetic_provenance": ORIGINAL + "; verify its Content Credentials with c2patool or contentcredentials.org/verify",
    "spectral_upsampling": REVERSE, "synthetic_visual_artifacts": REVERSE,
    "temporal_spike": ORIGINAL, "face_region_flicker": "Compare the face with verified footage of the same person",
    "lipsync_mismatch": "Watch the speech at reduced speed with sound; ask whether the clip was dubbed or voiced over",
    "audio_splice": ORIGINAL, "audio_click": ORIGINAL, "uniform_pauses": VOICE,
    "synthetic_bandlimit": "Ask how the audio was recorded and shared; " + ORIGINAL[0].lower() + ORIGINAL[1:],
    "synthetic_silence": VOICE, "synthetic_prosody": VOICE,
    "caption_image_mismatch": REVERSE, "caption_frame_mismatch": REVERSE,
    "claim_date_mismatch": "Ask the source when and where it was captured; check the date against independent reports",
    "quote_not_in_transcript": "Listen to the full recording and ask the source where the quote comes from",
    "text_inconsistency": "Ask the author or source to clarify the contradicting statements",
}


def recommended(i):
    """Human checks for one finding: what to do next (by type) and where to look (by location)."""
    return [c for c in (CHECKS.get(i.type), human_check(i)) if c]


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


CREDENTIALS_NOTE = ("The indicator label is based on calibrated statistical checks only; signed Content Credentials are "
                    "a declaration by the tool that made the file and are shown separately.")
AI_SOURCE = {"trainedalgorithmicmedia": "AI-generated", "compositewithtrainedalgorithmicmedia": "partly AI-generated"}


def content_credentials(case, indicators):
    """Banner sentences for evidence files whose C2PA manifest has a valid signature and declares AI generation.
    Read back with c2patool: the indicator text can't tell a signed manifest claim from a loose XMP tag."""
    files = {p.name: p for p in case.evidence()}
    out = []
    for i in indicators:
        if i.type != "synthetic_provenance" or "C2PA manifest verified by c2patool" not in i.explanation or i.file not in files:
            continue
        tool = image._c2patool(files[i.file])
        m = tool and re.search(r"signed by (.*) \(((?:signer )?)(.*)\)$", tool[2])
        what = tool and next((AI_SOURCE[k] for k in AI_SOURCE if k in tool[1]), None)
        if m and what and "failed" not in tool[2]:
            out.append(f"Provenance ({i.file}): this file's signed Content Credentials declare it {what} "
                       f"(signed by {m[1]}; signer {m[3]}).")
    return list(dict.fromkeys(out))


def human_checks(indicators, limit=10):
    """Deduplicated recommended checks for the case, strongest indicators first."""
    ordered = sorted(indicators, key=lambda i: -i.score)
    return list(dict.fromkeys(c for i in ordered for c in recommended(i)))[:limit]


def summary(result, case=None):
    """LLM summary from indicators only; cached on the result so Streamlit reruns don't re-bill the API."""
    if "summary" not in result:
        a, inds = result["assessment"], result["indicators"]
        top = sorted(inds, key=lambda i: -i.score * fusion.weight(i))[:20]  # the summary names the strongest 2-3 anyway
        data = json.dumps({"assessment": a, "n_indicators": len(inds),
                           "indicators": [{**i.to_dict(), "counted_in_label": fusion.calibrated(i)} for i in top]}, default=str)
        r = llm.ask_json(SUMMARY_PROMPT, llm.untrusted("indicators", data), schema=SUMMARY_SCHEMA, name="summary", max_tokens=600,
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
             if cal else "none loaded (all weights 0)")
    calls = [e for e in log if e["action"] == "llm_call" and e["check"] != "transcribe"]
    skipped = sorted({e["check"] for e in log if e["action"] == "check_not_run"})
    semantic = ", ".join(sorted({f"{e['model']}, prompt {e.get('prompt_version', '?')}" for e in calls})) or "none run"
    if skipped:
        semantic += f" (not run: {', '.join(skipped)})"
    from . import pipeline  # here: pipeline imports report
    cov = result.get("coverage") or pipeline.coverage(case, result.get("context", ""))
    a = result["assessment"]
    groups = {"Edited": "signs of local edits, cuts or re-encoding", "Generated": "signs the content was made by a generator",
              "Context": "the caption or claims disagree with the media"}
    group_types = {g: sorted({i.type for i in inds if fusion.group(i) == g}) for g in groups}
    return TEMPLATE.render(missing=fusion.MISSING if fusion.missing() else "", recommended=recommended, calibrated=fusion.calibrated, uncal=fusion.UNCALIBRATED, headline=fusion.headline(a, cov), credentials=content_credentials(case, inds), credentials_note=CREDENTIALS_NOTE, align_rule=fusion.ALIGN_RULE, coverage_line=pipeline.coverage_line(cov), groups=groups,
                           group_types=group_types, file_headline=lambda f: fusion.headline(a, cov, f),
                           conversions=[e for e in log if e["action"] == "converted"],
                           calib=calib, semantic=semantic, provenance=llm.provenance(log), source=llm.source(log), case=case, a=result["assessment"], indicators=inds, rows=[(i, data_uri(i.artifact)) for i in inds],
                           integrity=result["integrity"], ingests=[e for e in log if e["action"] == "ingest"],
                           log=log, checks=checks, summary=summary(result, case))
