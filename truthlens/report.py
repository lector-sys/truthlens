"""OWNER D. HTML forensic report: self-contained (artifacts embedded), human checks, summary from indicators only."""
import base64, json, mimetypes, re, time
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
<style>{{ fonts }}{{ tokens }}
/* light print theme of the dashboard: same tokens (ui/tokens.css), wording and finding layout */
body{font:var(--fs-sm)/var(--lh-body) var(--font-ui);color:var(--print-text);background:var(--print-bg);max-width:var(--w-content);
  margin:auto;padding:var(--s-6) var(--s-5) var(--s-7)}
h1,h2,h3{font-family:var(--font-head);font-weight:var(--fw-semibold);color:var(--print-text);line-height:var(--lh-tight)}
h1{font-size:var(--fs-xl);margin:0 0 var(--s-2)}h2{font-size:var(--fs-lg);margin:var(--s-6) 0 var(--s-3);padding-bottom:var(--s-2);
  border-bottom:var(--bw) solid var(--print-border)}h3{font-size:var(--fs-md);margin:var(--s-5) 0 var(--s-2)}
.mono,code{font-family:var(--font-mono);font-variant-numeric:tabular-nums}.muted{color:var(--print-muted)}
.eyebrow{font:var(--fw-semibold) var(--fs-xs) var(--font-ui);letter-spacing:var(--track);text-transform:uppercase;color:var(--print-muted)}
.box{border:var(--bw) solid var(--print-border);border-radius:var(--r-card);padding:var(--s-3) var(--s-4);margin:var(--s-3) 0;background:var(--print-surface)}
.box.warn{border-color:var(--warning)}.box.danger{border-color:var(--danger)}.box.prov{border-color:var(--mark-detector)}
table{border-collapse:collapse;width:100%}th{font:var(--fw-semibold) var(--fs-xs) var(--font-ui);color:var(--print-muted);text-align:left;
  background:var(--print-surface);padding:var(--s-2) var(--s-3)}td{padding:var(--s-2) var(--s-3);border-top:var(--bw) solid var(--print-border);
  vertical-align:top;font-size:var(--fs-sm);overflow-wrap:break-word;word-break:normal}
.finding{display:grid;grid-template-columns:var(--w-thumb-min) 1fr var(--w-gauge);gap:var(--s-4);padding:var(--s-3) 0;
  border-top:var(--bw) solid var(--print-border);break-inside:avoid}
.finding .id{font:var(--fw-semibold) var(--fs-sm) var(--font-mono);color:var(--print-accent)}.finding .t{font-weight:var(--fw-semibold);font-size:var(--fs-md)}
.finding .meta{font:var(--fs-xs) var(--font-mono);color:var(--print-muted);margin:var(--s-1) 0}.finding p{margin:var(--s-1) 0}
.finding img{width:100%;border-radius:var(--r-chip);border:var(--bw) solid var(--print-border)}
.chip{display:inline-block;font:var(--fw-semibold) var(--fs-xs) var(--font-ui);padding:0 var(--s-2);border:var(--bw) solid currentColor;
  border-radius:var(--r-chip);margin-left:var(--s-2);vertical-align:middle}.chip.counted{color:var(--print-accent)}.chip.uncal{color:var(--print-muted)}
.log td{font:var(--fs-xs) var(--font-mono);overflow-wrap:anywhere}
@media print{h2{break-after:avoid}tr{break-inside:avoid}body{max-width:none;padding:0}}</style>
</head><body>
<div class="eyebrow">TruthLens · forensic report</div>
<h1>Case {{ case.id }}</h1>
{% if missing %}<div class="box danger"><b>{{ missing }}</b></div>{% endif %}
{% for c in credentials %}<div class="box prov"><b>{{ c|e }}</b></div>{% endfor %}
{% if headline.startswith(('Not analysed', 'Failed')) %}<p><b>Label:</b> {{ headline }}. No label is given.</p>
{% else %}<p><b>Label:</b> {{ headline }} (fused strength {{ '%.2f' % a.case.p }}).</p>{% endif %}
{% if credentials %}<p class="muted">{{ credentials_note }}</p>{% endif %}
{% if coverage_line %}<p class="muted">{{ coverage_line }}</p>{% endif %}
<div class="box warn">Automated indicators, not proof. Every finding needs human review.</div>
<p class="muted">Fusion weights: {{ calib }}. Semantic checks: {{ semantic }}.</p>
{% if provenance %}<p class="muted">{{ provenance }} Model answers: {{ source }}.</p>{% endif %}
<h2>Summary</h2><p>{{ summary }}</p>
<h3>Recommended human checks</h3><ol>{% for c in checks[:5] %}<li>{{ c }}</li>{% else %}<li>Spot-check the original files; absence of indicators is not proof of integrity.</li>{% endfor %}</ol>
<h2>Evidence and integrity</h2>
<table><tr><th>File</th><th>SHA-256</th><th>Unchanged since intake</th><th>Label for this file</th></tr>
{% for e in ingests %}<tr><td>{{ e.file }}</td><td class="mono" style="overflow-wrap:anywhere">{{ e.sha256 }}</td><td>{{ 'Yes' if integrity.get(e.file) else 'No: hash changed' }}</td>
<td>{{ file_headline(e.file) }}</td></tr>{% endfor %}</table>
{% for e in conversions %}<p class="muted">{{ e.file }} was analysed via {{ e.to }} (converted with {{ e.tool }}, SHA-256 {{ e.sha256 }}):
compression-based checks (ELA) see the conversion, not the original. The evidence hash is the original file's.</p>{% endfor %}
<h2>Findings</h2>
{% if not indicators %}<p>No findings. That does not show the media is unaltered.</p>{% endif %}
{% for g, what in groups.items() %}{% set grp = rows|selectattr('1.type', 'in', group_types[g])|list %}{% if grp %}
<h3>{{ g }}: {{ what }}</h3>
{% for t, i, img in grp %}<div class="finding"><div class="id">{{ t }}</div><div>
<div class="t">{{ name(i.type) }}<span class="chip {{ 'counted' if calibrated(i) else 'uncal' }}">{{ 'Counted' if calibrated(i) else 'Uncalibrated' }}</span></div>
<div class="meta">{{ i.modality }} · {{ i.file }}{% if i.related %} ↔ {{ i.related|join(', ') }}{% endif %} · {{ where(i.location) }} · score {{ '%.2f' % i.score }} · reliability {{ i.reliability }}</div>
<p>{{ i.explanation }}</p>{% if not calibrated(i) %}<p class="muted">{{ uncal }}</p>{% endif %}
<p><b>Could also be:</b> {{ i.alt_explanations|join('; ') }}</p><p><b>What to check:</b> {{ recommended(i)|join('; ') }}</p></div>
<div>{% if img %}<img src="{{ img }}" alt="{{ name(i.type) }}: evidence artifact for {{ t }}">{% endif %}</div></div>{% endfor %}{% endif %}{% endfor %}
<h2>All recommended human checks</h2><ol>{% for c in checks %}<li>{{ c }}</li>{% else %}<li>Spot-check the original files; absence of indicators is not proof of integrity.</li>{% endfor %}</ol>
<h2>Limitations</h2><ul>
<li>Checks are heuristics tuned on a small team-made test set; reliability labels reflect that, not a guarantee.</li>
<li>Re-compression, resizing and platform uploads can create or hide indicators.</li>
<li>{{ align_rule }}</li>
<li>The language model reads text, compares captions with images and lists visible artifacts with their location; it is never asked to judge authenticity.</li>
<li>Fully generated media without provenance metadata or visible artifacts may show no indicators at all.</li>
<li>Absence of indicators does not prove the media is unaltered.</li></ul>
<h2>Custody log</h2><table class="log"><tr><th>Time</th><th>Action</th><th>Details</th></tr>
{% for e in log %}<tr><td style="white-space:nowrap">{{ when(e.ts) }}</td><td>{{ e.action|replace('_', ' ') }}</td>
<td>{% for k, v in e.items() if k not in ('ts', 'action') %}{{ k }}={{ v }}{% if not loop.last %}, {% endif %}{% endfor %}</td></tr>{% endfor %}</table>
</body></html>""")
TOKENS = Path(__file__).resolve().parent.parent / "ui" / "tokens.css"


def _when(ts):
    lt = time.localtime(ts)
    return f"{lt.tm_mday} {time.strftime('%b %Y, %H:%M:%S', lt)}"


FONTS = Path(__file__).resolve().parent.parent / "static" / "fonts"


def font_faces():
    """@font-face rules with the bundled fonts embedded, so the report stays one self-contained file."""
    return "".join(f'@font-face{{font-family:"{fam}";src:url(data:font/woff2;base64,'
                   f'{base64.b64encode((FONTS / f"{f}.woff2").read_bytes()).decode()}) format("woff2");font-weight:100 900}}'
                   for fam, f in (("Inter", "inter"), ("Space Grotesk", "space-grotesk"), ("JetBrains Mono", "jetbrains-mono"))
                   if (FONTS / f"{f}.woff2").exists())


def where(loc):
    """A finding's location in words: region, time, span, metadata tag."""
    if "bbox" in loc:
        x, y, w, h = loc["bbox"]
        s = f"x{x} y{y} · {w}×{h}px" + (f" ↔ x{loc['bbox2'][0]} y{loc['bbox2'][1]}" if "bbox2" in loc else "")
        return s + (f" @ {loc['t']:.1f}s" if "t" in loc else "")
    if "t" in loc:
        return f"{loc['t']:.2f}s" + (f" · frame {loc['frame']}" if "frame" in loc else "")
    if "start" in loc:
        return f"{loc['start']:.2f}–{loc['end']:.2f}s"
    if "span" in loc:
        return f"“{loc['span']}”"
    return f"metadata: {loc['meta']}" if "meta" in loc else "whole file"


def tagged(indicators):
    """[(tag, finding)]: evidence-tag numbers E-01, E-02... in finding order (contribution to the label, then score),
    the same in the dashboard and the report."""
    order = sorted(indicators, key=lambda i: (-fusion.contribution(i), -i.score))
    return [(f"E-{n:02d}", i) for n, i in enumerate(order, 1)]


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
    "learned_synthetic_image": ORIGINAL + "; " + REVERSE[0].lower() + REVERSE[1:],
    "learned_synthetic_video": ORIGINAL + "; " + REVERSE[0].lower() + REVERSE[1:],
    "learned_synthetic_audio": VOICE + "; " + ORIGINAL[0].lower() + ORIGINAL[1:],
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
    if result.get("failed"):
        return "The analysis did not complete, so there are no findings to summarise."
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
    return TEMPLATE.render(missing=fusion.MISSING if fusion.missing() else "", recommended=recommended, where=where, calibrated=fusion.calibrated, uncal=fusion.UNCALIBRATED, headline=result.get("headline") or fusion.headline(a, cov), credentials=content_credentials(case, inds), credentials_note=CREDENTIALS_NOTE, align_rule=fusion.ALIGN_RULE, coverage_line=pipeline.coverage_line(cov), groups=groups,
                           group_types=group_types, file_headline=lambda f: fusion.headline(a, cov, f),
                           conversions=[e for e in log if e["action"] == "converted"],
                           calib=calib, semantic=semantic, provenance=llm.provenance(log), source=llm.source(log), case=case, a=result["assessment"], indicators=inds, rows=[(t, i, data_uri(i.artifact)) for t, i in tagged(inds)], fonts=font_faces(), tokens=TOKENS.read_text(encoding='utf-8') if TOKENS.exists() else '',
                           when=_when, name=lambda t: re.sub(r'\b(Ela|Exif|Av|Ai|Xmp|C2pa)\b', lambda m: {'Av': 'A/V', 'C2pa': 'C2PA'}.get(m[0], m[0].upper()), t.replace('_', ' ').capitalize()),
                           integrity=result["integrity"], ingests=[e for e in log if e["action"] == "ingest"],
                           log=log, checks=checks, summary=summary(result, case))
