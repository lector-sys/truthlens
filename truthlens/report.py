"""OWNER D. HTML forensic report. TODO: embed artifacts as base64, LLM summary (from indicators only), PDF export."""
from jinja2 import Template

TEMPLATE = Template("""<!DOCTYPE html><html><head><meta charset="utf-8"><title>TruthLens report {{ case.id }}</title>
<style>body{font-family:sans-serif;max-width:960px;margin:auto}td,th{border:1px solid #ccc;padding:4px;vertical-align:top}table{border-collapse:collapse;width:100%}</style>
</head><body>
<h1>TruthLens forensic report: {{ case.id }}</h1>
<p><b>Overall:</b> {{ a.case.label }} (indicator strength {{ '%.2f' % a.case.p }}).
A score is an indicator, not a verdict. All findings require human review.</p>
<h2>Evidence & integrity</h2>
<table><tr><th>File</th><th>SHA-256</th><th>Unchanged</th><th>Assessment</th></tr>
{% for e in ingests %}<tr><td>{{ e.file }}</td><td><code>{{ e.sha256 }}</code></td><td>{{ integrity.get(e.file) }}</td>
<td>{{ a.files.get(e.file, {}).get('label', 'No significant indicators') }}</td></tr>{% endfor %}</table>
<h2>Flagged evidence</h2>
<table><tr><th>File</th><th>Indicator</th><th>Location</th><th>Score / reliability</th><th>Why</th><th>Alternative explanations</th></tr>
{% for i in indicators %}<tr><td>{{ i.file }}{% if i.related %} ↔ {{ i.related|join(', ') }}{% endif %}</td><td>{{ i.type }}</td>
<td>{{ i.location }}</td><td>{{ '%.2f' % i.score }} / {{ i.reliability }}</td><td>{{ i.explanation }}</td><td>{{ i.alt_explanations|join('; ') }}</td></tr>{% endfor %}</table>
<h2>Limitations</h2><ul>
<li>Pretrained detectors are not calibrated on this evidence's source; treat their scores as weak signals.</li>
<li>Absence of indicators does not prove authenticity.</li></ul>
<h2>Analysis log</h2><pre>{% for e in log %}{{ e }}
{% endfor %}</pre></body></html>""")


def render(case, result):
    log = case.entries()
    return TEMPLATE.render(case=case, a=result["assessment"], indicators=result["indicators"], integrity=result["integrity"],
                           ingests=[e for e in log if e["action"] == "ingest"], log=log)
