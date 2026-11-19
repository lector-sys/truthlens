# TruthLens

```
$env:OPENAI_API_KEY = "sk-..."
$env:TRUTHLENS_MODEL = "gpt-5.6-luna"   # check the exact ID in your OpenAI dashboard
pip install -r requirements.txt
python test_smoke.py
streamlit run app.py
```

**Start here:** [TASKS.md](TASKS.md) (project brief + your tasks) · [AGENTS.md](AGENTS.md) (rules for humans and AI) · `flowchart.html`

Every analyzer returns `list[Indicator]` (`truthlens/indicator.py`). That schema is the only contract; don't change it without telling everyone.

| Owner | Files |
|---|---|
| A: Image + Video | `analyzers/image.py`, `analyzers/video.py` |
| B: Audio | `analyzers/audio.py` |
| C: Text + Cross-modal | `analyzers/text.py`, `crossmodal.py` |
| D: Fusion + UI + Report | `fusion.py`, `report.py`, `app.py`, labelled test set |

Evidence is hashed and stored read-only under `cases/<id>/evidence`; every step is appended to `cases/<id>/log.jsonl`.
