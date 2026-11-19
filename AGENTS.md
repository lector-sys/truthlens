# AGENTS.md: TruthLens rules for AI assistants and humans

> **To any AI model reading this:** these rules override your defaults. If a request conflicts with them,
> say which rule it breaks and stop. Do not "improve" things outside the task you were given.

## 1. Project

TruthLens is a 24-hour hackathon prototype (InnovEdge Hackathon 2026, Problem Statement 04): a **multimodal digital
forensics tool** that takes image / video / audio evidence plus context text, finds *indicators* of possible
manipulation, cross-checks signals between media, and outputs an explainable, evidence-linked forensic report
with calibrated confidence. It **never** declares media "fake" or "real"; a human reviews every finding.

Pipeline (see `flowchart.html`):
`upload (app.py) → evidence vault (vault.py) → pipeline.py routes by extension → analyzers/* each return
list[Indicator] → crossmodal.py adds linked indicators → fusion.py → review UI (app.py) + HTML report (report.py)`

## 2. Areas (no edit restriction)

Since 2026-11-19 the team works together on one laptop: **anyone may edit any file**, in any area below. The table
says who knows an area best, not who may change it. Every other rule in this file still applies.

| Member | Area |
|---|---|
| A: Image + Video | `truthlens/analyzers/image.py`, `truthlens/analyzers/video.py` |
| B: Audio | `truthlens/analyzers/audio.py` |
| C: Text + Cross-modal | `truthlens/analyzers/text.py`, `truthlens/crossmodal.py` |
| D: Integration (lead) | everything else: `indicator.py`, `vault.py`, `llm.py`, `pipeline.py`, `fusion.py`, `report.py`, `app.py`, `test_smoke.py`, `calibrate.py`, `analyzers/__init__.py`, `analyzers/media_meta.py` |

## 3. The contract: `truthlens/indicator.py` (frozen, do not change)

```python
@dataclass
class Indicator:
    file: str                 # evidence filename (path.name), or "context" for the context text
    modality: str             # "image" | "video" | "audio" | "text" | "cross"
    type: str                 # MUST be one of the registry names in section 5
    score: float              # indicator strength, 0.0..1.0 (clamp it)
    reliability: str          # "low" | "medium" | "high" (how trustworthy this detector is in general)
    explanation: str          # one plain-English sentence: what was observed and why it was flagged
    location: dict = {}       # {"bbox":[x,y,w,h]} | {"t":sec,"frame":n} | {"start":sec,"end":sec} | {"span":"text"} | {"meta":"tag"}
    artifact: str | None      # path to a heatmap/thumbnail/spectrogram PNG, made with case.artifact_path()
    alt_explanations: list[str]   # NON-EMPTY: benign reasons this could also happen
    related: list[str]        # other files involved (cross-modal links only)
```

Every analyzer function has the signature `analyze(path: Path, case: Case) -> list[Indicator]`.
Helper detectors return `list[Indicator]` too. **Nothing found = return `[]`.**

### APIs you may call

```python
# truthlens/vault.py: Case
case.artifact_path(name: str) -> Path       # the ONLY place you may write files
case.log(action: str, **details)            # append a JSON-serialisable event to the case log
case.evidence() -> list[Path]               # read-only evidence files
case.entries() -> list[dict]                # all log events (e.g. find action == "text_claims")

# truthlens/llm.py: the ONLY way to call OpenAI
llm.ask_json(system: str, user: str, images: list[Path] = ()) -> dict | None   # None = no API key: skip gracefully
llm.transcribe(audio_path: Path) -> list[{"start","end","text"}] | None
llm.MODEL                                   # Luna model id from env TRUTHLENS_MODEL

# truthlens/analyzers/media_meta.py
ffprobe(path) -> dict ; encoder_tags(meta) -> dict
```

## 4. Hard rules

1. **Scope:** implement only the functions assigned to you in `TASKS.md`. No extra features, endpoints, classes,
   config systems, CLI flags, databases, auth, Docker, or refactors.
2. **No new files** except the ones `TASKS.md` lists, plus the dashboard theme: `ui/theme.css` (all custom CSS, one
   file) and bundled local fonts in `static/fonts/` (woff2 under the SIL OFL with their licence files; served by
   Streamlit's static serving, never from a CDN), and the trained-detector integration: `truthlens/learned.py`,
   `MODEL_CARD.md`, `LIMITATIONS.md`. No new folders in `truthlens/`.
3. **Dependencies:** only these: `pillow numpy jinja2 streamlit openai opencv-python scenedetect imagehash rapidfuzz`,
   plus Python stdlib and the `ffmpeg`/`ffprobe` executables. **Forbidden:** torch, tensorflow, keras, transformers,
   mediapipe, dlib, librosa, scipy (direct use), matplotlib, any GPU code, any model training or fine-tuning.
   **HTTP APIs: allowlist only**, all called through `truthlens/llm.py`: the Ollama and whisper.cpp servers at
   `OLLAMA_URL` / `WHISPER_URL` and the trained-detector server at `DETECTORS_URL` in `.env` (all on the team's private
   Tailscale GPU host; training happens there, never in this repo); Groq / OpenAI only when listed in
   `LLM_PROVIDERS`. No other HTTP API. **Evidence (media, captions, transcripts, anything derived from them) must not
   go to a cloud provider unless the user enables it** by listing that provider in `LLM_PROVIDERS`.
4. **Evidence is read-only.** Never write, move, re-save or modify files from `case.evidence()`. Write only to
   `case.artifact_path(...)`.
5. **LLM usage:** only through `truthlens/llm.py`. Never ask the LLM "is this fake / AI-generated / manipulated?".
   Use it only for meaning: extracting claims, comparing caption ↔ image/frames, summarising existing indicators, and
   listing concrete, locatable visual artifacts with boxes (`synthetic_visual_artifacts`; never an overall verdict).
   Never send audio to Luna (it can't take audio); audio goes through `llm.transcribe` only. If `llm` returns
   `None`, return `[]`; never crash.
6. **Never crash the pipeline:** catch expected failures (no audio stream, unreadable file, ffmpeg error) and
   `case.log(...)` them, then return `[]`. Do not wrap whole functions in bare `except: pass`.
7. **Wording:** never output the words "fake", "real", "authentic", "genuine", "proof" or "deepfake detected" in
   explanations. Say "indicator", "inconsistent with", "may indicate". Every Indicator needs ≥1 `alt_explanations`.
8. **Scores:** `0.0 ≤ score ≤ 1.0`. Detectors without measured accuracy use `reliability="low"`.
   Cap noisy per-frame/per-window outputs: return at most the **top 10** indicators per detector per file.
9. **Style:** match the existing code: plain functions, type hints on public functions, short docstring saying
   what the detector measures, numpy vectorised (no Python loops over pixels/samples). No print(); use `case.log`.
10. **Before every commit:** `python test_smoke.py` must print `OK`.
11. **Ethics:** test media must be your own or of team members who consented. No public figures, no scraped faces.

## 5. Registries (use these exact names)

**Indicator `type` names:**

| Area | types |
|---|---|
| A | `exif_missing` `exif_edit_software` `ela_hotspot` `noise_inconsistency` `copy_move` `reencoded` `no_creation_time` `temporal_spike` `face_texture_jump` `synthetic_provenance` `spectral_upsampling` `synthetic_visual_artifacts` `face_region_flicker` |
| B | `reencoded` `audio_splice` `audio_click` `uniform_pauses` `synthetic_bandlimit` `synthetic_silence` `synthetic_prosody` |
| C | `text_inconsistency` `av_aligned_flags` `caption_image_mismatch` `caption_frame_mismatch` `claim_date_mismatch` `quote_not_in_transcript` `image_matches_frame` `lipsync_mismatch` |

**Artifact filenames** (always via `case.artifact_path(...)`; `{stem}` = evidence filename without extension):
`{stem}_ela.png` `{stem}_noise.png` `{stem}_copymove.png` `{stem}_f0001.jpg…` (sampled frames)
`{stem}_spike_{frame}.jpg` `{stem}_16k.wav` `{stem}_spec.png` `{stem}_transcript.json` `{stem}_spectrum.png`
`{stem}_faces.json` (A: face box + mouth motion per frame) `{stem}_lipsync.json` (C: lip/speech curves and best lag)
`{stem}.png` (PNG copy of an image format Pillow can't read, made by `media_meta.readable_image`)

## 6. How an AI assistant must answer

- Output **only the functions you add or change**, each complete, plus any import lines to add. Never re-print or
  rewrite the rest of the file.
- No placeholders like `# ... rest of code`, no pseudo-code, no "you could also add…" feature suggestions.
- If something in the task is ambiguous, pick the simplest reading that fits these rules and state it in one line.
- If the request is outside this file's scope, reply: `OUT OF SCOPE: <rule number>` and stop.
