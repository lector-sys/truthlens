# NOTES_STATE: where the build stands (D's working notes)

**Blocked on the real test set: semantic weights are 0 on synthetic data. Demo cases must use real photos and clips.**

## Waiting on (blocked, everything else is built)
- **Real test set**: `testset/media/` has only synthetic `ph_*` files. Calibration numbers are stamped SYNTHETIC until real rows land.
- Nothing else: semantic checks and transcripts now run on D's own GPU server (see below).

## Status (2026-11-19, GPU server connected)
- **Models run on `aprameya`** (home GPU server, reached over Tailscale): Ollama `truthlens-vl` (qwen3-vl 8B Q4_K_M) for
  claims / caption↔image / caption↔frames / summary; whisper.cpp server (large-v3-turbo) for transcripts. Set in `.env`
  (git-ignored, see `.env.example`). Provider order `LLM_PROVIDER`, then `LLM_PROVIDERS`; default **ollama only**.
  Groq/OpenAI run only if listed there (AGENTS §4.3: evidence goes to a cloud provider only if the user enables it).
  **Cache first** (key: evidence bytes + prompt + prompt version + model); then each provider, schema-checked locally and
  retried once; then "not run". `demo.py --fresh` / dashboard "Analyze fresh" bypass the cache. Case record, Overview and
  report say "live" or "result from cache (date)".
- Whisper gets only the 16 kHz mono audio as FLAC plus fixed options; caption/claim text is never sent (smoke-tested).
- Report + Overview state provider/host and "Evidence stayed on the investigator's own machines (no cloud AI used)"
  only when no cloud provider was used. Header pill pings `/api/tags` at load and every 30 s: green = this case used the
  primary provider, amber = a fallback answered for this case, grey = server offline.
- `python demo.py --prewarm`: 38/38 cases (3 demo + 35 testset) fully cached (6 min live; a cached re-run of the 3 demos takes 2 s). `--offline` serves from cache/.
- Demo cases against truthlens-vl (per-stage seconds, warm model):
  | Case | Before (no model) | Now, calibrated | Run time |
  |---|---|---|---|
  | demo1 | Moderate | **Moderate** 0.70 (claim_date_mismatch, exif_edit_software, noise); caption_image_mismatch fires 0.99 but weight 0.0 | image 0.2 · text 2.0 · cross 2.4 · total 6.5 |
  | demo2 | Strong | **Strong** 0.97 (audio_splice, temporal_spike, av_aligned_flags); quote_not_in_transcript 0.58 | video 0.3 · audio 9.2 (whisper 9.1 for 8 s) · text 1.3 · cross 8.5 · total 23.4 |
  | demo3 | No significant | **No significant** 0.03; caption_image_mismatch fires 0.99 but weight 0.0 | image 0.1 · text 1.2 · cross 4.3 · total 8.0 |
- Uncalibrated, demo1 went Strong (0.88) and demo3 Moderate (0.60) on caption_image_mismatch: the placeholder images are
  abstract squares, so "not a library" is a correct observation on *every* placeholder. Calibration on the SYNTHETIC set
  therefore gives the semantic types ~0 weight (caption_image 0.0, caption_frame 0.04, quote 0.18: the placeholder audio
  is tones, transcribed as "*crickets*"). Expect usable weights only once real media replaces `ph_*`.
- `calibration.json` re-fit with truthlens-vl (SYNTHETIC, n=35, held-out caught 4/8, false alarms 1/6).
- `python test_smoke.py` OK (now also: schema-invalid reply retried, provider down → cache, whisper request has no caption).
- Video frames' noise/ELA fire a lot on the test-pattern videos; expect calibration on real video to down-weight or not.
- Whisper runs ~1.1× real time on this server; if that's too slow for live use, check it's on the GPU or use `small`.

## Module map
| File | Role |
|---|---|
| `app.py` | Streamlit review UI (`streamlit run app.py`, or `python demo.py`) |
| `truthlens/vault.py` | `Case`: SHA-256 on ingest, read-only evidence, artifacts dir, append-only `log.jsonl` |
| `truthlens/pipeline.py` | evidence → analyzers by extension → text claims → crossmodal → fusion |
| `truthlens/analyzers/{image,video,audio,text}.py` | per-medium detectors, each `analyze(path, case) -> list[Indicator]` |
| `truthlens/crossmodal.py` | A/V time linking, caption↔image/frames, claimed date↔metadata, quote↔transcript |
| `truthlens/fusion.py` | noisy-OR over strongest indicator per (file, type), weights from `calibration.json` / reliability |
| `truthlens/llm.py` | the only model caller: Ollama/Groq (Chat Completions) or Luna (Responses), local schema check, retry, provider fallback, cache; whisper.cpp transcripts; `provenance()`, `ping()` |
| `truthlens/report.py` | self-contained HTML report |
| `calibrate.py` | labelled test set → per-type precision (Wilson CI), held-out case metrics, `calibration.json` |
| `testset/make_placeholder.py` | synthetic stand-in media |

## Indicator schema (frozen: `truthlens/indicator.py`)
`Indicator(file, modality, type, score, reliability, explanation, location, artifact, alt_explanations, related)`.
The brief's names map as: `media`→`modality`, `benign_explanations`→`alt_explanations`, `artifacts`→`artifact` (one path).
`location`: `{"bbox":[x,y,w,h]}` | `{"t","frame"}` | `{"start","end"}` | `{"span"}` | `{"meta"}`.

## A-tasks (TASKS.md §4, verbatim scope)
A1 `noise_inconsistency` (median-filter residual, 32 px block std, flag outliers, `{stem}_noise.png`) ·
A2 `copy_move` (ORB 3000, knn k=3, ratio 0.75, >40 px apart, ≥10 pairs, `{stem}_copymove.png`) · A3 wire into `image.analyze` ·
A4 `sample_frames` (ffmpeg fps=1, ≤60, `{stem}_f%04d.jpg`) · A5 `frame_forensics` (ELA + noise per frame, top 10) ·
A6 `scene_cuts` (scenedetect ContentDetector) · A7 `temporal_spikes` (160 px gray diffs, robust z>4, >0.25 s from a cut) ·
A8 [stretch] `face_texture_jump` · A9 wire into `video.analyze`.

## Branches
`member-a`, `member-b`, `member-c` are each behind `main` (2 commits on 2026-11-19). B's and C's files were written on main:
B and C must `git pull origin main` before more work. No force-pushes.

## Conflicts between the continuation brief and TASKS.md / AGENTS.md (AGENTS.md wins, per the brief)
| Brief asks | Rule | What was built instead |
|---|---|---|
| FastAPI + Vite/React/TS/Tailwind dashboard, SSE endpoints | AGENTS §4.1 no endpoints, §4.3 allowed deps only | Streamlit dashboard (`app.py`) with the brief's screens, palette and wording; one command `python demo.py` |
| scipy / scikit-image in detectors | §4.3 forbids scipy | numpy + OpenCV + Pillow only |
| `faster-whisper` offline transcription | §4.3 | No offline transcript; quote check reports "not run" without a key |
| wavesurfer.js, bundled Inter/JetBrains/Space Grotesk fonts | §4.3 | Native `st.audio` + spectrogram PNG; system font stack (works offline) |
| Playwright screenshots | §4.3 | Screenshots taken with the browser by hand, if available |
| `claims` as `[{text,type,value,precision,span}]` | TASKS C1 "keep the same JSON keys" | Same keys (`who/what/where/when/quotes/inconsistencies`), now schema-enforced |
| `label ∈ {manipulated, untouched}` | TASKS §5 `real|manipulated` | CSV keeps `real`; the UI and report say "untouched" |
| sha256 + source/licence columns in labels.csv | TASKS §5 fixed columns | `calibrate.py` validates the existing columns; provenance is in `COLLECTING.md` |
| `docs/DEMO_SCRIPT.md` | DEMO.md already owns the script (D6) | DEMO.md updated with the 3-minute click path |
| `get_keyframes(case, n=3)` | C5 already picks 3 frames from A4's artifacts | not added (would duplicate C5) |
| SSE progress endpoint | §4.1 | `pipeline.run(progress=...)` drives `st.status` live per analyzer |
| Browser-side SHA-256 before upload | Streamlit can't run page JS | Server hashes the upload bytes, then re-hashes the stored copy and shows ✓ match |
| Light theme toggle | — | Streamlit's own Settings → Theme; the custom panels are tuned for dark; the HTML report is light |
| j/k + Space keyboard shortcuts | Streamlit widgets | Not built; standard Tab/Enter keyboard navigation works |
| Smoke grep bans "real" | Brief also wants a `REAL` dataset badge | Grep is case-sensitive: lowercase/Title-case banned, all-caps dataset tag allowed |
| `scene_cuts(path)` | TASKS A6 signature | `scene_cuts(path, case)`, so a scenedetect failure is logged instead of crashing the video analyzer |
| Playwright screenshots | §4.3 | Headless Edge via DevTools protocol (scratch script, not in repo); 12 PNGs in `docs/screenshots/` reviewed and fixed |
