# NOTES_STATE: where the build stands (D's working notes)

## Waiting on (blocked, everything else is built)
- **Real test set**: `testset/media/` has only synthetic `ph_*` files. Calibration numbers are stamped SYNTHETIC until real rows land.
- **API key**: `OPENAI_API_KEY` not set on D's machine, so Luna checks (caption↔image, caption↔frames, quote↔transcript, summary)
  have only been run against a mocked client. Confirm the Luna model id for `TRUTHLENS_MODEL`.

## Status (2026-11-19, all phases built)
- Phases 0-5 done and committed; `python test_smoke.py` OK (A/B/C detectors, mocked Luna, prompt injection, wording grep).
- Demo cases on placeholders: demo1 **Moderate** (Strong expected once caption↔image runs with a key), demo2 **Strong** with
  `av_aligned_flags` (picture jump 4.8 s ↔ audio splice 5.0 s), demo3 **No significant indicators**.
- `calibration.json` is from the SYNTHETIC set (git-ignored; re-run `python calibrate.py` on real media).
- Video frames' noise/ELA fire a lot on the test-pattern videos; expect calibration on real video to down-weight or not.

## Module map
| File | Role |
|---|---|
| `app.py` | Streamlit review UI (`streamlit run app.py`, or `python demo.py`) |
| `truthlens/vault.py` | `Case`: SHA-256 on ingest, read-only evidence, artifacts dir, append-only `log.jsonl` |
| `truthlens/pipeline.py` | evidence → analyzers by extension → text claims → crossmodal → fusion |
| `truthlens/analyzers/{image,video,audio,text}.py` | per-medium detectors, each `analyze(path, case) -> list[Indicator]` |
| `truthlens/crossmodal.py` | A/V time linking, caption↔image/frames, claimed date↔metadata, quote↔transcript |
| `truthlens/fusion.py` | noisy-OR over strongest indicator per (file, type), weights from `calibration.json` / reliability |
| `truthlens/llm.py` | the only OpenAI caller: Responses API + JSON schema, disk cache, transcription |
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
