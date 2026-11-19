# NOTES_STATE: where the build stands (D's working notes)

**Blocked on the real test set: semantic weights are 0 on synthetic data. Demo cases must use real photos and clips.**

## Problem-statement coverage (PS04, 2026-11-19)
| Requirement | Covered by | Status / limits |
|---|---|---|
| Edited images (splice, clone, retouch) | `ela_hotspot` `noise_inconsistency` `copy_move` `exif_*` | built, calibrated on SYNTHETIC only |
| AI-generated images | `synthetic_provenance` (C2PA via c2patool, XMP, generator names) `spectral_upsampling` `synthetic_visual_artifacts` | built; visual artifacts uncalibrated (weight 0) |
| Edited video (cuts, re-joins) | `temporal_spike` (not at scene cuts) · per-frame ELA/noise · `reencoded` `no_creation_time` | built |
| Deepfake / synthetic video | `lipsync_mismatch` (Haar face + mouth motion vs speech envelope, offset > 150 ms or r < 0.25) · `face_region_flicker` · image checks on 3 keyframes · `synthetic_provenance` on the file | built; **uncalibrated (weight 0)**; needs a face in > 50% of frames, else "not run: no face"; tested on generated cartoons only |
| Spliced audio | `audio_splice` `audio_click` | built |
| Synthetic / cloned speech | `synthetic_bandlimit` (hard cutoff, native rate) · `synthetic_silence` (digital / too-clean pauses, no breaths) · `synthetic_prosody` (flat pitch + loudness per sentence) | built; **uncalibrated (weight 0)**; no voice-identity comparison (would need a reference recording + speaker model) |
| Text inputs | caption/claims box **and** `.txt` / `.md` / pasted text as evidence (own hash + custody); claims cross-checked against the media | built; no AI-authorship detection (out of scope by design, TASKS.md) |
| Cross-media consistency | `av_aligned_flags` `caption_image_mismatch` `caption_frame_mismatch` `claim_date_mismatch` `quote_not_in_transcript` `lipsync_mismatch` | built |
| Explainable, located findings | every Indicator: location, explanation, benign alternatives, artifact; Edited / Generated / Context groups | built |
| Calibrated confidence | `calibrate.py` (Wilson CIs, held-out split); uncalibrated types weigh 0 | **blocked on the real test set** |
| Human review | "Recommended human checks" per finding (by type + location) and in the summary (dashboard + report) | built |
| Evidence preservation | SHA-256 on ingest, read-only copies, append-only log, conversions logged, integrity re-check | built |
| Honest coverage | "Not analysed" / "Limited analysis" + "Checks run: X of Y. Not run: ..." | built |

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

## AI-generated images + coverage honesty (2026-11-19)
- **Labels** are now "No significant / Weak / Moderate / Strong indicators of manipulation or synthetic generation".
  No analyzer ran → **"Not analysed: <why>"** (grey, no gauge). Checks skipped and nothing found → **"Limited analysis: no
  significant indicators"**. Overview and report show "Checks run: X of Y. Not run: ... (why)" (`pipeline.coverage`,
  recomputed from the log on load, so old cases relabel too: case_20261119_190051 now says Not analysed).
- Findings are grouped **Edited / Generated / Context** (`fusion.group`) on the Evidence and Findings screens and in the report.
- Formats: `.avif .heic .heif .jxl .jp2 .gif` route to the image analyzer. Pillow 13 reads AVIF; anything it can't read is
  converted by ffmpeg to `artifacts/{stem}.png` (`media_meta.readable_image`, logged as `converted` with its sha256).
  The evidence hash stays the original's; EXIF is reported "not run"; ELA carries a conversion note.
- New A-types: `synthetic_provenance` (byte scan for C2PA/JUMBF, XMP DigitalSourceType, generator names; c2patool used if on
  PATH, not installed here → "C2PA manifest present (not verified)"), `spectral_upsampling` (residual FFT: off-centre peaks
  + Nyquist-tail ratio; untouched 0.85-6, 2x upsampled 0.02-0.22; `{stem}_spectrum.png`), `synthetic_visual_artifacts`
  (truthlens-vl lists located artifacts, 0-1000 boxes converted to pixels; violet overlay in the viewer; on a test sign
  with garbled text it boxed it within 15 px).
- Test set: `generator` column; `ai_generated` rows need it; calibrate warns below 15 rows / 3 generators and flags files
  under 768 px as likely thumbnails. 6 `ph_img_ai_*` placeholders (XMP tag, SD parameters chunk, 2x upsampled).
- Calibration (SYNTHETIC, n=41): synthetic_provenance 0.53, spectral_upsampling 0.41; `synthetic_visual_artifacts` never
  fired on the abstract placeholders, so it has no measured weight (falls back to reliability low = 0.3). Held-out caught
  6/10, false alarms 0/6. Real generator output is needed before any of these numbers mean anything.
- Fresh re-runs of the two uploads (new cases `*_fresh`, originals kept for custody): both "Limited analysis: no
  significant indicators", 2 of 5 checks (no caption); provenance: nothing; spectral: tail 1.23 / 0.92, no peaks;
  visual artifacts: none listed; only exif_missing (0.20).
- llm: timeouts are now retried once (APITimeoutError subclasses APIConnectionError, which skips straight to the next provider).
- **Uncalibrated = not counted**: once calibration.json exists, a type it has no weight for gets weight 0 and its findings
  say "uncalibrated (not counted in label)" (cards, Findings table, report). Today that is `synthetic_visual_artifacts`
  (never fired on placeholders) and `exif_missing` (0.20 < the 0.25 firing threshold). No calibration file at all →
  reliability defaults, as before.
- **60 s timeout on the .avif call (corrected)**: a network/connection stall, *not* queueing. The home PC's Ollama log
  (19:30-19:46) shows every request came from this laptop, none took > 5 s, and nothing reached the server between 18:19
  and 19:36: the two calls that timed out (visual artifacts, then summary) never arrived. At the time Ollama was called
  through the openai SDK's shared, pooled client (60 s read timeout, timeouts not retried), so a dead pooled connection
  or a stalled Tailscale path would hang the full 60 s; which of the two can't be told from the data we have. An
  earlier note here blamed "another client's generation": that was wrong. The 46.6 s / 2,500-token measurement was my
  own reproduction test (19:44:19 in the server log); it shows how queueing behaves, it is not what happened.
  Image size (600x400, 56 KB) and output length (7 tokens) were never the issue.
- **Client-side timing + faster retry (after the correction)**: every GPU-server call (Ollama and whisper) goes through
  `llm._post`: a fresh connection per attempt (never pooled), connect timeout 3 s, time-to-first-byte 20 s for model
  calls (whisper: 30 s + audio length), 60 s for any later read. Each call logs `connect_s`, `ttfb_s`, `total_s`
  (client) next to `queue_s`, `load_s`, `prefill_s`, `gen_s` (server). A failure is logged as `llm_attempt_failed` with
  `network: true` and the `stage` it stalled at (connect / first byte / read), then retried once on a new connection.
- **Streamed Ollama + /api/ps probe** (replaces guessing from the 20 s limit): first chunk = prefill done (≤ 20 s),
  then ≤ 10 s idle between chunks, no total cap. On any timeout the client GETs /api/ps (3 s) before retrying:
  answers with a loaded model → verdict "busy", no answer → "network" (refused/reset → network without probing).
  /api/ps lists loaded models and truthlens-vl stays loaded, so "busy" = reachable but slow. Live (demo2 media):
  connect 0.07-0.16 s, first chunk 0.26-2.5 s, total 1.2-4.1 s, queue ≤ 0.16 s, no warnings. Smoke-tested against a
  fake server: healthy 1.6 s stream with 0.5 s idle limit passes; hung chat + answering /api/ps → busy; stream that
  stops after one chunk → "idle"; silent socket → network, retried on a new connection.
- **c2patool 0.27.23 installed on the laptop** (official contentauth/c2pa-rs Windows release, sha256 checked against
  GitHub's digest) in `%LOCALAPPDATA%\Programs\c2patool\c2patool`, on the user PATH and as `C2PATOOL` in `.env`.
  Install steps: README *Setup -> c2patool*. The parser follows c2patool's `validation_state`: Valid = "verified ...
  signature and content hashes valid, signed by <issuer> (signer not checked against a trust list)", Trusted = "... on
  the trust list", else "failed verification (<codes>)". Tested with its test certificate (valid + one flipped byte ->
  `assertion.dataHash.mismatch`); a real ChatGPT original is still to be tested.
- Dashboard warnings (`llm.server_warnings`): "GPU server busy, waited X s" when the server reports `queue_s` > 5 s;
  "GPU server busy (running ...), timed out at <stage>, retried" when a call timed out and /api/ps answered; "GPU server
  didn't respond (network), retried" when /api/ps didn't answer either or the connection was refused/reset.
- No other process on the laptop calls the GPU server (checked 19:5x: only the dashboard; no connections to it).
  Consistent with the server log: this laptop is the only client.

## Scope gaps closed (2026-11-19, later)
- OpenCV pinned to 4.x (`opencv-python<5`, 4.14 installed): 5.x removed Haar cascades. Haar finds a drawn face, so the
  lip-sync smoke test uses a generated cartoon (`make_placeholder.talking_cartoon`), no face images.
- Lip-sync on generated cartoons: in sync 0 ms (r 0.83), audio +200 ms → +200, +300 ms → +320 (40 ms frames), other
  take → r 0.16 (flagged at < 0.25). Face flicker: none on 5 clean clips, catches a 2-frame face flash at 2.40 s.
  Face stats use the upper 60% of the face box (a talking mouth isn't flicker); the last box is kept through a few
  missed detections; spread floored at 1 grey level.
- Band limit at the file's own sample rate (the 16 kHz copy can't show 11-16 kHz cuts): edge = last frequency within
  20 dB of the speech band, needs >= 20 dB drop over +-0.8 kHz and >= 40 dB down above (16-bit floors sit ~55 dB down).
  Silence: pauses <= -85 dBFS or > 50 dB below speech; breaths = bumps > 12 dB in pauses >= 300 ms. Prosody: median
  sentence < 1.2 semitones and < 4 dB. None of these fire on the existing placeholder speech.
- New placeholders: 2 cloned_voice (gated, upsampled 16 → 44.1 kHz), 2 ai_video (audio 300 ms late; other take),
  2 untouched talking cartoons. Cartoons carry sensor noise + room tone so older detectors aren't fed digital
  perfection; they still trip temporal_spike / audio_click (only the mouth moves; abrupt synthetic syllables).
- Summary call: max 600 tokens and only the 20 strongest findings (it overflowed 300 tokens on a busy case).
- Calibration not re-run: the new types weigh 0 until measured on real cloned voices / AI video.

## Module map
| File | Role |
|---|---|
| `app.py` | Streamlit review UI (`streamlit run app.py`, or `python demo.py`) |
| `truthlens/vault.py` | `Case`: SHA-256 on ingest, read-only evidence, artifacts dir, append-only `log.jsonl` |
| `truthlens/pipeline.py` | evidence → analyzers by extension → text claims → crossmodal → fusion |
| `truthlens/analyzers/{image,video,audio,text}.py` | per-medium detectors, each `analyze(path, case) -> list[Indicator]` |
| `truthlens/crossmodal.py` | A/V time linking, caption↔image/frames, claimed date↔metadata, quote↔transcript |
| `truthlens/fusion.py` | noisy-OR over strongest indicator per (file, type), weights from `calibration.json` / reliability |
| `truthlens/llm.py` | the only model caller: Ollama (native /api/chat, per-call timings), Groq (Chat Completions) or Luna (Responses), local schema check, retry, provider fallback, cache; whisper.cpp transcripts; `provenance()`, `source()`, `ping()` |
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
