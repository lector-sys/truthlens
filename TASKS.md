# TruthLens: Project Brief & Team Tasks

**InnovEdge Hackathon 2026 · Problem Statement 04 · 24 hours · team of 4**

Read sections 1–3 once. Then go straight to **your** section in 4. Rules for every human and AI are in
[`AGENTS.md`](AGENTS.md); if anything here seems to conflict with it, `AGENTS.md` wins.

---

## 1. What we are building (and why)

### The problem
AI tools can now fake photos, videos and voices convincingly. Investigators need a tool that **examines media
systematically, points to exactly what looks suspicious, cross-checks different media against each other,
and is honest about how sure it is**, without pretending an automated score is proof.

### Our answer: TruthLens
You upload a "case": one or more images / videos / audio files plus the context text that came with them
(a caption, a claimed date and place, a quote). TruthLens:

1. **Preserves the evidence.** It hashes every file (SHA-256), stores a read-only copy, and logs every step.
2. **Runs forensic detectors per media type.** These are classic, explainable signal processing on CPU: error level
   analysis, noise patterns, copy-move, metadata, frame jumps, audio splices.
3. **Reads the context text** with the Luna model (OpenAI API) to extract checkable claims.
4. **Cross-checks across media.** Does the caption match the image? Does the claimed date match the metadata?
   Was the quote actually said in the audio? Do a video glitch and an audio splice happen at the same second?
5. **Fuses everything** into a cautious label: *No / Weak / Moderate / Strong indicators of manipulation*.
6. **Produces a report** with each flag's location (box, frame, timestamp, text span), why it was flagged,
   benign alternative explanations, confidence, limitations, and the evidence hashes.

### Design principles (our pitch)
| Principle | What it means in code |
|---|---|
| **Indicator, not verdict** | We never say "fake". Every flag lists `alt_explanations`. |
| **Explainable** | Every detector is a technique we can explain in one sentence and show as a heatmap or timeline. |
| **Honest confidence** | `reliability` per detector, measured on our own labelled test set (calibration table in the demo). |
| **Evidence preserved** | Hashes, read-only copies, append-only log, integrity re-check in the report. |
| **Cross-modal** | Most teams skip this part. It is our edge. |
| **No GPU, no training** | Forensics on CPU; Luna only for language and meaning (claims, caption ↔ image). |

### What the judges score → where we answer it
| Judging point (from PS 04) | Our feature |
|---|---|
| Useful analysis across multiple media types | analyzers for image, video, audio, text |
| Explainable, evidence-linked findings | heatmaps, frame thumbnails, spectrogram marks, text spans |
| Meaningful cross-modal correlation | `crossmodal.py` (5 checks) |
| Calibrated confidence and honest uncertainty | reliability weights + calibration table + limitations |
| Clear reporting, responsible handling of evidence | HTML report, SHA-256, read-only vault, log |

---

## 2. How the code fits together

```
truthlens/
├── app.py                  D  Streamlit UI: upload → review → download report
├── calibrate.py            D  (new) runs the labelled test set, prints the accuracy table for each detector
├── test_smoke.py           D  must print OK before every commit
├── AGENTS.md / TASKS.md       rules / this file
├── flowchart.html             architecture + 24h plan (open in browser)
├── testset/labels.csv     all labels for the test set (media itself is NOT in git, it's shared via Drive)
└── truthlens/
    ├── indicator.py        D  the shared Indicator dataclass (FROZEN)
    ├── vault.py            D  Case: hashing, read-only evidence, artifacts, log
    ├── llm.py              D  OpenAI: ask_json (Luna), transcribe (whisper-1)
    ├── pipeline.py         D  runs analyzers → crossmodal → fusion
    ├── fusion.py           D  noisy-OR over the strongest indicator per type, weighted by reliability
    ├── report.py           D  HTML report
    ├── crossmodal.py       C
    └── analyzers/
        ├── image.py        A
        ├── video.py        A
        ├── audio.py        B
        ├── text.py         C
        └── media_meta.py   D  ffprobe helpers
```

**Order of execution in `pipeline.run`:** each evidence file goes through its analyzer (image / video / audio). For video,
D also extracts the audio track and runs B's `analyze_audio`. Then `text.analyze(context)` runs, then
`crossmodal.analyze(...)`, which can rely on **artifacts already produced** (A's sampled frames, B's
transcripts, C's logged claims). Then fusion runs.

**Shared contracts** are in `AGENTS.md` §3 (Indicator fields, `case.*`, `llm.*`) and §5 (exact type names and
artifact filenames). Use them exactly. That is how our four branches will merge cleanly.

---

## 3. Team, workflow, timeline

| Member | Name | Code area | Non-code duty | Uses Claude? |
|---|---|---|---|---|
| A | _fill in_ | Image + Video | image/video test set | no |
| B | _fill in_ | Audio | audio test set + demo voice clip | no |
| C | _fill in_ | Text + Cross-modal | context texts + pitch deck | no |
| D | _fill in_ | Integration, fusion, UI, report, calibration | merges, demo script | **yes** |

### Setup (each member, 10 min)
```powershell
git clone <repo-url>; cd truthlens
python -m pip install pillow numpy jinja2 streamlit openai opencv-python scenedetect imagehash rapidfuzz
# ffmpeg + ffprobe must be on PATH:  ffprobe -version
$env:OPENAI_API_KEY = "<from D>"; $env:TRUTHLENS_MODEL = "<Luna model id from D>"
python test_smoke.py          # must print OK
git checkout -b member-a      # member-b / member-c ; D works on main
```

### Git rules
- Work only on your branch, commit only your own files (`AGENTS.md` §2).
- Commit message: `A: noise_inconsistency detector`. Small commits.
- Push and tell D at every sync point; D merges into `main`. Then you run `git pull origin main` and `git merge main`.
- Never commit media files, API keys, or `cases/`.

### Sync points (15 min, whole team)
| Hour | Everyone must have |
|---|---|
| 1 | Read this file; branches created; setup works; test-set collection started |
| 6 | Each analyzer returns ≥1 new real indicator on a real test file; merged to main |
| 12 | Full pipeline runs on the whole test set without errors; all must-haves merged |
| 16 | **Feature freeze for new detectors.** Calibration run; weights tuned |
| 19 | Stretch goals only if D approves; UI and report polish |
| 21 | **Code freeze.** Rehearse demo ×3, record backup video |
| 23–24 | Buffer |

### Definition of done (every task)
1. The function exists with the exact signature given below.
2. It returns `[]` (not a crash) on files where it doesn't apply.
3. On at least one **manipulated** test file it flags the right region, frame or time, and on at least one **real** file it doesn't.
4. `python test_smoke.py` prints `OK`.
5. Pushed on your branch, D told in chat.

---

## 4. Individual tasks

Priority: **[MUST]** has to be done by hour 12 · **[SHOULD]** by hour 16 · **[STRETCH]** only after D approves at hour 19.

### Member A: Image + Video forensics
**Files:** `truthlens/analyzers/image.py`, `truthlens/analyzers/video.py`
**Already working:** `exif()`, `ela()` in image.py; ffprobe metadata in video.py.

**A1 [MUST] `noise_inconsistency(path: Path, img: Image.Image, case: Case, block: int = 32, z: float = 3.0) -> list[Indicator]`**
- Measures: whether some regions have different sensor noise from the rest (a sign of pasted content).
- Method: `g = img.convert("L")`; residual = `abs(np.asarray(g, float32) - np.asarray(g.filter(ImageFilter.MedianFilter(3)), float32))`;
  per-block standard deviation (reshape trick as in `ela()`); flag blocks with `abs(block - mean) > z * std` (both too clean and too noisy).
- Output: one Indicator, type `noise_inconsistency`, reliability `low`, `location={"bbox":[x,y,w,h]}` around flagged
  blocks, artifact `{stem}_noise.png` (residual heatmap, normalised 0–255),
  alt: `["Mixed textures (sky vs foliage)", "Denoise/beauty filter", "Upscaling or resizing"]`. Score `min(1, max_dev/10)` in σ units.

**A2 [MUST] `copy_move(path: Path, img: Image.Image, case: Case, min_matches: int = 10, min_dist: int = 40) -> list[Indicator]`**
- Measures: the same patch of the image appearing twice (cloned/duplicated region).
- Method: `cv2.ORB_create(nfeatures=3000)` on the grayscale numpy image → `cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des, des, k=3)`;
  for each match list, skip the self-match (`m.queryIdx == m.trainIdx`), apply ratio test `m1.distance < 0.75 * m2.distance`,
  keep pairs whose keypoints are > `min_dist` px apart. Need ≥ `min_matches` pairs.
- Output: type `copy_move`, reliability `medium`, `location={"bbox": src_box, "bbox2": dst_box}`, artifact `{stem}_copymove.png`
  (image with `cv2.line` between matched points), score `min(1, n_pairs/50)`,
  alt: `["Repeating patterns (tiles, windows, fabric)", "Symmetric objects"]`.

**A3 [MUST]** Update `image.analyze` to return `exif + ela + noise_inconsistency + copy_move`.

**A4 [MUST] `sample_frames(path: Path, case: Case, fps: int = 1, max_frames: int = 60) -> list[tuple[float, Path]]`**
- `subprocess.run(["ffmpeg","-v","error","-i",str(path),"-vf",f"fps={fps}","-frames:v",str(max_frames), str(case.artifact_path(f"{path.stem}_f%04d.jpg"))])`
- Return `[(t_seconds, frame_path), ...]` sorted, where frame `i` (1-based) has `t = (i - 1) / fps`.

**A5 [MUST] `frame_forensics(path: Path, frames: list[tuple[float, Path]], case: Case) -> list[Indicator]`**
- For each frame: run `image.ela` and `image.noise_inconsistency` on it, then re-tag each result with
  `dataclasses.replace(ind, file=path.name, modality="video", location={**ind.location, "t": t, "frame": i})`.
- Return the **top 10 by score** only.

**A6 [SHOULD] `scene_cuts(path: Path) -> list[float]`**: `from scenedetect import detect, ContentDetector`;
  `[s.get_seconds() for s, _ in detect(str(path), ContentDetector())]`.

**A7 [SHOULD] `temporal_spikes(path: Path, case: Case, cuts: list[float], z: float = 4.0) -> list[Indicator]`**
- Measures: sudden frame-to-frame changes that are **not** scene cuts (splices, face-swap flicker).
- Method: `cv2.VideoCapture`; resize each frame to 160 px wide grayscale; `d[i] = mean(abs(f[i] - f[i-1]))`;
  robust z = `(d - median) / (1.4826 * MAD)`; flag frames over `z` that are more than 0.25 s from any cut.
- Output: type `temporal_spike`, reliability `low`, `location={"t": sec, "frame": n}`, artifact `{stem}_spike_{frame}.jpg`
  (that frame), score `min(1, z_value/10)`, top 10, alt: `["Camera flash or lighting change", "Fast motion", "Compression keyframe"]`.

**A8 [STRETCH] `face_texture_jump(...)`**: OpenCV Haar cascade
  (`cv2.data.haarcascades + "haarcascade_frontalface_default.xml"`) on sampled frames; Laplacian variance of the largest
  face crop; flag big jumps between neighbouring frames. Type `face_texture_jump`, reliability `low`.

**A9 [MUST]** Update `video.analyze` to return `metadata + frame_forensics(sample_frames(...)) + temporal_spikes(...)`.

**A non-code [MUST by hour 6]:** collect into `testset/media/` and add rows to `testset/labels.csv` (§5):
≥15 real phone photos (EXIF intact), ≥10 edited images (GIMP: splice and clone-stamp), ≥10 AI-generated images,
5 real short videos (10–30 s), 3 edited videos (cut and re-join, or insert a few frames).

**Out of scope for A:** any neural network / AI-image classifier, face recognition, uploading images anywhere.

---

### Member B: Audio forensics
**File:** `truthlens/analyzers/audio.py` · **numpy + stdlib only (no librosa)**
**Already working:** ffprobe metadata (`reencoded`).

**B1 [MUST] `decode(path: Path, case: Case) -> Path | None`**
- `ffmpeg -v error -y -i <path> -vn -ac 1 -ar 16000 <case.artifact_path(f"{stem}_16k.wav")>`.
  Return the wav path, or `None` (and `case.log("no_audio", file=...)`) if ffmpeg fails or there is no audio stream.
  Must work on **video files too** (D calls it for videos).

**B2 [MUST] `load(wav: Path) -> tuple[np.ndarray, int]`**: stdlib `wave` + `np.frombuffer(..., np.int16) / 32768` → float32 mono, sample rate.

**B3 [MUST] `splice_detection(y: np.ndarray, sr: int, file_name: str) -> list[Indicator]`**
- Measures: (a) **noise-floor jumps**: background noise suddenly changing level, typical when two recordings are joined;
  (b) **clicks**: a sample-level discontinuity at a hard cut.
- (a) Frame RMS in dB (25 ms frames, 10 ms hop, vectorised with `np.lib.stride_tricks.sliding_window_view`);
  noise floor per 0.5 s window = 10th percentile of frame dB; flag window boundaries where adjacent floors differ by > 6 dB.
  → type `audio_splice`, reliability `low`, `location={"start": t-0.25, "end": t+0.25}`, score `min(1, jump_db/20)`.
- (b) `dx = abs(diff(y))`; robust z (median/MAD); flag samples with z > 8 that are not inside loud speech (frame dB below
  the median) → type `audio_click`, reliability `low`, same location format.
- Merge flags closer than 0.3 s; top 10. alt: `["Speaker moved / door opened", "Automatic gain control", "Microphone bump"]`.

**B4 [MUST] `spectrogram(y: np.ndarray, sr: int, flags: list[Indicator], out: Path) -> Path`**
- numpy STFT (512-sample Hann window, hop 160) → `20*log10(abs+1e-9)` → normalise to 0–255 → `Image.fromarray`, flip vertically
  (low frequencies at the bottom), resize to ≤1600 px wide; draw a red translucent rectangle (`ImageDraw`) over each flag's
  start–end. Save to `{stem}_spec.png`; set it as `artifact` on every flag.

**B5 [MUST] transcript.** `segs = llm.transcribe(wav)`; if not None, write `json.dumps(segs)` to
  `case.artifact_path(f"{stem}_transcript.json")` and `case.log("transcript", file=file_name, segments=len(segs))`.
  **No indicator.** C uses this file. (Name the transcript with the *original* evidence stem, e.g. `clip_transcript.json` for `clip.mp4`.)

**B6 [MUST] `analyze_audio(wav: Path, case: Case, file_name: str, stem: str) -> list[Indicator]`**: load → splice_detection
  → spectrogram → transcript. Then `analyze(path, case)` = metadata + `decode` + `analyze_audio(wav, case, path.name, path.stem)`.

**B7 [STRETCH] `uniform_pauses(y, sr, file_name) -> list[Indicator]`**: find silent gaps (frame dB < floor + 6 for ≥150 ms);
  if ≥6 pauses and the coefficient of variation of pause lengths is < 0.2, flag `uniform_pauses`, reliability `low`, score 0.3
  (synthetic speech often pauses with machine regularity).

**B non-code [MUST by hour 6]:** ≥10 real voice recordings of consenting members (phone), ≥5 TTS/voice-clone clips
**of a consenting member's own voice**, ≥5 spliced recordings (Audacity: join two clips from different rooms).
Plus the **demo case 2 video**: record a member talking, then replace 3–5 s of the audio with their cloned voice.

**Out of scope for B:** voice-clone classifiers, ML models, speaker identification, sending audio to Luna.

---

### Member C: Context text + Cross-modal correlation
**Files:** `truthlens/analyzers/text.py`, `truthlens/crossmodal.py`
**Already working:** Luna claim extraction + regex fallback (`text.py`); `link_timestamps`, `caption_vs_images` (`crossmodal.py`).

**C1 [MUST]** In `CLAIMS_PROMPT`, require `when` items as ISO dates (`YYYY-MM-DD`, or `YYYY-MM` / `YYYY` if that's all
the text gives) and `where` as plain place names. Keep the same JSON keys.

**C2 [MUST] `_claims(case: Case) -> dict`**: return the most recent `case.entries()` item with `action == "text_claims"`
(minus `ts`/`action`), or `{}`.

**C3 [MUST] `claimed_date_vs_metadata(claims: dict, case: Case) -> list[Indicator]`**
- Capture date per evidence file: image → `img.getexif().get_ifd(0x8769).get(36867)` (DateTimeOriginal), else tag 306,
  format `"YYYY:MM:DD HH:MM:SS"`; video/audio → `encoder_tags(ffprobe(p)).get("creation_time")` (ISO).
- Compare against each claimed date **at the claim's precision** (year / month / day). Mismatch → type
  `claim_date_mismatch`, reliability `medium`, score 0.6, `location={"span": claimed}`, `related=["context"]`,
  explanation names both dates, alt: `["Camera clock set wrong", "Metadata rewritten on export", "Claim refers to publication date, not capture"]`.
- Files with no date → skip (A already flags missing EXIF).

**C4 [MUST] `quote_vs_transcript(claims: dict, case: Case) -> list[Indicator]`**
- For each evidence file `p` with `case.artifact_path(f"{p.stem}_transcript.json")` existing: join segment texts;
  for each quote `rapidfuzz.fuzz.partial_ratio(quote.lower(), transcript.lower())`.
- Ratio < 70 → type `quote_not_in_transcript`, reliability `medium`, score `(100 - ratio)/100`, file `p.name`,
  `location={"span": quote}`, `related=["context"]`, alt: `["Paraphrased quote", "Transcription error", "Quote from a different recording"]`.

**C5 [MUST] `caption_vs_frames(context_text: str, case: Case) -> list[Indicator]`**: for each video, take 3 evenly spaced
frames from A's `{stem}_f*.jpg` artifacts (`sorted(case.artifact_path("").glob(f"{p.stem}_f*.jpg"))`), make **one**
`llm.ask_json(CAPTION_PROMPT, ..., images=frames)` call, and emit type `caption_frame_mismatch` the same way as `caption_vs_images`.

**C6 [MUST]** Update `crossmodal.analyze` to return
`link_timestamps + caption_vs_images + caption_vs_frames + claimed_date_vs_metadata + quote_vs_transcript`
(each claim-based check uses `_claims(case)`).

**C7 [STRETCH] `image_matches_frame(case) -> list[Indicator]`**: `imagehash.phash` of each image vs each sampled frame;
Hamming distance ≤ 8 → type `image_matches_frame`, reliability `medium`, score 0.2, `related=[video name]`
(this links the still to the video so reviewers compare their metadata).

**C non-code:**
- [MUST by hour 6] write the `context` column for every test-set row and the 3 demo cases (caption, claimed date/place, a
  quote). Include **wrong** captions, dates and quotes on purpose for the manipulated cases.
- [MUST by hour 19] the pitch deck: problem → architecture (flowchart) → live demo → calibration table → limitations → ethics.

**Out of scope for C:** asking Luna whether media is fake or AI-made, fact-checking against the web, AI-text detectors,
lip-sync (unless D approves at hour 19).

---

### Member D: Integration lead (uses Claude)
**Files:** everything not owned by A, B or C.

- **D1 [DONE]** Video files are routed to both `video.analyze` and `audio.analyze` (`analyzers/__init__.py`), so audio
  flags from videos reach `link_timestamps`. **B:** this is why `audio.analyze` must accept video files.
- **D2 [MUST]** `calibrate.py` (new): read `testset/labels.csv`, run each row as its own case with its `context`, and print
  a markdown table per indicator `type`: fired on manipulated / fired on real / precision; plus case-level label vs truth.
  The output goes on a slide.
- **D3 [MUST]** `fusion.py`: per-type reliability overrides from the calibration numbers.
- **D4 [MUST]** `app.py`: per-modality views: image with heatmap artifacts side by side; video flagged-frame thumbnails
  sorted by `t`; audio `st.audio` + spectrogram; cross-modal table; integrity status.
- **D5 [MUST]** `report.py`: embed artifacts as base64; "Recommended human checks" section; LLM summary written
  **only from the indicator JSON** (no new claims).
- **D6** Merge branches at every sync point, keep `test_smoke.py` green, own the demo script and backup video.

---

## 5. Test set (everyone)

```
testset/
├── labels.csv      # committed
└── media/          # NOT in git: share via the team Drive folder
```

`labels.csv` columns (one row per file):

```csv
file,label,manipulation,case,context,owner
img_real_01.jpg,real,none,,"Photo of the campus library, taken 2026-11-02",A
img_splice_01.jpg,manipulated,splice,,"Flood waters reach the campus library on 2026-11-02",A
clip_voice_01.mp4,manipulated,voice_clone,demo2,"Interview recorded 2026-11-10. He said ""we cancelled the fest""",B
```

- `label`: `real` | `manipulated`
- `manipulation`: `none` | `splice` | `copy_move` | `ai_generated` | `video_cut` | `voice_clone` | `audio_splice` | `wrong_caption` | `wrong_date` | `wrong_quote`
- `case`: `demo1` | `demo2` | `demo3` for the three demo cases, otherwise empty.
- **Consent:** only media of yourselves or members who agreed. No public figures, no faces scraped from the web.

### The 3 demo cases
1. **demo1: spliced photo plus false caption.** Expect copy-move/ELA/noise flags, caption mismatch, date mismatch → *Strong*.
2. **demo2: video with a cloned-voice segment.** Expect audio splice + video temporal flag at the same second (linked),
   quote not in transcript → *Moderate/Strong*.
3. **demo3: untouched phone photo + honest caption.** Expect *No significant indicators* plus limitations. This proves we don't flag everything.

**Never cut:** evidence vault · one heatmap or timeline per modality · ≥2 cross-modal checks · limitations · report.
**Cut first if late:** A8 → C7 → B7 → A6/A7 (keep A4/A5).

---

## 6. Using another AI model (ChatGPT, Codex, Gemini, Copilot…)

Tools that read repo files (Codex, Cursor, Copilot agent mode, Claude Code) pick up `AGENTS.md` automatically. For a
**web chat**, start every new conversation with these three steps:

1. Paste the **entire `AGENTS.md`**.
2. Paste your **current file(s)** (only the ones you own).
3. Paste **your directive block** below, with the `TASK:` line edited to the one task you're doing now.

One task per conversation. When the answer arrives, check it against "Definition of done" (§3) before committing.

### Directive: Member A
```text
ROLE: Member A of TruthLens (image + video forensics). Follow AGENTS.md strictly; it overrides your defaults.
EDITABLE FILES: truthlens/analyzers/image.py, truthlens/analyzers/video.py. Do not modify, rewrite or re-print any other file.
ALLOWED IMPORTS: stdlib, numpy, PIL (Pillow), cv2 (opencv-python), scenedetect, and project modules (..indicator, .media_meta, . import image).
FORBIDDEN: torch/tensorflow/transformers/mediapipe/dlib/scipy/matplotlib, any ML model or classifier, any network call, writing outside case.artifact_path(), modifying evidence files.
TASK: <paste exactly one task from TASKS.md §4 Member A, e.g. "A1 noise_inconsistency" with its full bullet text>
OUTPUT FORMAT:
  1. Import lines to add (if any).
  2. The complete new/changed function(s) only, with type hints and a one-line docstring saying what it measures.
  3. One line: how to test it on a file in testset/media/.
CONSTRAINTS: return list[Indicator]; [] when nothing found; Indicator.type must be one of AGENTS.md §5 for member A; score in [0,1];
  non-empty alt_explanations; top 10 max per detector; vectorised numpy, no per-pixel Python loops; no print().
If my request goes beyond this TASK, reply "OUT OF SCOPE: <AGENTS.md rule>" and stop.
```

### Directive: Member B
```text
ROLE: Member B of TruthLens (audio forensics). Follow AGENTS.md strictly; it overrides your defaults.
EDITABLE FILES: truthlens/analyzers/audio.py only. Do not modify, rewrite or re-print any other file.
ALLOWED IMPORTS: stdlib (wave, json, subprocess, pathlib), numpy, PIL (Image, ImageDraw), and project modules (..indicator, .. import llm, .media_meta).
FORBIDDEN: librosa/scipy/torch/transformers/matplotlib/pydub, any ML or voice-clone classifier, sending audio to Luna/ask_json,
  any network call except llm.transcribe, writing outside case.artifact_path(), modifying evidence files.
TASK: <paste exactly one task from TASKS.md §4 Member B, e.g. "B3 splice_detection" with its full bullet text>
OUTPUT FORMAT:
  1. Import lines to add (if any).
  2. The complete new/changed function(s) only, with type hints and a one-line docstring saying what it measures.
  3. One line: how to test it on a file in testset/media/.
CONSTRAINTS: return list[Indicator]; [] when nothing found or no audio; Indicator.type must be one of AGENTS.md §5 for member B;
  location = {"start": sec, "end": sec}; score in [0,1]; non-empty alt_explanations; merge flags < 0.3 s apart; top 10 max;
  vectorised numpy (sliding_window_view), no per-sample Python loops; no print().
If my request goes beyond this TASK, reply "OUT OF SCOPE: <AGENTS.md rule>" and stop.
```

### Directive: Member C
```text
ROLE: Member C of TruthLens (context text + cross-modal correlation). Follow AGENTS.md strictly; it overrides your defaults.
EDITABLE FILES: truthlens/analyzers/text.py, truthlens/crossmodal.py. Do not modify, rewrite or re-print any other file.
ALLOWED IMPORTS: stdlib, numpy, PIL, rapidfuzz, imagehash, and project modules (.indicator, . import llm, .analyzers.media_meta).
FORBIDDEN: asking the LLM whether media is fake/AI-generated/edited, web search or fact-checking APIs, AI-text detectors, torch/transformers/CLIP,
  calling OpenAI except via llm.ask_json, writing outside case.artifact_path().
TASK: <paste exactly one task from TASKS.md §4 Member C, e.g. "C4 quote_vs_transcript" with its full bullet text>
OUTPUT FORMAT:
  1. Import lines to add (if any).
  2. The complete new/changed function(s) or prompt constant only, with type hints and a one-line docstring.
  3. One line: how to test it on a test-set case.
CONSTRAINTS: return list[Indicator]; [] when inputs are missing or llm returns None; Indicator.type must be one of AGENTS.md §5 for member C;
  cross-modal indicators set related=[...] to the other source ("context" or a filename); score in [0,1]; non-empty alt_explanations;
  LLM prompts must request JSON and must say "Do NOT judge whether the media is AI-generated or edited."; no print().
If my request goes beyond this TASK, reply "OUT OF SCOPE: <AGENTS.md rule>" and stop.
```

### Reviewing AI output: 30-second checklist
- [ ] Only my file(s) changed (`git status`)
- [ ] No forbidden import (`AGENTS.md` §4.3)
- [ ] `type` name is in the registry, artifact name matches the pattern
- [ ] No "fake/real/authentic" wording; `alt_explanations` filled
- [ ] Works on one manipulated and one real test file
- [ ] `python test_smoke.py` → `OK`
