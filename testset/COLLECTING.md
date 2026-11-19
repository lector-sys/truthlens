# Collecting the real test set

Goal: **≥ 15 manipulated + ≥ 15 untouched per media type** (image, video, audio), plus **≥ 15 each of `ai_generated`
images, `cloned_voice` audio and `ai_video` clips, each from ≥ 3 different tools**. `python calibrate.py` prints the current counts and every label problem. Media goes in `testset/media/` (git-ignored, share via the team Drive);
one row per file in `labels.csv` (columns in TASKS.md §5, plus `generator` = the tool, for `ai_generated` / `cloned_voice` / `ai_video` rows). Delete the `ph_` placeholder rows once real files exist.

## Sources (consent first: AGENTS.md §4.11)
- **Untouched**: your own phone photos / videos / voice memos, straight off the device (no app edits, EXIF intact).
  Write the provenance (device, date, who recorded it) in the `context` column or the Drive folder's README.
- No public figures, no media scraped from the web, no faces of anyone who hasn't agreed.

## Making honest manipulations (start from an untouched file, keep both)
| Type (`manipulation`) | How |
|---|---|
| `splice` | GIMP: paste an object from another photo of yours, blend edges, export JPEG |
| `copy_move` | GIMP clone-stamp: duplicate a region (a window, a person) elsewhere in the same photo |
| `ai_generated` | Image generator, prompt describing a scene like your untouched set. Name the generator in the `generator` column (e.g. `midjourney-v7`, `sdxl`, `gpt-image`); use ≥ 3 generators. Keep the **original download** (full size, metadata intact): no screenshots, no search-result thumbnails, no re-saves. `calibrate.py` flags files under 768 px as likely thumbnails. Add a few that went through WhatsApp / Instagram as extra rows: platforms strip C2PA/XMP, which is the realistic case. |
| `video_cut` | `ffmpeg` / Shotcut: cut 2-5 s out of a clip and re-join, or insert frames from another take |
| `voice_clone` | TTS / voice clone of a **consenting member's own voice**, dubbed over 3-5 s of a clip |
| `cloned_voice` | A whole recording in a cloned voice: clone **only a team member's own voice, with their written agreement**, from their own untouched recording, then say different sentences. Name the tool in `generator`; ≥ 3 tools; keep the tool's original export (no re-recording through a speaker). Add the untouched recording it was cloned from as a `none` row. |
| `ai_video` | A generated talking-head or scene clip (text/image-to-video, face reenactment). Faces only of **consenting team members** (or none). Name the tool in `generator`; ≥ 3 tools; keep the original download. |
| `audio_splice` | Audacity: join two recordings from different rooms, no crossfade |
| `wrong_caption` / `wrong_date` / `wrong_quote` | An untouched file with a context that contradicts it (date ≠ EXIF, quote never said) |

## Consent for voices and faces
Cloning or animating a person needs that person's **explicit, written agreement** for this test set, and only team members
may be cloned or animated. Never public figures, never people who did not agree. Note the agreement (who, date) in the Drive
folder README. Delete the files if anyone withdraws.

The `ph_vid_face_real_*`, `ph_vid_ai_*` and `ph_aud_clone_*` placeholders are smoke-test material only:
`calibrate.py` skips them (they trip older detectors for synthetic reasons), so these categories have **no calibration
data until real files are added**.

## What the `cloned_voice` / `ai_video` rows test
`synthetic_bandlimit` (hard cutoff), `synthetic_silence` (too-clean pauses, no breaths), `synthetic_prosody` (flat pitch and
loudness), `lipsync_mismatch` (mouth motion vs speech) and `face_region_flicker`, plus the image checks on 3 keyframes.
Untouched voice memos and phone videos of team members are the false-alarm side: include quiet rooms, phone calls and
monotone readings among them.

## What the `ai_generated` rows test
`synthetic_provenance` (C2PA / XMP DigitalSourceType / generator names in metadata), `spectral_upsampling` (FFT traces of
upsampling) and `synthetic_visual_artifacts` (vision model lists garbled text, hands, shadows... with boxes). Untouched
photos are the false-alarm side for all three, so include phone photos with text, hands and crowds in the untouched set.

## Rules of thumb
- Vary devices, lighting, rooms and lengths; a detector that only works on one phone isn't calibrated.
- Don't tune thresholds on the files you report: `calibrate.py` fits weights on 60% and reports the held-out 40%.
- Re-encode a few untouched files (WhatsApp, Instagram) as untouched rows: platform processing is the main false-alarm source.
