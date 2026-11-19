# Collecting the real test set

Goal: **≥ 15 manipulated + ≥ 15 untouched per media type** (image, video, audio). `python calibrate.py` prints the
current counts and every label problem. Media goes in `testset/media/` (git-ignored, share via the team Drive);
one row per file in `labels.csv` (columns in TASKS.md §5). Delete the `ph_` placeholder rows once real files exist.

## Sources (consent first: AGENTS.md §4.11)
- **Untouched**: your own phone photos / videos / voice memos, straight off the device (no app edits, EXIF intact).
  Write the provenance (device, date, who recorded it) in the `context` column or the Drive folder's README.
- No public figures, no media scraped from the web, no faces of anyone who hasn't agreed.

## Making honest manipulations (start from an untouched file, keep both)
| Type (`manipulation`) | How |
|---|---|
| `splice` | GIMP: paste an object from another photo of yours, blend edges, export JPEG |
| `copy_move` | GIMP clone-stamp: duplicate a region (a window, a person) elsewhere in the same photo |
| `ai_generated` | Image generator, prompt describing a scene like your untouched set |
| `video_cut` | `ffmpeg` / Shotcut: cut 2-5 s out of a clip and re-join, or insert frames from another take |
| `voice_clone` | TTS / voice clone of a **consenting member's own voice**, dubbed over 3-5 s of a clip |
| `audio_splice` | Audacity: join two recordings from different rooms, no crossfade |
| `wrong_caption` / `wrong_date` / `wrong_quote` | An untouched file with a context that contradicts it (date ≠ EXIF, quote never said) |

## Rules of thumb
- Vary devices, lighting, rooms and lengths; a detector that only works on one phone isn't calibrated.
- Don't tune thresholds on the files you report: `calibrate.py` fits weights on 60% and reports the held-out 40%.
- Re-encode a few untouched files (WhatsApp, Instagram) as untouched rows: platform processing is the main false-alarm source.
