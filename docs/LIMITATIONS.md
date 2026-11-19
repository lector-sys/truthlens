# What TruthLens cannot do

TruthLens points a human at places worth checking. It does not decide whether media was manipulated.

- **Fully generated media with no local edits.** Our detectors look for *inconsistencies inside* a file (a pasted region,
  a joined recording, a jump in the picture). An image or voice generated in one piece can be consistent throughout and
  show no indicators. Trained AI-generation detectors run on the GPU host, but they don't count yet (see below).
- **Heavy platform recompression.** WhatsApp, Instagram and similar re-encode and resize. That can erase error-level and
  noise traces, and it can also create them. The `reencoded` indicator says when a file was processed, not by whom.
- **Small test set.** Weights come from a team-made labelled set of tens of files per media type. Every rate is shown
  with a 95% interval, and the intervals are wide. Read the bars, not the dots. The dashboard and report say whether the
  numbers came from the SYNTHETIC placeholder set or the REAL one.
- **Metadata can be rewritten.** A matching EXIF date or creation time does not show the date is right, and a mismatch
  can be an unset camera clock. Those checks say "inconsistent with the claim", nothing more.
- **Language-model checks are semantic only.** Luna reads the caption, extracts claims and compares the caption with
  images and frames. It never judges whether media is edited. Without an API key these checks show *not run*.
- **Transcripts can be wrong.** A quote flagged "not found" may be a transcription error or a paraphrase.
- **No indicators does not mean unaltered.** Every case still needs human review.

## Trained AI-generation detectors (learned_synthetic_image / _audio / _video)

On our own untouched phone and WhatsApp recordings, the first trained detectors flagged 2 of 3 as AI. They were trained
on studio speech and stock footage. They don't count toward the label until calibrated on our own media.

Detectors v2 sharply lowered false alarms on our phone media (held-out: image 1/29, video 0/22, audio 10/114 windows; wide intervals). v2 catches fewer AI videos than v1 (89% vs 98% on the test set), Sora detection fell to 12% on a small sample, and audio false alarms on public untouched test files are 9.3%.

They are also weak on generators not in training (e.g. Sora, Kling, Pika, Midjourney), and there is no face-swap
training data. Each finding shows its model version, score and threshold. Its weight stays 0 ("uncalibrated (not
counted in label)") until a media type has at least 10 untouched and 10 AI files of our own (our test set plus the
detector server's phone hold-out); the server's own test-split predictions are never used. Files the detectors were
trained on are excluded from calibration by SHA-256. The model card shown on the Calibration page comes from the
detector server only.
