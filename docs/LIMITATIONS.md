# What TruthLens cannot do

TruthLens points a human at places worth checking. It does not decide whether media was manipulated.

- **Fully generated media with no local edits.** Our detectors look for *inconsistencies inside* a file (a pasted region,
  a joined recording, a jump in the picture). An image or voice generated in one piece can be consistent throughout and
  show no indicators. We deliberately use no "AI-generated" classifier: none we could run on a CPU would be trustworthy.
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
