# TruthLens: Demo Script

**Length:** 5 min demo + Q&A · **Driver (laptop):** D · **Speaker (slides):** C · **Q&A on their parts:** A (image/video), B (audio)
Lines in quotes are what to say. *Italics* are actions. Rehearse at least 3 times with a timer before hour 23.

---

## Before you go on stage (T-30 min)

- [ ] Laptop on charger, Do Not Disturb on, browser zoom 125%, close all other tabs.
- [ ] `$env:OPENAI_API_KEY` and `$env:TRUTHLENS_MODEL` set in the terminal that runs Streamlit.
- [ ] `python test_smoke.py` → `OK`, then `python demo.py --fresh` once **online** (fills `cache/` with every Luna answer).
- [ ] On stage run `python demo.py --offline`: zero API calls, identical output, opens on demo2's Cross-media screen.
- [ ] Demo files ready in a folder on the desktop: `demo1_photo.jpg`, `demo2_clip.mp4`, `demo3_photo.jpg`, plus the 3 context texts in a notepad.
- [ ] Custody → *Export report (HTML)* for demo1-3 and open them in browser tabs: the fallback if the laptop app fails.
- [ ] Backup screen-recording of the full demo on the desktop and on a phone.
- [ ] Slide with the `calibrate.py` table (from the last calibration run) and the flowchart.

## The 3-minute click path (dashboard)

Judging point in **bold**; the click in *italics*.

| Time | Screen | Say / show |
|---|---|---|
| 0:00 | *Sidebar → demo1 → Overview* | **Multi-media + calibrated**: "One label on a four-step scale, never fake or real. The bars say which findings drive it, score × measured weight." |
| 0:30 | *Evidence*, toggle *Noise map*, drag *Opacity* | **Explainable**: "The pasted region has different sensor noise; here's the box, the heatmap, and the innocent explanations listed with it." |
| 1:00 | *Sidebar → demo2 → Cross-media* | **Cross-media**: "Video lane, audio lane: the picture jumps at 4.8 s, the background noise jumps 20 dB at 5.0 s. Two independent detectors, same moment, one linked finding." |
| 1:30 | *Evidence*, *Play from a flagged span* | Play 0.5 s before the splice; point at the shaded spectrogram band. |
| 2:00 | *Calibration* | **Honest confidence**: "Each detector's precision with its 95% interval, fitted on one split, scored on held-out files. The badge says which dataset these came from." |
| 2:20 | *Sidebar → demo3 → Overview* | "Our untouched control: no significant indicators. We don't flag everything." |
| 2:40 | *Custody → Verify integrity → Export report* | **Evidence handling**: "SHA-256 at intake, read-only copy, every step logged, including each model call and which checks didn't run." |

**If something fails:** the app crashes → switch to the exported report tab. A semantic check shows *not run* →
"That check needs the language model; everything else ran offline, and the report says exactly what was skipped."

**If the network dies mid-demo:** keep going. The forensic detectors run offline. Say: *"Our forensics run fully offline; only the language checks need the API. Here's the same case from our pre-run report."* Switch to the report tab.

---

## 0:00–0:30 · Hook (C)

*Slide: one manipulated image next to the original.*

> "One of these photos was edited in ten minutes with free tools. Investigators get files like this every day, plus
> AI voices and cut videos. A detector that says '87% fake' doesn't help them. They need to know **what** looks wrong,
> **where**, **why**, and **how sure** we are. That's TruthLens."

## 0:30–1:00 · How it works (C)

*Slide: flowchart.*

> "You upload a case: images, video, audio and the caption that came with them. We hash every file first, so the evidence
> can be shown to be unchanged. Each media type goes through explainable forensic checks on a normal CPU. Then
> we **cross-check media against each other and against the caption**. Finally we combine everything into a cautious label,
> never 'fake' or 'real', just the strength of the indicators."

---

## 1:00–2:30 · Demo 1: spliced photo + false caption (D drives, C narrates)

*Upload `demo1_photo.jpg`. Paste context: e.g. "Flood waters reach the campus library on 2026-11-02". Click Analyze.*

1. *Point at the top:* overall **Strong indicators**, integrity green.
   > "First thing it tells us: the evidence is untouched since upload. That's the SHA-256 check."
2. *Open the image row:* original vs ELA heatmap vs copy-move.
   > "Error level analysis: this region re-compresses differently from the rest of the photo. And copy-move found the same
   > patch twice: these lines connect the cloned areas."
3. *Read one indicator's "Could also be" line aloud.*
   > "Every finding lists innocent explanations. We don't hide uncertainty, we show it."
4. *Scroll to cross-modal table:* caption mismatch + date mismatch.
   > "Now the cross-check. The caption says a flood; the model describes what's actually in the image, and they disagree.
   > The caption says November 2nd, but the camera metadata says a different date."
5. *Scroll to Recommended human checks.*
   > "And it tells the investigator exactly what to look at next."

## 2:30–3:45 · Demo 2: video with a cloned-voice segment (the cross-modal moment)

*Upload `demo2_clip.mp4`. Paste context with the quote. Analyze (takes ~20–40 s, so talk through it).*

> "This is one of our team members. We replaced four seconds of his voice with an AI clone of **his own** voice, with his consent."

1. *Video row:* flagged-moment thumbnail at ~X s; spectrogram with the red segment at the same time.
   > "The audio analysis finds a jump in background noise at 12.3 seconds: two recordings were joined. The video analysis
   > independently flags a frame jump at 12.4 seconds."
2. *Cross-modal table:* `av_aligned_flags`.
   > "Neither signal is convincing alone. **Together**, at the same moment in two different media, they are. That's
   > cross-modal correlation."
3. *Cross-modal table:* `quote_not_in_transcript`.
   > "And the quote in the caption never appears in what is actually said."

## 3:45–4:15 · Demo 3: the honest control

*Upload `demo3_photo.jpg` with its true caption. Analyze.*

> "Now an untouched phone photo with a true caption." *(Shows No / Weak indicators.)* "No significant indicators. A
> tool that flags everything is useless. And note it still says absence of indicators isn't proof."

## 4:15–4:45 · How sure are we? (C)

*Slide: calibration table.*

> "We built a labelled test set of N files. This table shows, for each detector, how often it fires on manipulated vs
> untouched media. That's why ELA is marked low reliability and copy-move medium. The weights in our scoring come
> from these numbers, not guesses."

## 4:45–5:00 · Close (C)

*Click "Download report", open it.*

> "Everything ends in a self-contained report: hashes, every flag with its location and evidence image, limitations, and a
> full log. TruthLens doesn't replace the investigator. It shows them where to look, and how much to trust it."

---

## Q&A prep

| Likely question | Answer (who) |
|---|---|
| How accurate is it? | Show the calibration table. "Measured on our own N-file set; small, so we label reliability honestly instead of claiming a percentage." (D) |
| Why not a deep-learning deepfake detector? | "They don't generalise to new generators and can't explain themselves. Our design accepts one as just another indicator with its own measured reliability; we chose explainable signals for the prototype." (A) |
| Can it be fooled? | "Yes. Re-compression and careful editing can hide traces; that's in our limitations. That's exactly why it's indicators plus human review, not a verdict." (A/B) |
| What does the LLM do? Can it hallucinate? | "Only language: extracting claims and comparing captions with images. It is never asked whether media is fake. The summary is written only from our indicators." (C) |
| How do you handle evidence? | "SHA-256 at upload, read-only copy, append-only log, re-hashed when the report is generated." (D) |
| What about social-media compressed files? | "ELA weakens, so it's low reliability and lists 'platform re-encoding' as an alternative; metadata and cross-modal checks still work." (A) |
| Audio: how do you detect clones? | "We don't claim to detect cloning directly. We detect the splice: noise-floor jumps and clicks where audio was joined. Plus the transcript-vs-quote check." (B) |
| Privacy? | "Forensics run locally. Only captions, images and audio for transcription go to the API, and the tool runs offline without it." (C) |
| Scalability? | "Each file and modality is independent, so it parallelises; video frames are sampled and capped." (D) |
| What next? | "C2PA content-credential verification, lip-sync checking, and calibration on a large public forensics dataset." (C) |

**Rules for Q&A:** answer in two sentences, then stop. If you don't know: *"We didn't measure that; here's how we would."* Never claim accuracy we didn't measure.
