# Model card: trained AI-generation detectors

Served by the GPU host at `DETECTORS_URL` (`/image`, `/audio`, `/video`, `/health`, `/exports/test_predictions/{modality}`).
TruthLens shows them as `learned_synthetic_image` / `_audio` / `_video` findings (Generated group). The full training
card, with dataset licences and URLs, is on the GPU host (`C:\truthlens-gpu\train\MODEL_CARD.md`); this copy holds what
TruthLens relies on, measured from the server's own held-out test export on 2026-11-20.

| Detector | Version | Threshold | Trained on | Held-out: caught | Held-out: false alarms |
|---|---|---|---|---|---|
| image | image-clipL14-20261120 | 0.7368 | GenImage subset: ADM, BigGAN, GLIDE, SD 1.5, VQDM, Wukong vs ImageNet photos | 304/418 = 73% | 16/437 = 3.7% |
| audio | audio-wavlmB+-20261120 | 0.7147 | ASVspoof 2019 LA: attacks A07-A19 vs VCTK studio speech | 192/223 = 86% | 25/364 = 6.9% |
| video | video-clipL14-20261120 | 0.5359 | AnimateDiff-Lightning, CogVideoX-5B, Runway, Stable Diffusion video, VideoPoet vs Pexels stock clips | 76/88 = 86% | 3/68 = 4.4% |

Per generator (caught at threshold): image ADM 45/68, BigGAN 74/85, GLIDE 49/65, SD 1.5 32/59, VQDM 46/69, Wukong 58/72;
audio A07 14/19, A08 19/21, A09 21/21, A11 13/15, A12 12/19, A13 19/19, A14 15/16, A15 16/17, A16 9/14, A17 25/25,
A18 17/22, A19 12/15; video AnimateDiff-Lightning 11/13, CogVideoX-5B 15/20, Runway 16/19, SD video 14/16, VideoPoet 20/20.

## On our own files (out of distribution)

| File | Detector | Score | Threshold | Result |
|---|---|---|---|---|
| 1.m4a.mp4 (iPhone voice memo) | audio | 0.90 | 0.71 | **false alarm** |
| 5.m4a.mp4 (iPhone voice memo) | audio | 0.62 | 0.71 | below threshold |
| WhatsApp video (phone camera) | video | 0.69 | 0.54 | **false alarm** |
| WhatsApp video, soundtrack | audio | 1.00 | 0.71 | would be a false alarm (soundtracks are not sent to the audio detector) |

2 of 3 of our untouched phone/WhatsApp files are flagged. The held-out numbers above come from the training
distribution and do not carry over to phone or messenger media.

## Limitations

- Weak on generators not in training (e.g. Sora, Kling, Pika, Midjourney); no face-swap training data.
- Audio was trained on studio speech: phone voice-memo processing and messenger compression look out of distribution.
- Video was trained on stock footage vs text-to-video clips: handheld phone video and WhatsApp recompression look out
  of distribution.
- Per-window / per-segment scores reuse the whole-file threshold and are indicative only.
- Weights in the label come from `calibrate.py` (public_test rows plus our own files; see the Calibration page).
