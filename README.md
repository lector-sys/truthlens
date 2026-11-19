# TruthLens

Multimodal forensic workbench: image, video and audio evidence plus the caption that came with it go in; located,
explained, cross-checked **indicators** come out, fused into a cautious label (*No / Weak / Moderate / Strong
indicators*) with calibrated weights and a full chain of custody. A person reviews every finding.

## Setup (5 commands, Windows PowerShell)
```
git clone https://github.com/lector-sys/truthlens; cd truthlens
python -m pip install -r requirements.txt           # + ffmpeg/ffprobe on PATH
copy .env.example .env                              # fill OPENAI_API_KEY (optional: without it semantic checks show "not run")
python test_smoke.py                                # must print OK
python demo.py                                      # seeds demo1-3, opens the dashboard (add --offline on stage)
```

| Variable | Default | Used for |
|---|---|---|
| `OPENAI_API_KEY` | none | Luna semantic checks + transcription; without it those checks are recorded as *not run* |
| `TRUTHLENS_MODEL` | `gpt-5.6-luna` | claims, caption↔image, caption↔frames, report summary |
| `TRUTHLENS_TRANSCRIBE_MODEL` | `whisper-1` | transcripts with word timestamps |

## How it works
```
upload ─► vault (SHA-256, read-only copy, append-only log)
        ─► analyzers by type ── image: EXIF · ELA · noise map · copy-move
                              ├ video: metadata · per-frame ELA/noise · temporal spikes (not scene cuts)
                              └ audio: noise-floor splices · clicks · spectrogram · transcript
        ─► text: claims (Luna; regex offline)
        ─► cross-media: A/V flags within 0.5 s · caption↔image/frames · claimed date↔metadata · quote↔transcript
        ─► fusion: noisy-OR of score × calibrated weight (calibration.json)
        ─► dashboard (app.py) + self-contained HTML report
```
Every detector returns the frozen `Indicator` schema (`truthlens/indicator.py`). Luna calls go only through
`truthlens/llm.py` (strict JSON schemas, evidence text fenced as untrusted, disk cache so re-runs make zero calls).

## Calibration
`python calibrate.py` runs `testset/labels.csv` (see [testset/COLLECTING.md](testset/COLLECTING.md)), fits per-detector
weights on 60% and reports held-out hit / false-alarm rates with Wilson 95% intervals into `calibration.json`, which
fusion and the dashboard load. Synthetic `ph_*` placeholder rows (`python testset/make_placeholder.py`) are refused unless
`--allow-placeholder`, and are then stamped SYNTHETIC everywhere.

## Screens
| Overview | Cross-media | Evidence |
|---|---|---|
| ![](docs/screenshots/demo2_Overview_1440x900.png) | ![](docs/screenshots/demo2_Cross-media_1440x900.png) | ![](docs/screenshots/demo1_Evidence_1440x900.png) |

More in `docs/screenshots/`. Demo script: [DEMO.md](DEMO.md) · limits: [docs/LIMITATIONS.md](docs/LIMITATIONS.md) ·
team brief: [TASKS.md](TASKS.md) · rules: [AGENTS.md](AGENTS.md) · state + open items: [NOTES_STATE.md](NOTES_STATE.md)
