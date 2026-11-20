# Setup and configuration

## Install (Windows PowerShell)
```
git clone https://github.com/lector-sys/truthlens; cd truthlens
python -m pip install -r requirements.txt           # plus ffmpeg / ffprobe on PATH
copy .env.example .env                              # fill in the server URLs (see SERVER_SETUP.md)
python test_smoke.py                                # must print OK
python demo.py                                      # seeds demo1-3 and opens the dashboard (--offline: cache only)
```

## Settings (`.env`, git-ignored)

| Variable | Default | Used for |
|---|---|---|
| `LLM_PROVIDER` / `LLM_PROVIDERS` | `ollama` | providers tried in order; `groq` / `openai` (cloud) only if listed here |
| `OLLAMA_URL`, `TRUTHLENS_TEXT_MODEL`, `TRUTHLENS_VISION_MODEL` | none, `truthlens-vl` | claims, caption vs image/frames, visual artifacts, summary |
| `WHISPER_URL` | none | transcripts with word timestamps (whisper.cpp server) |
| `DETECTORS_URL` | none | trained AI-generation detectors (image, video, audio) |
| `C2PATOOL` | `c2patool` on PATH | full path to c2patool if it is not on PATH |
| `GROQ_API_KEY`, `GROQ_MODEL`, `OPENAI_API_KEY`, `TRUTHLENS_MODEL` | none | optional cloud fallbacks (evidence leaves the laptop) |
| `TRUTHLENS_PUBLIC`, `TRUTHLENS_ACCESS_CODE` | off | public demo mode (below) |

Without a reachable server, model checks come from `cache/` or are recorded as *not run*, and the label says so.
All service URLs can point at `127.0.0.1` when the app runs on the GPU PC itself.

## c2patool (verified C2PA manifests)
Without it, C2PA manifests are found by a byte scan and reported as "C2PA manifest present (not verified)". With it,
`synthetic_provenance` validates the manifest and names the signer. Install the official release
(`contentauth/c2pa-rs`, v0.27.23) on Windows:
```powershell
$v = "c2patool-v0.27.23"
gh release download $v -R contentauth/c2pa-rs -p "$v-x86_64-pc-windows-msvc.zip" -D "$env:TEMP\c2patool"
# check: Get-FileHash ...zip must equal b3fba7bdfb6f9c0b43d47e50d1db66785404f3e4b687e7127991157852b29cb5
Expand-Archive "$env:TEMP\c2patool\$v-x86_64-pc-windows-msvc.zip" "$env:LOCALAPPDATA\Programs\c2patool" -Force
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:LOCALAPPDATA\Programs\c2patool\c2patool", "User")
c2patool --version          # in a new terminal; restart the dashboard afterwards
```
Programs started before the PATH change don't see it: set `C2PATOOL=<full path to c2patool.exe>` in `.env` instead.
Linux / macOS: use the `x86_64-unknown-linux-gnu.tar.gz` / `universal-apple-darwin.zip` asset, or `cargo install c2patool`.

## Calibration
`python calibrate.py` runs every row of `testset/labels.csv` (see [testset/COLLECTING.md](../testset/COLLECTING.md)),
fits per-check weights on 60% and reports held-out hit and false-alarm rates with Wilson 95% intervals into
`calibration.json`, which fusion and the dashboard load. Synthetic `ph_*` placeholder rows are refused unless
`--allow-placeholder` is given, and are then stamped SYNTHETIC everywhere. Trained detectors weigh 0 until a media type
has 10 untouched and 10 confirmed AI files of our own; files the detector server trained on are excluded by SHA-256,
and our own test files are listed in `testset/exclude_from_training.sha256` so the server never trains on them.

## Public mode
Off by default; without `TRUTHLENS_PUBLIC=1` the app is the local workspace.

- `TRUTHLENS_ACCESS_CODE`: visitors must enter it first (constant-time compare, 5 tries per IP per 10 minutes).
- Each visitor sees only the three demo cases (read-only) and the cases their own browser session created, stored under
  `cases_public/<session>/` and deleted 24 hours after creation.
- Uploads: at most 3 files of 50 MB per case, video and audio up to 120 s, 5 cases per session per hour, 30 per IP per
  day, refused below 10 GB of free disk. Analyses run one at a time, with a queue position.
- Errors show only a reference id; host names, addresses and paths are removed from the page and the report.
- Start with `python demo.py --no-browser` (it passes `--client.showErrorDetails none` in public mode).
