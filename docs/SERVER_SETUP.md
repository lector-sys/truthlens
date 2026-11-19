# Setup: GPU server and local tools

TruthLens runs on the team laptop. Model calls go to the team's GPU server over Tailscale; everything else runs locally.

## GPU server (already set up: don't change it without the owner)
- Ollama (`truthlens-vl`) and the whisper.cpp server, at `OLLAMA_URL` / `WHISPER_URL` in `.env` (see `.env.example`).
- The laptop must be on the tailnet: `tailscale status` should list the server as online.
- Health: the dashboard header pill pings `/api/tags` every 30 s.
- **One request at a time.** Ollama runs `truthlens-vl` one request at a time, so a call made while anyone else is
  generating waits for them (measured: a 0.5 s call took 46.6 s behind a 2,500-token generation). Each `llm_call` in a
  case log records `queue_s` (time spent waiting), `load_s`, `prefill_s`, `gen_s`, tokens and the image size sent. A call
  still times out after 60 s and is retried once (`llm_attempt_failed` in the log). If `queue_s` is often large, the
  server owner can set `OLLAMA_NUM_PARALLEL=2` (needs extra VRAM per parallel slot) or keep other jobs off the server
  while cases run.

## c2patool (optional, for verified C2PA manifests)
Without it, TruthLens detects C2PA manifests by scanning the file's bytes and reports them as
"C2PA manifest present (not verified)". With `c2patool` on PATH it reads and validates the manifest instead
(`image._c2patool`): the claim generator, the declared digital source type, and any validation failures.

Use the official release from the Content Authenticity Initiative (`contentauth/c2pa-rs`, tags `c2patool-v*`;
v0.27.23 at the time of writing):

**Windows (PowerShell)**
```powershell
$v = "c2patool-v0.27.23"   # or the newest c2patool-v* tag on https://github.com/contentauth/c2pa-rs/releases
gh release download $v -R contentauth/c2pa-rs -p "*x86_64-pc-windows-msvc.zip" -D "$env:TEMP\c2patool"
Expand-Archive "$env:TEMP\c2patool\$v-x86_64-pc-windows-msvc.zip" "$env:LOCALAPPDATA\Programs\c2patool" -Force
# add the folder that contains c2patool.exe to your user PATH, then open a new terminal:
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:LOCALAPPDATA\Programs\c2patool", "User")
c2patool --version
```
(No `gh`? Download the same `...-x86_64-pc-windows-msvc.zip` from the releases page in a browser.)

**Linux / macOS**: the `...-x86_64-unknown-linux-gnu.tar.gz` / `...-universal-apple-darwin.zip` asset from the same
release, unpacked onto PATH; or `cargo install c2patool`.

Check it works: `c2patool some_image.jpg` prints the manifest store as JSON (or reports that there is none). Restart
the dashboard afterwards; no TruthLens setting is needed.
