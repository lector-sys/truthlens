# Setup: GPU server

TruthLens runs on the team laptop. Model calls go to the team's GPU server over Tailscale; everything else runs locally.

## GPU server (already set up: don't change it without the owner)
- Ollama (`truthlens-vl`) and the whisper.cpp server, at `OLLAMA_URL` / `WHISPER_URL` in `.env` (see `.env.example`).
- The laptop must be on the tailnet: `tailscale status` should list the server as online.
- Health: the dashboard header pill pings `/api/tags` every 30 s.
- **One request at a time.** Ollama runs `truthlens-vl` one request at a time, so a call made while another job is
  generating waits for it (test: a 0.5 s call took 46.6 s behind a 2,500-token generation). The dashboard shows
  "GPU server busy, waited X s" when the server reports more than 5 s of `queue_s`. If that becomes common, the server
  owner can set `OLLAMA_NUM_PARALLEL=2` (extra VRAM per slot) or keep other jobs off the server while cases run.
- **Network stalls.** A request can also stall before it reaches the server (what caused the one 60 s timeout seen so
  far). Each attempt uses a fresh connection with a 3 s connect timeout and a 20 s time-to-first-byte limit (whisper:
  30 s + audio length), and is retried once. The dashboard then shows "GPU server didn't respond (network), retried".
- **What each call logs** (case log, `llm_call`): client side `connect_s`, `ttfb_s`, `total_s`; server side `queue_s`,
  `load_s`, `prefill_s`, `gen_s`, tokens; the image size sent. Failed attempts: `llm_attempt_failed` with `network`,
  `stage` (connect / first byte / read) and the same client timings.

## Local tools
c2patool (verified C2PA manifests) is installed on the laptop, not the server: see the README, *Setup → c2patool*.
