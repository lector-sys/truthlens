import dataclasses, json, threading
from . import analyzers, crossmodal, fusion, llm, report
from .analyzers import text as text_analyzer, image, IMAGE_EXTS, TEXT_EXTS
from .analyzers.media_meta import ffprobe, has_stream
from .indicator import Indicator

WATCHDOG_S, WATCHDOG_PER_S = 60.0, 2.0  # an analyzer gets 60 s + 2 s per second of media, then counts as failed


def route(path, case):
    """(analyzers, media seconds) for one evidence file, logged as "routed" (coverage reads it back). Audio/video go by
    the streams ffprobe finds, not the extension: an audio-only .mp4 takes the audio path, a silent video skips audio."""
    mods, dur = analyzers.BY_EXT.get(path.suffix.lower(), ()), 0.0
    if analyzers.video in mods or analyzers.audio in mods:
        meta = ffprobe(path)
        dur = float(meta.get("format", {}).get("duration") or 0)
        mods = tuple(m for m, kind in ((analyzers.video, "video"), (analyzers.audio, "audio")) if has_stream(meta, kind))
    case.log("routed", file=path.name, analyzers=[m.__name__ for m in mods], seconds=dur)
    return mods, dur


def guarded(fn, limit, *args):
    """fn(*args) with a time limit: TimeoutError after `limit` s, so a hung decoder can't stall the case."""
    # ponytail: a timed-out worker thread can't be killed and runs on until it returns; run analyzers in a child
    # process if a real hang keeps burning CPU (routing by streams removed the one we've seen).
    box = {}

    def work():
        try:
            box["ok"] = fn(*args)
        except BaseException as e:  # handed back to the caller below
            box["err"] = e
    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(limit)
    if t.is_alive():
        raise TimeoutError(f"timed out after {round(limit, 1):g} s")
    if "err" in box:
        raise box["err"]
    return box["ok"]


def mark_failed(case, why):
    """Close a case that could not finish: logged, and a result.json saying so, so it never looks like a pending run."""
    case.log("analysis_failed", reason=why)
    (case.dir / "result.json").write_text(json.dumps({"indicators": [], "context": "", "failed": why}), encoding="utf-8")


def run(case, context_text="", progress=lambda step, state: None):
    """Analyze every evidence file, then context text, cross-modal checks and fusion. `progress(step, state)` is called
    with state "running" / "done" / "error" for each step so a UI can show live status. Result saved to result.json.
    Anything that escapes leaves the case marked failed (never half-finished)."""
    try:
        return _run(case, context_text, progress)
    except Exception as e:
        mark_failed(case, f"analysis did not complete ({e!r})"[:300])
        raise


def _run(case, context_text, progress):
    indicators = []
    for path in case.evidence():
        mods, dur = route(path, case)
        if not mods:
            case.log("skipped", file=path.name, reason="no audio or video stream" if dur else "unsupported type")
            continue
        for mod in mods:
            step = f"{mod.__name__.rsplit('.', 1)[1]} · {path.name}"
            progress(step, "running")
            try:
                found = guarded(mod.analyze, WATCHDOG_S + WATCHDOG_PER_S * dur, path, case)
            except Exception as e:  # one broken or hung analyzer must not kill the case
                case.log("analyzer_error", file=path.name, analyzer=mod.__name__,
                         error=str(e) if isinstance(e, TimeoutError) else repr(e))
                progress(step, "error")
                continue
            case.log("analyzed", file=path.name, analyzer=mod.__name__, n=len(found))
            progress(step, "done")
            indicators += found
        if analyzers.audio in mods and analyzers.video not in mods:  # Content Credentials in audio: display only
            try:
                indicators += [dataclasses.replace(i, modality="audio") for i in image.synthetic_provenance(path, case)]
            except Exception as e:
                case.log("analyzer_error", file=path.name, analyzer="provenance", error=repr(e))
    if context_text:
        progress("text · context", "running")
        indicators += text_analyzer.analyze(context_text, case)
        progress("text · context", "done")
    progress("cross-media checks", "running")
    indicators += crossmodal.analyze(indicators, all_text(case, context_text), case)
    progress("cross-media checks", "done")
    result = {"indicators": indicators, "assessment": fusion.assess(indicators), "integrity": case.verify(),
              "context": context_text}
    case.log("assessed", **result["assessment"]["case"])
    report.summary(result, case)  # once per run, saved below: viewing the case must not call (or log) the model again
    result["source"] = llm.source(case.entries())  # "live" or "result from cache (date)"
    result["coverage"] = coverage(case, context_text)
    result["headline"] = fusion.headline(result["assessment"], result["coverage"])
    case.log("coverage", ran=len(result["coverage"]["ran"]), total=result["coverage"]["total"],
             not_run=result["coverage"]["not_run"], headline=result["headline"])
    (case.dir / "result.json").write_text(json.dumps({**result, "indicators": [i.to_dict() for i in indicators]}),
                                          encoding="utf-8")
    return result


def load(case):
    """The saved result of a previous run (None if never analyzed). Fusion and integrity are recomputed now, so new
    calibration weights and any tampering since the run show up."""
    p = case.dir / "result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    inds = [Indicator(**d) for d in r["indicators"]]
    a, cov = fusion.assess(inds), coverage(case, r.get("context", ""))
    return {**r, "indicators": inds, "assessment": a, "integrity": case.verify(), "coverage": cov,
            "headline": "Failed: " + r["failed"] if r.get("failed") else fusion.headline(a, cov)}


def all_text(case, context_text=""):
    """The caption plus every text evidence file, each labelled: what the cross-media checks compare against the media."""
    parts = [context_text.strip()] + [f"[{p.name}]\n" + p.read_text(encoding="utf-8", errors="replace").strip()
                                      for p in case.evidence() if p.suffix.lower() in TEXT_EXTS]
    return "\n\n".join(x for x in parts if x)


SEMANTIC = {"claims": "claims from the caption", "caption_consistency": "caption ↔ image / frames",
            "synthetic_visual_artifacts": "visual artifacts (model)", "transcribe": "transcript"}


def coverage(case, context_text=""):
    """Which checks ran, from the case log: {"ran": [...], "not_run": {check: why}, "total", "analysed",
    "files": {file: {"analysed": n, "why"}}}. A check counts as run only if it produced a result (even "nothing found")."""
    log, ev = case.entries(), case.evidence()
    context_text = all_text(case, context_text)  # text evidence files count as claims to check, like a caption
    routed = {e["file"]: e["analyzers"] for e in log if e["action"] == "routed"}  # older cases: by extension

    def mods_of(p):
        return routed[p.name] if p.name in routed else [m.__name__ for m in analyzers.BY_EXT.get(p.suffix.lower(), ())]
    kinds = {m.rsplit(".", 1)[1] for p in ev for m in mods_of(p)}
    calls = {e["check"] for e in log if e["action"] == "llm_call"}
    skipped = {}
    for e in log:
        if e["action"] == "check_not_run":
            skipped.setdefault(e["check"], e["reason"])
    checks, files = {}, {}
    for p in ev:
        done = {e["analyzer"] for e in log if e["action"] == "analyzed" and e["file"] == p.name}
        skip = next((e["reason"] for e in log if e["action"] == "skipped" and e["file"] == p.name), None)
        mods = mods_of(p) or sorted(done)
        if not mods or (skip and not done):
            why = f"{skip or 'unsupported file type'} ({p.suffix or 'no extension'})"
            checks[f"analysis · {p.name}"] = why
            files[p.name] = {"analysed": 0, "why": why}
            continue
        for m in mods:
            err = next((e["error"] for e in log if e["action"] == "analyzer_error" and e["file"] == p.name
                        and e["analyzer"] == m), None)
            checks[f"{m.rsplit('.', 1)[1]} analysis · {p.name}"] = None if m in done else f"failed: {err[:80]}" if err else "not run"
        files[p.name] = {"analysed": len(done)} if done else {"analysed": 0, "why": "every analyzer failed"}
    want = (["claims"] + ["caption_consistency"] * bool(kinds & {"image", "video"})
            + ["synthetic_visual_artifacts"] * ("image" in kinds) + ["transcribe"] * any(
                analyzers.audio.__name__ in mods_of(p) and not any(e["action"] == "no_audio" and e["file"] == p.name
                                                                 for e in log) for p in ev))
    for c in want:
        needs_text = c in ("claims", "caption_consistency")
        checks[SEMANTIC[c]] = ("no caption or claims text supplied" if needs_text and not context_text else
                               None if c in calls else skipped.get(c, "not run"))
    checks["claimed date ↔ metadata"] = None if context_text else "no caption or claims text supplied"
    for c, why in skipped.items():  # detector-level skips (e.g. EXIF lost in a format conversion)
        if c not in SEMANTIC:
            checks[c] = why
    not_run = {c: w for c, w in checks.items() if w}
    return {"ran": [c for c, w in checks.items() if not w], "not_run": not_run, "total": len(checks),
            "analysed": sum(f["analysed"] for f in files.values()), "why": "; ".join(f"{k}: {v['why']}" for k, v in files.items() if not v["analysed"]),
            "files": files}


def coverage_line(cov):
    """"Checks run: X of Y. Not run: a (why); b (why)." or None when everything ran."""
    if not cov["not_run"]:
        return None
    return (f"Checks run: {len(cov['ran'])} of {cov['total']}. Not run: "
            + "; ".join(f"{c} ({w})" for c, w in cov["not_run"].items()) + ".")
