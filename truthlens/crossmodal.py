"""OWNER C. Cross-modal correlation: time-aligned A/V flags, caption<->image/frames (vision model),
claimed date<->EXIF/ffprobe, quoted words<->transcript, lip motion<->speech audio (lip-sync)."""
import json, re
import numpy as np
from PIL import Image
from rapidfuzz import fuzz
from . import llm
from .indicator import Indicator
from .analyzers import IMAGE_EXTS
from .analyzers.audio import load as load_wav
from .analyzers.media_meta import ffprobe, encoder_tags, readable_image

AV_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".wav", ".mp3", ".m4a", ".flac", ".ogg")
VIDEO_EXTS = AV_EXTS[:5]
CAPTION_PROMPT = """You compare a caption with an image for a forensic reviewer.
Return JSON: {"consistent": true|false, "mismatches": [{"claim": "<caption part>", "observed": "<what the image shows>"}], "confidence": 0..1}.
Judge only whether the image content plausibly matches the caption. Do NOT judge whether the image is AI-generated or edited."""
CAPTION_VERSION = "v2"
CAPTION_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["consistent", "mismatches", "confidence"],
                  "properties": {"consistent": {"type": "boolean"}, "confidence": {"type": "number"},
                                 "mismatches": {"type": "array", "items": {
                                     "type": "object", "additionalProperties": False, "required": ["claim", "observed"],
                                     "properties": {"claim": {"type": "string"}, "observed": {"type": "string"}}}}}}


def _ask_caption(context_text, images, case, note=""):
    return llm.ask_json(CAPTION_PROMPT, llm.untrusted("caption", context_text) + note, images, schema=CAPTION_SCHEMA,
                        name="caption_consistency", version=CAPTION_VERSION, effort="medium", case=case)


def _conf(r):
    return float(min(1.0, max(0.0, r.get("confidence", 0.5))))


def _t(ind):
    loc = ind.location
    return loc.get("t", loc.get("start"))


def link_timestamps(indicators, window=0.5):
    """Video flag and audio flag within `window` seconds -> one linked cross-modal indicator. An audio flag located as a
    span (a splice lies somewhere between two pauses) links when the video time is within `window` of that span."""
    timed = [i for i in indicators if _t(i) is not None]
    vids = [i for i in timed if i.modality == "video"]
    auds = [i for i in timed if i.modality == "audio"]
    out = []
    for v in vids:
        for a in auds:
            lo, hi = a.location.get("start", _t(a)), a.location.get("end", _t(a))
            if lo - window <= _t(v) <= hi + window:
                out.append(Indicator(v.file, "cross", "av_aligned_flags", min(1.0, (v.score + a.score) / 2 + 0.2), "medium",
                                     f"Video '{v.type}' at {_t(v):.2f}s coincides with audio '{a.type}' at {_t(a):.2f}s.",
                                     location={"t": _t(v)}, related=[a.file],
                                     alt_explanations=["Scene cut with natural audio change"]))
    return out


def caption_vs_images(context_text, case):
    out = []
    for p in case.evidence():
        src = readable_image(p, case) if p.suffix.lower() in IMAGE_EXTS else None
        if src is None:
            continue
        r = _ask_caption(context_text, [src], case)
        if r is None:
            return []  # offline
        case.log("caption_check", file=p.name, model=llm.MODEL, **r)
        for m in r.get("mismatches", []):
            out.append(Indicator(p.name, "cross", "caption_image_mismatch", _conf(r), "medium",
                                 f"Caption says '{m['claim']}', image shows: {m['observed']}.",
                                 location={"span": m["claim"]}, related=["context"],
                                 alt_explanations=["Caption is vague or figurative", "Model misread the scene"]))
    return out


def caption_vs_frames(context_text, case):
    """Caption vs 3 evenly spaced sampled frames of each video (A's {stem}_f*.jpg artifacts), one Luna call per video."""
    out = []
    for p in case.evidence():
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        frames = sorted(case.artifact_path("").glob(f"{p.stem}_f[0-9]*.jpg"))
        if not frames:
            continue
        pick = [frames[i] for i in sorted(set(np.linspace(0, len(frames) - 1, 3).astype(int)))]
        at = ", ".join(f"{int(f.stem[-4:]) - 1}s" for f in pick)  # A4: frame i (1-based) at (i - 1) s
        r = _ask_caption(context_text, pick, case, f"\nThe images are frames from one video, at {at}.")
        if r is None:
            return []  # offline
        case.log("caption_frames_check", file=p.name, model=llm.MODEL, frames=[f.name for f in pick], **r)
        for m in r.get("mismatches", []):
            out.append(Indicator(p.name, "cross", "caption_frame_mismatch", _conf(r), "medium",
                                 f"Caption says '{m['claim']}', video frames show: {m['observed']}.",
                                 location={"span": m["claim"]}, related=["context"],
                                 alt_explanations=["Caption is vague or figurative", "Model misread the scene",
                                                   "Sampled frames miss the described moment"]))
    return out


def _claims(case):
    """All claims logged by text.analyze in this case (caption + every text evidence file), lists merged, {} if none."""
    out = {}
    for e in case.entries():
        if e["action"] == "text_claims":
            for k, v in e.items():
                if isinstance(v, list):
                    out.setdefault(k, []).extend(x for x in v if x not in out[k])
    return out


def _capture_date(p):
    """ISO capture date (YYYY-MM-DD) from EXIF DateTimeOriginal/DateTime or container creation_time, else None."""
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        try:
            ex = Image.open(p).getexif()
        except OSError:
            return None
        d = ex.get_ifd(0x8769).get(36867) or ex.get(306)
        return str(d)[:10].replace(":", "-") if d else None
    if ext in AV_EXTS:
        d = encoder_tags(ffprobe(p)).get("creation_time")
        return d[:10] if d else None
    return None


def claimed_date_vs_metadata(claims, case):
    """Claimed date (at its own precision: year / month / day) vs each file's recorded capture date."""
    when = [str(w).strip() for w in claims.get("when", [])]
    when = [w for w in when if re.fullmatch(r"\d{4}(-\d{2}){0,2}", w)]  # non-ISO (e.g. regex 1/2/24) is skipped
    out = []
    for p in case.evidence():
        got = _capture_date(p) if when else None
        if not got:
            continue  # A already flags missing metadata
        for w in when:
            if got[:len(w)] != w:
                out.append(Indicator(p.name, "cross", "claim_date_mismatch", 0.6, "medium",
                                     f"Context claims {w}, but {p.name} metadata records capture on {got}.",
                                     location={"span": w}, related=["context"],
                                     alt_explanations=["Camera clock set wrong", "Metadata rewritten on export",
                                                       "Claim refers to publication date, not capture"]))
    return out


def quote_vs_transcript(claims, case, min_ratio=70):
    """Each claimed quote fuzzy-matched against B's transcript of each recording; low match = quote not heard."""
    quotes = [q for q in claims.get("quotes", []) if str(q).strip()]
    out = []
    for p in case.evidence():
        tp = case.artifact_path(f"{p.stem}_transcript.json")
        if not quotes or not tp.exists():
            continue
        said = " ".join(s["text"] for s in json.loads(tp.read_text(encoding="utf-8"))).lower()
        if not said.strip():
            continue
        for q in quotes:
            ratio = fuzz.partial_ratio(str(q).lower(), said)
            if ratio < min_ratio:
                out.append(Indicator(p.name, "cross", "quote_not_in_transcript", (100 - ratio) / 100, "medium",
                                     f"Quoted words '{q}' were not found in the transcript of {p.name} "
                                     f"(best match {ratio:.0f}/100).",
                                     location={"span": q}, related=["context"],
                                     alt_explanations=["Paraphrased quote", "Transcription error",
                                                       "Quote from a different recording"]))
    return out


def lipsync_series(t, mouth, env, max_lag=0.5):
    """Cross-correlate mouth motion with the change of the speech envelope, over frames with a face during speech.
    -> {"lag_ms": best lag (> 0: the audio comes after the lips), "r": correlation there, "r0": at lag 0, "n": frames}."""
    t, m, a = np.asarray(t, float), np.asarray(mouth, float), np.asarray(env, float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.04
    speech = a > np.percentile(a, 40)
    ok = np.isfinite(m) & speech
    if ok.sum() < 3 / dt:  # need >= 3 s of speech with a visible face
        return None
    k = np.ones(3) / 3
    m = np.convolve(np.nan_to_num(m), k, "same")
    a = np.convolve(np.abs(np.diff(a, prepend=a[0])), k, "same")  # motion goes with *changes* in loudness
    m, a = (m - m[ok].mean()) / (m[ok].std() + 1e-9), (a - a[ok].mean()) / (a[ok].std() + 1e-9)
    best = None
    for lag in range(-round(max_lag / dt), round(max_lag / dt) + 1):
        sh = np.roll(a, -lag)  # audio shifted earlier by `lag` frames
        use = ok & np.roll(ok, -lag)
        r = float((m[use] * sh[use]).mean()) if use.sum() > 10 else -1.0
        if lag == 0:
            r0 = r
        if best is None or r > best[1]:
            best = (lag, r)
    return {"lag_ms": round(best[0] * dt * 1000), "r": round(best[1], 3), "r0": round(r0, 3), "n": int(ok.sum())}


def lipsync_mismatch(case, low=0.25, offset_ms=150):
    """Per video with a tracked face (A's {stem}_faces.json) and decoded audio (B's {stem}_16k.wav): flag low lip/speech
    correlation or a consistent offset > `offset_ms`. Needs a face in > 50% of sampled frames, else logged not run."""
    out = []
    for p in case.evidence():
        fj, wav = case.artifact_path(f"{p.stem}_faces.json"), case.artifact_path(f"{p.stem}_16k.wav")
        if p.suffix.lower() not in VIDEO_EXTS or not fj.exists() or not wav.exists():
            continue
        track = json.loads(fj.read_text(encoding="utf-8"))
        if track["face_ratio"] <= 0.5:
            case.log("check_not_run", check="lipsync", file=p.name,
                     reason=f"no face: visible in {track['face_ratio']:.0%} of sampled frames (needs > 50%)")
            continue
        y, sr = load_wav(wav)
        rows = track["rows"]
        t = np.array([r["t"] for r in rows])
        half = max(1, int(0.02 * sr))
        env = np.array([np.sqrt((y[max(0, int(x * sr) - half):int(x * sr) + half] ** 2).mean() + 1e-12) for x in t])
        mouth = [r["mouth"] if r.get("mouth") is not None else np.nan for r in rows]
        res = lipsync_series(t, mouth, env)  # linear RMS: the mouth opens with amplitude, not with dB
        if res is None:
            case.log("check_not_run", check="lipsync", file=p.name, reason="under 3 s of speech with a visible face")
            continue
        case.artifact_path(f"{p.stem}_lipsync.json").write_text(json.dumps({**res, "t": t.tolist(), "mouth": mouth,
                                                                            "audio_db": (20 * np.log10(env)).tolist()},
                                                                           default=lambda v: None), encoding="utf-8")
        case.log("lipsync_check", file=p.name, **res)
        lag, r, r0 = res["lag_ms"], res["r"], res["r0"]
        if abs(lag) > offset_ms and r > 0.3 and r - r0 > 0.1:
            score, why = min(1.0, abs(lag) / 400), (f"Lip motion and speech line up best when the audio is shifted by "
                                                    f"{lag:+d} ms (r {r:.2f} vs {r0:.2f} as recorded)")
        elif r < low:
            score, why = 0.4, f"Lip motion barely follows the speech (best correlation {r:.2f} over {res['n']} frames)"
        else:
            continue
        out.append(Indicator(p.name, "cross", "lipsync_mismatch", score, "low", why + ", as in dubbed or generated video.",
                             location={"start": float(t[0]), "end": float(t[-1]), "lag_ms": lag},
                             alt_explanations=["Dubbing or voice-over", "Encoder A/V drift (constant offset)",
                                               "Face turned away or mouth hidden"]))
    return out


def analyze(indicators, context_text, case):
    out = link_timestamps(indicators) + lipsync_mismatch(case)
    if context_text:
        out += caption_vs_images(context_text, case) + caption_vs_frames(context_text, case)
    claims = _claims(case)
    return out + claimed_date_vs_metadata(claims, case) + quote_vs_transcript(claims, case)
