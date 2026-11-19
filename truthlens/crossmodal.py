"""OWNER C. Cross-modal correlation: time-aligned A/V flags, caption<->image/frames (Luna vision),
claimed date<->EXIF/ffprobe, quoted words<->transcript."""
import json, re
import numpy as np
from PIL import Image
from rapidfuzz import fuzz
from . import llm
from .indicator import Indicator
from .analyzers.media_meta import ffprobe, encoder_tags

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
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
    """Video flag and audio flag within `window` seconds -> one linked cross-modal indicator."""
    timed = [i for i in indicators if _t(i) is not None]
    vids = [i for i in timed if i.modality == "video"]
    auds = [i for i in timed if i.modality == "audio"]
    out = []
    for v in vids:
        for a in auds:
            if abs(_t(v) - _t(a)) <= window:
                out.append(Indicator(v.file, "cross", "av_aligned_flags", min(1.0, (v.score + a.score) / 2 + 0.2), "medium",
                                     f"Video '{v.type}' at {_t(v):.2f}s coincides with audio '{a.type}' at {_t(a):.2f}s.",
                                     location={"t": _t(v)}, related=[a.file],
                                     alt_explanations=["Scene cut with natural audio change"]))
    return out


def caption_vs_images(context_text, case):
    out = []
    for p in case.evidence():
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        r = _ask_caption(context_text, [p], case)
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
    """Latest claims logged by text.analyze, or {}."""
    e = next((e for e in reversed(case.entries()) if e["action"] == "text_claims"), None)
    return {k: v for k, v in e.items() if k not in ("ts", "action")} if e else {}


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


def analyze(indicators, context_text, case):
    out = link_timestamps(indicators)
    if context_text:
        out += caption_vs_images(context_text, case) + caption_vs_frames(context_text, case)
    claims = _claims(case)
    return out + claimed_date_vs_metadata(claims, case) + quote_vs_transcript(claims, case)
