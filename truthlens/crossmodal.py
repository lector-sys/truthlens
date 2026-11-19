"""OWNER C. Cross-modal correlation. Working: time-aligned flag linking, caption<->image (Luna vision).
TODO: claimed date/place<->EXIF/ffprobe, quote<->transcript (llm.transcribe), caption<->video keyframes,
phash image<->frame, stretch: lip-sync (mouth motion vs audio energy)."""
from . import llm
from .indicator import Indicator

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
CAPTION_PROMPT = """You compare a caption with an image for a forensic reviewer.
Return JSON: {"consistent": true|false, "mismatches": [{"claim": "<caption part>", "observed": "<what the image shows>"}], "confidence": 0..1}.
Judge only whether the image content plausibly matches the caption. Do NOT judge whether the image is AI-generated or edited."""


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
        r = llm.ask_json(CAPTION_PROMPT, f"Caption/context: {context_text}", [p])
        if r is None:
            return []  # offline
        case.log("caption_check", file=p.name, model=llm.MODEL, **r)
        for m in r.get("mismatches", []):
            out.append(Indicator(p.name, "cross", "caption_image_mismatch", float(r.get("confidence", 0.5)), "medium",
                                 f"Caption says '{m['claim']}', image shows: {m['observed']}.",
                                 location={"span": m["claim"]}, related=["context"],
                                 alt_explanations=["Caption is vague or figurative", "Model misread the scene"]))
    return out


def analyze(indicators, context_text, case):
    out = link_timestamps(indicators)
    if context_text:
        out += caption_vs_images(context_text, case)
    return out
