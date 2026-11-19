"""OWNER B. Audio forensics. Working: container metadata.
TODO: Whisper transcript (save to case artifacts as <stem>_transcript.json for cross-modal);
spectral-flux / noise-floor splice detection; windowed synthetic-voice classifier; prosody.
Emit location={"start": s, "end": s}."""
from ..indicator import Indicator
from .media_meta import ffprobe, encoder_tags


def analyze(path, case):
    meta = ffprobe(path)
    case.log("ffprobe", file=path.name, format=meta.get("format", {}).get("format_name"))
    enc = encoder_tags(meta).get("encoder", "")
    if enc and "lavf" in enc.lower():
        return [Indicator(path.name, "audio", "reencoded", 0.25, "low",
                          f"Audio written by FFmpeg ({enc}): processed after capture.",
                          location={"meta": "encoder"}, alt_explanations=["Platform transcoding"])]
    return []
