"""OWNER A. Video forensics. Working: container metadata.
TODO: frame sampling (1-2 fps + scene cuts) -> image.analyze per frame; optical-flow / frame-diff spikes;
face-crop temporal consistency. Emit location={"t": seconds, "frame": n}."""
from ..indicator import Indicator
from .media_meta import ffprobe, encoder_tags


def analyze(path, case):
    meta = ffprobe(path)
    case.log("ffprobe", file=path.name, format=meta.get("format", {}).get("format_name"))
    tags = encoder_tags(meta)
    out = []
    enc = tags.get("encoder", "")
    if enc and "lavf" in enc.lower():
        out.append(Indicator(path.name, "video", "reencoded", 0.25, "low",
                             f"Container written by FFmpeg ({enc}): file was re-encoded/re-muxed after capture.",
                             location={"meta": "encoder"}, alt_explanations=["Platform transcoding", "Benign conversion"]))
    if "creation_time" not in tags:
        out.append(Indicator(path.name, "video", "no_creation_time", 0.15, "low",
                             "No creation_time in container metadata.", alt_explanations=["Stripped by platform"]))
    return out
