"""Shared helpers: ffprobe for video + audio, Pillow-readable copy of any image."""
import json, subprocess
from PIL import Image
from ..vault import sha256


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True)
    return json.loads(out.stdout or "{}")


def has_stream(meta, kind):
    """Does ffprobe `meta` list a `kind` ("video" / "audio") stream? Cover art (attached_pic) is not video."""
    return any(s.get("codec_type") == kind and not s.get("disposition", {}).get("attached_pic")
               for s in meta.get("streams", []))


def encoder_tags(meta):
    tags = {}
    for s in meta.get("streams", []):
        tags.update(s.get("tags", {}))
    tags.update(meta.get("format", {}).get("tags", {}))  # container-level wins (e.g. encoder=Lavf over stream Lavc)
    return {k.lower(): v for k, v in tags.items()}


def readable_image(path, case):
    """`path` if Pillow decodes it, else a PNG converted with ffmpeg into the artifacts folder (`{stem}.png`; logged as
    "converted" in custody; the evidence file and its hash stay as ingested), else None (logged)."""
    try:
        with Image.open(path) as im:
            im.load()
        return path
    except (OSError, ValueError, Image.DecompressionBombError):
        pass
    out = case.artifact_path(f"{path.stem}.png")
    if not out.exists():
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-frames:v", "1", str(out)],
                           capture_output=True, text=True)
        if r.returncode or not out.exists():
            case.log("conversion_failed", file=path.name, tool="ffmpeg", error=r.stderr.strip()[-300:])
            return None
        case.log("converted", file=path.name, to=out.name, tool="ffmpeg", sha256=sha256(out),
                 note="analysis uses this PNG copy; the evidence file and its hash are unchanged")
    return out
