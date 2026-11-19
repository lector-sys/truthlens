"""Shared ffprobe helper for video + audio."""
import json, subprocess


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True)
    return json.loads(out.stdout or "{}")


def encoder_tags(meta):
    tags = {}
    for s in meta.get("streams", []):
        tags.update(s.get("tags", {}))
    tags.update(meta.get("format", {}).get("tags", {}))  # container-level wins (e.g. encoder=Lavf over stream Lavc)
    return {k.lower(): v for k, v in tags.items()}
