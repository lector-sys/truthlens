"""OWNER A. Video forensics: container metadata, per-frame ELA + noise on 1 fps samples, frame-difference spikes
that are not scene cuts. Emit location={"t": seconds, "frame": n}."""
import dataclasses, subprocess
import cv2
import numpy as np
from PIL import Image
from ..indicator import Indicator
from .media_meta import ffprobe, encoder_tags
from . import image


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
    return out + frame_forensics(path, sample_frames(path, case), case) + temporal_spikes(path, case, scene_cuts(path, case))


def sample_frames(path, case, fps=1, max_frames=60):
    """ffmpeg -> {stem}_f0001.jpg... at `fps`; [(t_seconds, frame_path)] sorted. [] if ffmpeg fails."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vf", f"fps={fps}", "-frames:v", str(max_frames),
                        str(case.artifact_path(f"{path.stem}_f%04d.jpg"))], capture_output=True, text=True)
    if r.returncode:
        case.log("frame_sampling_failed", file=path.name, error=r.stderr.strip()[-300:])
    frames = sorted(case.artifact_path("").glob(f"{path.stem}_f[0-9][0-9][0-9][0-9].jpg"))
    return [((int(f.stem[-4:]) - 1) / fps, f) for f in frames]


def frame_forensics(path, frames, case, top=10):
    """ELA + noise inconsistency on each sampled frame, re-tagged to the video at that time; strongest `top`."""
    out = []
    for i, (t, fp) in enumerate(frames, 1):
        img = Image.open(fp)
        for ind in image.ela(fp, img, case) + image.noise_inconsistency(fp, img, case):
            out.append(dataclasses.replace(ind, file=path.name, modality="video",
                                           location={**ind.location, "t": t, "frame": i},
                                           explanation=f"Frame at {t:.1f}s: {ind.explanation}"))
    return sorted(out, key=lambda i: -i.score)[:top]


def scene_cuts(path, case):
    """Scene-cut times (s) from PySceneDetect's content detector; [] if it can't read the file."""
    from scenedetect import detect, ContentDetector
    try:
        return [s.get_seconds() for s, _ in detect(str(path), ContentDetector())]
    except Exception as e:  # unreadable/odd container: spikes are then judged without cut exclusion
        case.log("scene_detect_failed", file=path.name, error=repr(e)[:300])
        return []


def temporal_spikes(path, case, cuts, z=4.0, top=10):
    """Sudden frame-to-frame change (robust z of mean abs difference) more than 0.25 s from any scene cut."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    prev, d = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (160, max(1, round(160 * g.shape[0] / g.shape[1])))).astype(np.float32)
        d.append(np.abs(g - prev).mean() if prev is not None else 0.0)
        prev = g
    if len(d) < 10:
        cap.release()
        return []
    d = np.array(d[1:])  # d[k] = change from frame k to k+1
    med = np.median(d)
    zs = (d - med) / (1.4826 * np.median(np.abs(d - med)) + 1e-6)
    cuts = np.array(cuts or [-1e9])
    n = np.arange(1, len(d) + 1)
    hit = np.flatnonzero((zs > z) & (np.abs(n / fps - cuts[:, None]).min(axis=0) > 0.25))
    out = []
    for k in hit[np.argsort(-zs[hit])][:top]:
        frame, t = int(n[k]), float(n[k] / fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, f = cap.read()
        art = case.artifact_path(f"{path.stem}_spike_{frame}.jpg")
        if ok:
            cv2.imwrite(str(art), f)
        out.append(Indicator(path.name, "video", "temporal_spike", float(min(1.0, zs[k] / 10)), "low",
                             f"Abrupt picture change at {t:.2f}s (frame {frame}, {zs[k]:.1f}σ above typical motion) "
                             f"that is not a detected scene cut.",
                             location={"t": t, "frame": frame}, artifact=str(art) if ok else None,
                             alt_explanations=["Camera flash or lighting change", "Fast motion", "Compression keyframe"]))
    cap.release()
    return out
