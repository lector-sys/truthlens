"""OWNER A. Video forensics. Edited: container metadata, per-frame ELA + noise on 1 fps samples, frame-difference spikes
that are not scene cuts. Generated: face-region flicker, the image checks for generated content on 3 keyframes,
provenance metadata of the video file. Also tracks the face (OpenCV Haar) and mouth motion per frame into
{stem}_faces.json for the cross-modal lip-sync check. Emit location={"t": seconds, "frame": n}."""
import dataclasses, json, subprocess
import cv2
import numpy as np
from PIL import Image
from ..indicator import Indicator
from .media_meta import ffprobe, encoder_tags, has_stream
from . import image


def analyze(path, case):
    meta = ffprobe(path)
    case.log("ffprobe", file=path.name, format=meta.get("format", {}).get("format_name"))
    if not has_stream(meta, "video"):  # second guard (pipeline routes by streams): OpenCV can hang on these
        case.log("check_not_run", check="video analysis", file=path.name, reason="no video stream")
        return []
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
    frames, cuts = sample_frames(path, case), scene_cuts(path, case)
    return (out + frame_forensics(path, frames, case) + temporal_spikes(path, case, cuts)
            + face_region_flicker(path, face_track(path, case), cuts, case) + keyframe_checks(path, frames, case)
            + [dataclasses.replace(i, modality="video") for i in image.synthetic_provenance(path, case)])


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


EDGE = 0.5  # s at each end not judged: recording start/stop and camera handling


def temporal_spikes(path, case, cuts, z=4.0, top=10):
    """Sudden frame-to-frame change (robust z of mean abs difference) more than 0.25 s from any scene cut and more
    than EDGE s from either end of the clip."""
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
    end = (len(d) + 1) / fps
    hit = np.flatnonzero((zs > z) & (np.abs(n / fps - cuts[:, None]).min(axis=0) > 0.25)
                         & (n / fps >= EDGE) & (n / fps <= end - EDGE))
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
                             f"that is not a detected scene cut (0-{EDGE:g} s and {end - EDGE:.1f}-{end:.1f} s "
                             f"not judged: recording start/stop).",
                             location={"t": t, "frame": frame}, artifact=str(art) if ok else None,
                             alt_explanations=["Camera flash or lighting change", "Fast motion", "Compression keyframe"]))
    cap.release()
    return out


FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
GENERATED_ALT = ["Video compression removes fine detail and adds block patterns"]


def _face(gray):
    """Largest frontal face [x, y, w, h] in full-frame pixels (detected on a 320 px wide copy), or None."""
    k = 320 / gray.shape[1]
    small = cv2.resize(gray, (320, max(1, round(gray.shape[0] * k))))
    found = FACE.detectMultiScale(small, 1.1, 4, minSize=(max(24, small.shape[0] // 8),) * 2)
    if not len(found):
        return None
    x, y, w, h = max(found, key=lambda b: b[2] * b[3])
    return [int(round(v / k)) for v in (x, y, w, h)]


def face_track(path, case, rate=25, max_s=60, redetect=3):
    """Per frame (up to `rate` fps, first `max_s` s): largest face box (re-detected every `redetect` samples), mouth
    motion energy (mean abs change of the lower-middle face crop, 32x16 px), mean colour and noise inside the face vs the
    rest of the frame. Saved as {stem}_faces.json; returns {"fps", "face_ratio", "rows"}."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / rate))
    rows, box, prev, k, miss = [], None, None, 0, 0
    while k / fps < max_s:
        ok, f = cap.read()
        if not ok:
            break
        if k % step == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            if len(rows) % redetect == 0:
                found = _face(g)
                miss = 0 if found else miss + 1
                box = found or (box if miss * redetect <= 5 else None)  # keep the last box through a few missed detections
            row = {"t": round(k / fps, 3), "frame": k}
            if box:
                x, y, w, h = box
                mouth = cv2.resize(g[y + int(0.65 * h):y + int(0.95 * h), x + int(0.25 * w):x + int(0.75 * w)], (32, 16))
                mouth = mouth.astype(np.float32)
                lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
                u = int(0.6 * h)  # upper face only (forehead, eyes, cheeks): a talking mouth must not count as flicker
                area = f.shape[0] * f.shape[1] - w * h
                row.update(box=box, mouth=float(np.abs(mouth - prev).mean()) if prev is not None else None,
                           face_rgb=f[y:y + u, x:x + w].reshape(-1, 3).mean(axis=0).round(2).tolist(),
                           bg_rgb=((f.reshape(-1, 3).sum(axis=0) - f[y:y + h, x:x + w].reshape(-1, 3).sum(axis=0))
                                   / max(area, 1)).round(2).tolist(),
                           face_noise=round(float(lap[y:y + u, x:x + w].mean()), 3),
                           bg_noise=round(float((lap.sum() - lap[y:y + h, x:x + w].sum()) / max(area, 1)), 3))
                prev = mouth
            else:
                prev = None
            rows.append(row)
        k += 1
    cap.release()
    track = {"fps": fps / step, "face_ratio": round(sum("box" in r for r in rows) / max(len(rows), 1), 3), "rows": rows}
    case.artifact_path(f"{path.stem}_faces.json").write_text(json.dumps(track), encoding="utf-8")
    case.log("face_track", file=path.name, samples=len(rows), face_ratio=track["face_ratio"])
    return track


def face_region_flicker(path, track, cuts, case, z=6.0, top=10):
    """Frame-to-frame change of colour and noise inside the face box minus the same change in the background:
    a face that flickers while the scene stays steady. Robust z > `z`, > 0.25 s from scene cuts. Needs a face in
    > 50% of frames."""
    if track["face_ratio"] <= 0.5:
        case.log("check_not_run", check="face_region_flicker", file=path.name,
                 reason=f"no face: visible in {track['face_ratio']:.0%} of sampled frames (needs > 50%)")
        return []
    r = [x for x in track["rows"] if "box" in x]
    if len(r) < 10:
        return []
    fc, bc = np.array([x["face_rgb"] for x in r]), np.array([x["bg_rgb"] for x in r])
    fn, bn = np.log(np.array([x["face_noise"] for x in r]) + 1e-3), np.log(np.array([x["bg_noise"] for x in r]) + 1e-3)
    # face change beyond what the whole scene did (lighting, exposure and compression hit both)
    d = (np.abs(np.diff(fc, axis=0)).mean(axis=1) - np.abs(np.diff(bc, axis=0)).mean(axis=1)
         + 10 * (np.abs(np.diff(fn)) - np.abs(np.diff(bn))))
    med = np.median(d)
    # ponytail: spread floored at 1 grey level: a still face (spread ~0) must not turn box jitter into 'flicker'
    zs = (d - med) / max(1.4826 * np.median(np.abs(d - med)), 1.0)
    t = np.array([x["t"] for x in r[1:]])
    far = np.abs(t[:, None] - np.array(cuts or [-1e9])[None, :]).min(axis=1) > 0.25
    hit = np.flatnonzero((zs > z) & far)
    return [Indicator(path.name, "video", "face_region_flicker", float(min(1.0, zs[k] / 15)), "low",
                      f"At {t[k]:.2f}s the colour/noise inside the face box jumps {zs[k]:.0f}σ more than the rest of the "
                      f"frame, not at a scene cut: a face that changes independently of the scene.",
                      location={"t": float(t[k]), "frame": r[k + 1]["frame"], "bbox": r[k + 1]["box"]},
                      alt_explanations=["Lighting change on the face", "Compression of a moving face", "Fast head motion"])
            for k in hit[np.argsort(-zs[hit])][:top]]


def keyframe_checks(path, frames, case):
    """spectral_upsampling + synthetic_visual_artifacts on 3 evenly spaced sampled frames, re-tagged to the video."""
    pick = [frames[i] for i in sorted(set(np.linspace(0, len(frames) - 1, 3).astype(int)))] if frames else []
    out = []
    for t, fp in pick:
        img = Image.open(fp)
        for ind in image.spectral_upsampling(fp, img, case) + image.synthetic_visual_artifacts(fp, fp, img.size, case):
            out.append(dataclasses.replace(ind, file=path.name, modality="video", location={**ind.location, "t": t},
                                           explanation=f"Keyframe at {t:.1f}s: {ind.explanation}",
                                           alt_explanations=ind.alt_explanations + GENERATED_ALT))
    return out
