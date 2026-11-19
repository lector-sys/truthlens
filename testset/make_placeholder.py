"""python testset/make_placeholder.py: SYNTHETIC stand-in test set so calibrate.py runs before real media arrives.
Writes ph_* files to testset/media/ and their rows to labels.csv (rows without the ph_ prefix are kept).
Numbers from this set only prove the plumbing works; they are not detector accuracy. Delete ph_ rows once real media lands."""
import csv, io, subprocess, wave
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MEDIA = ROOT / "media"
SR = 16000
rng = np.random.default_rng(42)


def scene(sigma=4.0, w=640, h=480):
    """Smooth gradient + soft blobs + sensor-like noise."""
    yy, xx = np.mgrid[0:h, 0:w] / max(w, h)
    base = 60 + 120 * xx + 40 * yy
    for _ in range(6):
        cx, cy, r, a = rng.random(), rng.random() * 0.75, 0.05 + 0.15 * rng.random(), rng.uniform(-60, 60)
        base = base + a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2)
    rgb = np.stack([base * k for k in (1.0, 0.9, 0.8)], axis=2)
    for _ in range(int(w * h / 800)):  # textured "objects" so keypoint detectors have corners, as in photos
        x, y, s = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(4, 24))
        rgb[y:y + s, x:x + int(rng.integers(4, 24))] += rng.uniform(-50, 50, 3)
    rgb = rgb + rng.normal(0, sigma, (h, w, 3))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def exif(date, software=None):
    e = Image.Exif()
    e[271], e[272], e[306] = "Placeholder", "PH-1", date.replace("-", ":") + " 10:00:00"
    e.get_ifd(0x8769)[36867] = e[306]
    if software:
        e[305] = software
    return e


def jpeg(arr, q=90, ex=None):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=q, **({"exif": ex} if ex else {}))
    return buf.getvalue()


def splice(date, software):
    """Once-compressed scene with an uncompressed, noisier patch pasted in, then saved again."""
    bg = np.asarray(Image.open(io.BytesIO(jpeg(scene(), q=70)))).copy()
    s = int(rng.integers(80, 160))
    y, x = int(rng.integers(0, 480 - s)), int(rng.integers(0, 640 - s))
    bg[y:y + s, x:x + s] = scene(sigma=14, w=s, h=s)
    return jpeg(bg, q=95, ex=exif(date, software))


def copy_move(date):
    a = scene()
    s = 96
    a[300:300 + s, 400:400 + s] = a[60:60 + s, 80:80 + s]
    return jpeg(a, q=95, ex=exif(date))


XMP = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
       b'<rdf:Description xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" Iptc4xmpExt:DigitalSourceType='
       b'"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/></rdf:RDF></x:xmpmeta>')


def jpeg_xmp(arr):
    """JPEG carrying an XMP packet that declares DigitalSourceType = trainedAlgorithmicMedia (as generators write it)."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=92, xmp=XMP)
    return buf.getvalue()


def png_params(arr, text="Steps: 30, Sampler: DPM++ 2M, Model: stable diffusion xl"):
    """PNG, optionally with an A1111-style 'parameters' text chunk naming the generator."""
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    if text:
        info.add_text("parameters", text)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def upsampled():
    """Scene rendered at half size and bicubic-upsampled 2x: no metadata, only the spectral trace."""
    return np.asarray(Image.fromarray(scene(w=320, h=240)).resize((640, 480), Image.BICUBIC))


def speech(sec=8.0, floor_db=-50.0, f0=150.0):
    """Syllable-like harmonic bursts with pauses over a constant noise floor."""
    t = np.arange(int(sec * SR)) / SR
    env, pos = np.zeros_like(t), 0.1
    while pos < sec - 0.3:
        on = rng.uniform(0.12, 0.25)
        i0, n = int(pos * SR), int(on * SR)
        env[i0:i0 + n] = np.hanning(n) * rng.uniform(0.3, 0.6)
        pos += on + rng.uniform(0.05, 0.15)
    pitch = f0 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 5))
    return env * voice + rng.normal(0, 10 ** (floor_db / 20), len(t))


def wav_bytes(y):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def spliced_speech(cut=None):
    """Two 'rooms' joined at `cut` s: different noise floor and voice."""
    cut = rng.uniform(3, 5) if cut is None else cut
    a, b = speech(cut, -50, 150), speech(8 - cut, -30, 175)
    return np.concatenate([a, b])


def clicked_speech():
    y = speech()
    quiet = np.flatnonzero(np.abs(y) < 0.005)
    k = quiet[len(quiet) // 2]
    y[k:k + 3] += 0.5  # hard-cut click inside a pause
    return y


def video(path, y, date=None, cut=None):
    """Test-pattern video with audio `y`. With `cut`, 3 s of picture are skipped at `cut` s and the rest is a
    slightly brighter "second take" (a re-joined edit)."""
    a = path.with_suffix(".tmp.wav")
    a.write_bytes(wav_bytes(y))
    d = len(y) / SR
    src = f"testsrc2=size=320x240:rate=10:duration={d + 3}"
    vf = (f"{src},split[x][y];[x]trim=0:{cut},setpts=PTS-STARTPTS[p];[y]trim={cut + 3}:{d + 3},setpts=PTS-STARTPTS,eq=brightness=0.08[q];"
          f"[p][q]concat=n=2:v=1[v]") if cut else f"{src}[v]"
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(a), "-filter_complex", vf, "-map", "[v]", "-map", "0:a",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", str(d)]
    cmd += ["-metadata", f"creation_time={date}T10:00:00Z"] if date else []
    subprocess.run(cmd + [str(path)], check=True)
    a.unlink()


def talking_cartoon(path, offset=0.0, sec=8.0, fps=25, seed=0):
    """Drawn face (OpenCV's Haar detector finds it) whose mouth opens with syllable-like audio bursts; the soundtrack
    is shifted by `offset` s (> 0: the audio comes late), as in a dubbed or badly generated talking-head clip."""
    import cv2
    r = np.random.default_rng(seed)
    env, pos = np.zeros(int(sec * SR)), 0.3
    while pos < sec - 0.5:  # syllables 0.15-0.35 s with 0.1-0.3 s gaps
        on = r.uniform(0.15, 0.35)
        i0, n = int(pos * SR), int(on * SR)
        env[i0:i0 + n] = np.hanning(n) * r.uniform(0.4, 0.8)
        pos += on + r.uniform(0.1, 0.3)
    t = np.arange(len(env)) / SR
    phase = 2 * np.pi * np.cumsum(140 * (1 + 0.15 * np.sin(2 * np.pi * 0.8 * t))) / SR
    voice = env * sum(np.sin(k * phase) / k for k in range(1, 5)) + r.normal(0, 0.003, len(t))
    shift = int(offset * SR)
    voice = np.concatenate([r.normal(0, 0.003, shift), voice[:len(voice) - shift]]) if shift > 0 else voice  # room tone, not zeros
    a = path.with_suffix(".tmp.wav")
    a.write_bytes(wav_bytes(voice))
    frames = []
    for k in range(int(sec * fps)):
        im = np.full((240, 320), 200, np.uint8)
        cv2.ellipse(im, (160, 120), (60, 80), 0, 0, 360, 170, -1)
        for x in (135, 185):
            cv2.ellipse(im, (x, 100), (12, 7), 0, 0, 360, 40, -1)
            cv2.line(im, (x - 14, 86), (x + 14, 86), 60, 3)
        cv2.line(im, (160, 105), (155, 135), 120, 3)
        opening = env[min(len(env) - 1, int(k / fps * SR))]
        cv2.ellipse(im, (160, 160), (22, 3 + int(14 * opening)), 0, 0, 360, 60, -1)
        frames.append(np.clip(cv2.GaussianBlur(im, (5, 5), 0) + r.normal(0, 2, im.shape), 0, 255).astype(np.uint8))  # sensor noise
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "320x240", "-r", str(fps),
                    "-i", "-", "-i", str(a), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
                   input=np.stack(frames).tobytes(), check=True)
    a.unlink()


def cloned_voice(path, seed=0):
    """Speech with digitally silent gaps, rendered at 16 kHz and upsampled to 44.1 kHz: a hard 8 kHz band limit."""
    r = np.random.default_rng(seed)
    y = speech() * r.uniform(0.8, 1.0)
    y[np.convolve(np.abs(y), np.ones(160) / 160, "same") < 0.01] = 0.0
    tmp = path.with_suffix(".tmp.wav")
    tmp.write_bytes(wav_bytes(y))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-ar", "44100", str(path)], check=True)
    tmp.unlink()


def dubbed_cartoon(path, seed=0):
    """Talking cartoon whose soundtrack comes from a different take (lips and speech don't match)."""
    a, b = path.with_name(path.stem + "_a.mp4"), path.with_name(path.stem + "_b.mp4")
    talking_cartoon(a, 0.0, seed=seed)
    talking_cartoon(b, 0.0, seed=seed + 100)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(a), "-i", str(b), "-map", "0:v", "-map", "1:a", "-c", "copy",
                    str(path)], check=True)
    a.unlink()
    b.unlink()


def main():
    MEDIA.mkdir(exist_ok=True)
    rows = []

    def add(name, data, label, manip, context, case="", owner="placeholder", generator=""):
        p = MEDIA / name
        if callable(data):
            data(p)
        else:
            p.write_bytes(data)
        rows.append({"file": name, "label": label, "manipulation": manip, "case": case, "context": context, "owner": owner,
                     "generator": generator})

    D, LIB = "2026-11-02", "Photo of the campus library, taken {}"
    for i in range(1, 7):
        add(f"ph_img_real_{i:02d}.jpg", jpeg(scene(), ex=exif(D)), "real", "none", LIB.format(D), "demo3" if i == 1 else "")
    for i in range(1, 3):
        add(f"ph_img_real_noexif_{i:02d}.jpg", jpeg(scene()), "real", "none", "Screenshot shared in the group chat")
    for i in range(1, 5):
        add(f"ph_img_splice_{i:02d}.jpg", splice(D, "GIMP 2.10" if i % 2 else None), "manipulated", "splice",
            f"Flood waters reach the campus library on {'2026-11-15' if i == 1 else D}", "demo1" if i == 1 else "")
    for i in range(1, 3):
        add(f"ph_img_copymove_{i:02d}.jpg", copy_move(D), "manipulated", "copy_move", LIB.format(D))
    for i in range(1, 4):
        add(f"ph_img_wrongdate_{i:02d}.jpg", jpeg(scene(), ex=exif(D)), "manipulated", "wrong_date", LIB.format("2026-11-15"))

    Q = 'Interview recorded 2026-11-10. She said "we cancelled the fest"'
    for i in range(1, 5):
        add(f"ph_aud_real_{i:02d}.wav", wav_bytes(speech()), "real", "none", Q)
    for i in range(1, 5):
        add(f"ph_aud_splice_{i:02d}.wav", wav_bytes(spliced_speech()), "manipulated", "audio_splice", Q)
    for i in range(1, 3):
        add(f"ph_aud_click_{i:02d}.wav", wav_bytes(clicked_speech()), "manipulated", "audio_splice", Q)

    V = "2026-11-10"
    for i in range(1, 4):
        add(f"ph_vid_real_{i:02d}.mp4", lambda p: video(p, speech(), V), "real", "none", Q)
    for i in range(1, 4):
        c = round(float(rng.uniform(3, 5)), 1)
        add(f"ph_vid_voice_{i:02d}.mp4", lambda p: video(p, spliced_speech(c), V, cut=c), "manipulated", "voice_clone", Q,
            "demo2" if i == 1 else "")
    for i in range(1, 3):
        add(f"ph_vid_wrongdate_{i:02d}.mp4", lambda p: video(p, speech(), "2026-10-01"), "manipulated", "wrong_date", Q)

    # ai_generated stand-ins, after everything else so the files above stay byte-identical (same rng sequence).
    # They only exercise the plumbing: XMP DigitalSourceType tag, a generator name in a PNG text chunk, 2x upsampling.
    for i in range(1, 3):
        add(f"ph_img_ai_xmp_{i:02d}.jpg", jpeg_xmp(scene()), "manipulated", "ai_generated", LIB.format(D), generator="ph-xmp")
    for i in range(1, 3):
        add(f"ph_img_ai_param_{i:02d}.png", png_params(scene()), "manipulated", "ai_generated", LIB.format(D), generator="ph-sd")
    for i in range(1, 3):
        add(f"ph_img_ai_up_{i:02d}.png", png_params(upsampled(), text=None), "manipulated", "ai_generated", LIB.format(D),
            generator="ph-upsample")

    # cloned_voice / ai_video stand-ins (and untouched talking heads to compare with), last for the same reason
    T = 'Statement recorded 2026-11-12. She said "the venue is closed"'
    for i in range(1, 3):
        add(f"ph_aud_clone_{i:02d}.wav", lambda p, i=i: cloned_voice(p, seed=i), "manipulated", "cloned_voice", T,
            generator="ph-gated-upsampled")
    for i in range(1, 3):
        add(f"ph_vid_face_real_{i:02d}.mp4", lambda p, i=i: talking_cartoon(p, 0.0, seed=20 + i), "real", "none", T)
    add("ph_vid_ai_late_01.mp4", lambda p: talking_cartoon(p, 0.3, seed=31), "manipulated", "ai_video", T, generator="ph-offset")
    add("ph_vid_ai_dub_01.mp4", lambda p: dubbed_cartoon(p, seed=32), "manipulated", "ai_video", T, generator="ph-dubbed")

    labels = ROOT / "labels.csv"
    with open(labels, encoding="utf-8") as f:
        keep = [r for r in csv.DictReader(f) if not r["file"].startswith("ph_")]
    with open(labels, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["file", "label", "manipulation", "case", "context", "owner", "generator"], restval="")
        w.writeheader()
        w.writerows(keep + rows)
    print(f"wrote {len(rows)} placeholder files to {MEDIA}")


if __name__ == "__main__":
    main()
