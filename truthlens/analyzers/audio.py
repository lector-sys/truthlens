"""OWNER B. Audio forensics: container metadata, noise-floor splice + click detection, spectrogram, transcript.
Emit location={"start": s, "end": s}."""
import json, subprocess, wave
import numpy as np
from PIL import Image, ImageDraw
from .. import llm
from ..indicator import Indicator
from .media_meta import ffprobe, encoder_tags

ALT = ["Speaker moved / door opened", "Automatic gain control", "Microphone bump"]


def analyze(path, case):
    meta = ffprobe(path)
    case.log("ffprobe", file=path.name, format=meta.get("format", {}).get("format_name"))
    out = []
    enc = encoder_tags(meta).get("encoder", "")
    if enc and "lavf" in enc.lower():
        out.append(Indicator(path.name, "audio", "reencoded", 0.25, "low",
                             f"Audio written by FFmpeg ({enc}): processed after capture.",
                             location={"meta": "encoder"}, alt_explanations=["Platform transcoding"]))
    wav = decode(path, case)
    return out + (analyze_audio(wav, case, path.name, path.stem) if wav else [])


def decode(path, case):
    """Any audio/video file -> 16 kHz mono wav artifact, or None if there is no decodable audio."""
    wav = case.artifact_path(f"{path.stem}_16k.wav")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       capture_output=True, text=True)
    if r.returncode or not wav.exists() or wav.stat().st_size <= 44:
        case.log("no_audio", file=path.name, error=r.stderr.strip()[-300:])
        return None
    return wav


def load(wav):
    """16-bit PCM wav -> (float32 mono in [-1, 1], sample rate)."""
    with wave.open(str(wav)) as w:
        sr, ch, raw = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    y = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    return (y.reshape(-1, ch).mean(axis=1) if ch > 1 else y), sr


def _frame_db(y, sr):
    """RMS level in dB of 25 ms frames with a 10 ms hop."""
    n, hop = int(0.025 * sr), int(0.010 * sr)
    if len(y) < n:
        return np.array([]), hop
    f = np.lib.stride_tricks.sliding_window_view(y, n)[::hop]
    return 20 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-10), hop


def _merge(t, s, gap=0.3, top=10):
    """Cluster times closer than `gap` s, keep the strongest of each cluster, return top-N [(t, score)] by score."""
    if not len(t):
        return []
    o = np.argsort(t)
    t, s = t[o], s[o]
    cid = np.concatenate([[0], np.cumsum(np.diff(t) >= gap)])
    o = np.lexsort((-s, cid))                      # by cluster, strongest first
    _, first = np.unique(cid[o], return_index=True)
    keep = o[first]
    return sorted(zip(t[keep].tolist(), s[keep].tolist()), key=lambda x: -x[1])[:top]


def _ind(file_name, typ, t, score, why):
    return Indicator(file_name, "audio", typ, float(min(1.0, score)), "low", why,
                     location={"start": max(0.0, t - 0.25), "end": t + 0.25}, alt_explanations=ALT)


def splice_detection(y, sr, file_name, jump_db=6.0, z=8.0):
    """(a) background noise-floor level jumps between 0.5 s windows, (b) sample-level clicks outside loud speech."""
    db, hop = _frame_db(y, sr)
    if not len(db):
        return []
    out = []
    per = int(0.5 * sr / hop)  # frames per 0.5 s window
    nw = len(db) // per
    if nw >= 2:
        floors = np.percentile(db[:nw * per].reshape(nw, per), 10, axis=1)
        jumps = np.abs(np.diff(floors))
        i = np.flatnonzero(jumps > jump_db)
        for t, j in _merge((i + 1) * per * hop / sr, jumps[i]):
            out.append(_ind(file_name, "audio_splice", t, j / 20,
                            f"Background noise level changes by {j:.1f} dB at {t:.2f}s, as when two recordings are joined."))
    dx = np.abs(np.diff(y))
    med = np.median(dx)
    zs = (dx - med) / (1.4826 * np.median(np.abs(dx - med)) + 1e-9)
    idx = np.flatnonzero(zs > z)
    # ignore steep edges inside loud speech; level = 110 ms median so the click doesn't make its own frame "loud"
    level = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(db, 5, mode="edge"), 11), axis=1)
    idx = idx[level[np.minimum(idx // hop, len(db) - 1)] < np.median(db) + 3]
    for t, zv in _merge(idx / sr, zs[idx]):
        out.append(_ind(file_name, "audio_click", t, zv / 40,
                        f"Sample-level discontinuity ({zv:.0f}σ) at {t:.2f}s in a quiet passage, typical of a hard cut."))
    return out


def spectrogram(y, sr, flags, out):
    """Log-magnitude STFT image (low frequencies at the bottom) with flagged spans shaded red."""
    y = np.pad(y, (0, max(0, 512 - len(y))))
    f = np.lib.stride_tricks.sliding_window_view(y, 512)[::160] * np.hanning(512)
    db = 20 * np.log10(np.abs(np.fft.rfft(f, axis=1)) + 1e-9).T
    db = 255 * (db - db.min()) / max(db.max() - db.min(), 1e-9)
    img = Image.fromarray(db[::-1].astype(np.uint8)).convert("RGBA")
    scale = min(1.0, 1600 / img.width)
    img = img.resize((max(1, int(img.width * scale)), img.height))
    over = Image.new("RGBA", img.size)
    d = ImageDraw.Draw(over)
    px = sr / 160 * scale  # pixels per second
    for i in flags:
        d.rectangle([i.location["start"] * px, 0, i.location["end"] * px, img.height], fill=(255, 0, 0, 90))
    Image.alpha_composite(img, over).convert("RGB").save(out)
    for i in flags:
        i.artifact = str(out)
    return out


def analyze_audio(wav, case, file_name, stem):
    """Decoded wav -> splice/click flags + spectrogram artifact + transcript artifact (for cross-modal)."""
    y, sr = load(wav)
    flags = splice_detection(y, sr, file_name)
    spectrogram(y, sr, flags, case.artifact_path(f"{stem}_spec.png"))
    try:
        segs = llm.transcribe(wav)
    except Exception as e:  # network / API error: no transcript, analysis continues
        case.log("transcript_error", file=file_name, error=repr(e))
        segs = None
    if segs is not None:
        case.artifact_path(f"{stem}_transcript.json").write_text(json.dumps(segs), encoding="utf-8")
        case.log("transcript", file=file_name, segments=len(segs))
    return flags
