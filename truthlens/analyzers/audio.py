"""OWNER B. Audio forensics. Edited: container metadata, noise-floor splice + click detection. Generated (signs of
synthetic or cloned speech): hard band limit, too-clean pauses without breaths, flat prosody. Spectrogram with the
flagged spans / cutoff drawn in, transcript. Emit location={"start": s, "end": s}."""
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
    if not wav:
        return out
    band = synthetic_bandlimit(*_native(path, meta), path.name)  # at the file's own rate: 16 kHz can't show 11-16 kHz cuts
    return out + band + analyze_audio(wav, case, path.name, path.stem, cutoff=band[0].location["hz"] if band else None)


def _native(path, meta, seconds=120):
    """Mono float32 at the file's own sample rate (first `seconds`) -> (y, sr), or (empty, 0) if there is none."""
    sr = next((int(s["sample_rate"]) for s in meta.get("streams", []) if s.get("codec_type") == "audio" and s.get("sample_rate")), 0)
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-t", str(seconds), "-f", "f32le", "-"],
                       capture_output=True)
    return (np.frombuffer(r.stdout, np.float32), sr) if sr and not r.returncode else (np.zeros(0, np.float32), 0)


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


def spectrogram(y, sr, flags, out, cutoff=None, spans=()):
    """Log-magnitude STFT image (low frequencies at the bottom) with flagged spans shaded red, extra `spans`
    [(start, end, rgba)] shaded in their colour, and a band-limit `cutoff` (Hz) drawn as a cyan line."""
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
    for a, b, rgba in spans:
        d.rectangle([a * px, 0, b * px, img.height], fill=rgba)
    if cutoff:
        row = max(0, img.height - 1 - round(min(cutoff, sr / 2) / (sr / 2) * (img.height - 1)))
        d.line([0, row, img.width, row], fill=(56, 189, 248, 255), width=2)
        d.text((4, max(0, row - 12)), f"cutoff {cutoff / 1000:.1f} kHz" + (" (above this view)" if cutoff >= sr / 2 else ""),
               fill=(56, 189, 248, 255))
    Image.alpha_composite(img, over).convert("RGB").save(out)
    for i in flags:
        i.artifact = str(out)
    return out


def analyze_audio(wav, case, file_name, stem, cutoff=None):
    """Decoded wav -> splice/click + synthetic-speech flags, spectrogram artifact, transcript artifact (for cross-modal)."""
    y, sr = load(wav)
    flags = splice_detection(y, sr, file_name)
    segs = llm.transcribe(wav, case)  # None when offline or the call failed (logged as check_not_run)
    if segs is not None:
        case.artifact_path(f"{stem}_transcript.json").write_text(json.dumps(segs), encoding="utf-8")
        case.log("transcript", file=file_name, segments=len(segs))
    sil, pauses = synthetic_silence(y, sr, file_name)
    pro, flat = synthetic_prosody(y, sr, file_name, segs)
    spec = spectrogram(y, sr, flags, case.artifact_path(f"{stem}_spec.png"), cutoff=cutoff,
                       spans=[(a, b, (80, 160, 255, 70)) for a, b in pauses] + [(a, b, (242, 181, 68, 60)) for a, b in flat])
    for i in sil + pro:
        i.artifact = str(spec)
    return flags + sil + pro


def _runs(mask):
    """[(start, end)] index ranges where a boolean array is True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def synthetic_bandlimit(y, sr, file_name):
    """Hard spectral cutoff well below Nyquist with near-zero energy above it (speech rendered or resampled at a lower
    rate, e.g. 8 / 11 / 12 / 16 kHz). Long-term spectrum of the louder half of the frames, at the file's own rate."""
    n = 4096 if sr > 24000 else 2048
    if not sr or len(y) < 8 * n:
        return []
    fr = np.lib.stride_tricks.sliding_window_view(y, n)[::n // 2] * np.hanning(n)
    pw = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    e = pw.sum(axis=1)
    spec = 10 * np.log10(pw[e >= np.median(e)].mean(axis=0) + 1e-20)
    spec = np.convolve(spec, np.ones(9) / 9, mode="same")
    f = np.fft.rfftfreq(n, 1 / sr)
    nyq, ref = sr / 2, np.median(spec[(f > 300) & (f < 3000)])
    live = np.flatnonzero(spec > ref - 20)
    fc = float(f[live.max()]) if len(live) else nyq  # edge: last frequency within 20 dB of the speech band
    if fc > 0.85 * nyq or nyq - fc < 1500:
        return []  # energy reaches (close to) Nyquist: no band limit
    drop = spec[(f > fc - 1200) & (f < fc - 400)].mean() - spec[(f > fc + 400) & (f < fc + 1200)].mean()
    above = np.median(spec[f > fc + 1000]) - ref
    if drop < 20 or above > -40:
        return []  # gentle roll-off (microphone, room) or real energy left above the edge
    return [Indicator(file_name, "audio", "synthetic_bandlimit", float(np.clip(drop / 50, 0, 1)), "low",
                      f"Energy stops abruptly at {fc / 1000:.1f} kHz ({drop:.0f} dB drop) and stays {-above:.0f} dB down up to "
                      f"{nyq / 1000:.1f} kHz: a hard band limit, as when speech is generated or resampled at a lower rate.",
                      location={"start": 0.0, "end": round(len(y) / sr, 2), "hz": round(fc)},
                      alt_explanations=["Phone or VoIP audio", "Low-bitrate MP3 / AAC", "Resampled from a lower sample rate"])]


def synthetic_silence(y, sr, file_name):
    """Pauses between words: noise floor near digital zero, or far cleaner (> 50 dB below speech) than real rooms
    give; and no breath-like bumps in the longer pauses. -> (indicators, [(start, end)] pause spans to shade)."""
    db, hop = _frame_db(y, sr)
    if len(db) < 100:
        return [], []
    level = np.percentile(db, 90)
    pauses = [(a, b) for a, b in _runs(db < level - 25) if b - a >= 15 and a > 0 and b < len(db)]  # >= 150 ms, inside speech
    if len(pauses) < 2:
        return [], []
    floor = float(np.median(np.concatenate([db[a:b] for a, b in pauses])))
    gap = level - floor
    long = [(a, b) for a, b in pauses if b - a >= 30]
    breaths = sum(bool((db[a:b] > floor + 12).any()) for a, b in long)
    if floor > -85 and gap < 50:
        return [], []
    score = 0.8 if floor <= -85 else float(np.clip(0.3 + (gap - 50) / 30, 0, 0.8))
    why = (f"Pauses between words are digital silence ({floor:.0f} dBFS)" if floor <= -85 else
           f"Pauses between words are {gap:.0f} dB below the speech, far cleaner than a real room")
    if len(long) >= 3 and not breaths:
        score, why = min(1.0, score + 0.1), why + f"; no breath-like sounds in {len(long)} longer pauses"
    a, b = max(pauses, key=lambda r: r[1] - r[0])
    spans = [(round(a * hop / sr, 2), round(b * hop / sr, 2)) for a, b in pauses]
    return [Indicator(file_name, "audio", "synthetic_silence", score, "low", why + ", as in synthetic or cloned speech.",
                      location={"start": round(a * hop / sr, 2), "end": round(b * hop / sr, 2)},
                      alt_explanations=["Noise gate or noise reduction", "Studio recording / edited voice-over"])], spans


def _f0(y, sr, frame=0.04, hop=0.01):
    """Autocorrelation pitch per 40 ms frame: (f0 Hz or nan if unvoiced, level dBFS, frame times)."""
    n, h = int(frame * sr), int(hop * sr)
    if len(y) < n:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    fr = np.lib.stride_tricks.sliding_window_view(y, n)[::h] * np.hanning(n)
    ac = np.fft.irfft(np.abs(np.fft.rfft(fr, 2 * n, axis=1)) ** 2, axis=1)[:, :n]
    ac /= ac[:, :1] + 1e-12
    lo, hi = int(sr / 400), int(sr / 70)  # 70-400 Hz
    lag = lo + ac[:, lo:hi].argmax(axis=1)
    db = 10 * np.log10((fr ** 2).mean(axis=1) + 1e-12)
    voiced = (ac[np.arange(len(ac)), lag] > 0.5) & (db > np.percentile(db, 90) - 20)
    return np.where(voiced, sr / lag, np.nan), db, np.arange(len(fr)) * hop


def synthetic_prosody(y, sr, file_name, segs=None, st_max=1.2, db_max=4.0):
    """Per sentence (transcript segments, else stretches between >= 300 ms pauses): pitch spread in semitones and
    level spread in dB over voiced frames. Natural speech is around 2-5 semitones and 4-8 dB; flag when the median
    sentence is below both limits. -> (indicators, [(start, end)] flat sentences to shade)."""
    f0, db, t = _f0(y, sr)
    if not len(f0):
        return [], []
    if segs:
        sent = [(s["start"], s["end"]) for s in segs]
    else:
        speech = np.isfinite(f0) | (db > np.percentile(db, 90) - 25)
        sent = [(t[a], t[b - 1]) for a, b in _runs(np.convolve(speech, np.ones(30), "same") > 0)]  # bridge < 300 ms gaps
    rows = []
    for a, b in sent:
        m = (t >= a) & (t <= b) & np.isfinite(f0)
        if m.sum() >= 20:
            st = 12 * np.log2(f0[m] / np.median(f0[m]))
            rows.append((a, b, float(st.std()), float(db[m].std())))
    if len(rows) < 2:
        return [], []
    st_med, db_med = np.median([r[2] for r in rows]), np.median([r[3] for r in rows])
    if st_med >= st_max or db_med >= db_max:
        return [], []
    flat = [(round(a, 2), round(b, 2)) for a, b, st, dv in rows if st < st_max]
    a, b, st, dv = min(rows, key=lambda r: r[2])
    score = float(np.clip(0.5 * (1 - st_med / st_max) + 0.5 * (1 - db_med / db_max), 0, 1))
    return [Indicator(file_name, "audio", "synthetic_prosody", score, "low",
                      f"Pitch varies only {st_med:.1f} semitones and loudness {db_med:.1f} dB per sentence (median of "
                      f"{len(rows)}; natural speech is about 2-5 semitones and 4-8 dB): unusually flat, as in synthetic speech.",
                      location={"start": round(a, 2), "end": round(b, 2)},
                      alt_explanations=["Monotone reading", "Scripted announcement", "Heavy compression or AGC"])], flat
