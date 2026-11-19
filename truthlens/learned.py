"""Trained AI-generation detectors on the GPU host (llm.detect, DETECTORS_URL) as findings: learned_synthetic_image /
_audio / _video. One finding per file when the detector's score reaches its own threshold; below it, nothing (the
score is logged with the call). Every finding names the model version, its score and threshold, what the server says it
was trained on, and where it is weak. The model card comes from the server (llm.model_card), never written here.
Weight: 0 until calibrate.py has enough of our own media (MIN_OWN), so the finding shows as uncalibrated."""
import base64
from PIL import Image
from . import llm
from .indicator import Indicator

WEAK = ("Weak on generators not in training (e.g. Sora, Kling, Pika, Midjourney); no face-swap training data.")
# "untouched", not the brief's "genuine": the output wording rules ban genuine / real / authentic / fake / proof / verdict
FIRST_RESULT = ("On our own untouched phone and WhatsApp recordings, the first trained detectors flagged 2 of 3 as AI. They "
                "were trained on studio speech and stock footage. They don't count toward the label until calibrated on our "
                "own media.")
# "untouched test files", not the brief's "real": same wording rule
V2 = ("Detectors v2 sharply lowered false alarms on our phone media (held-out: image 1/29, video 0/22, audio 10/114 windows; "
      "wide intervals). v2 catches fewer AI videos than v1 (89% vs 98% on the test set), Sora detection fell to 12% on a small sample, and audio false alarms on public untouched test files are 9.3%.")
ALT = {
    "image": ["phone processing (HDR, night mode, beauty filters, upscaling) the training photos did not have",
              "heavy recompression or resizing by a messenger or social site",
              "screenshots, graphics or edited composites, which the detector was not trained on"],
    "audio": ["phone voice-memo processing (automatic gain, noise reduction) unlike the studio training recordings",
              "a different microphone, room or language from the training speech",
              "lossy compression by a messenger (AAC/Opus)"],
    "video": ["stock-footage style differences: phone footage, low light or shaky handheld video the training clips lacked",
              "messenger recompression (WhatsApp, Telegram) and resizing",
              "filters, stabilisation or other in-app processing"],
}


def analyze(path, case, modality, seconds=0.0):
    """[Indicator] (empty below threshold or when the server is unavailable: then logged as check_not_run)."""
    r = llm.detect(path, modality, case, seconds)
    if not r or r.get("score", 0) < r.get("threshold", 1):
        return []
    s, thr, ver, loc = r["score"], r["threshold"], r.get("model_version", "?"), r.get("location", {})
    head = (f"A trained AI-generation detector ({ver}) scores this {modality} {s:.2f}, at or above its threshold of {thr:.2f}")
    trained = r.get("trained_on") or "not reported by the detector server (see the model card on the Calibration page)"
    tail = f" Trained on: {trained if isinstance(trained, str) else ', '.join(trained)}. {WEAK} {FIRST_RESULT} {V2}"
    location, artifact, where = {"model_version": ver, "threshold": thr}, None, ""
    if modality == "image":
        try:
            heat = case.artifact_path(f"{path.stem}_learned.png")
            heat.write_bytes(base64.b64decode(loc["heatmap_png_base64"]))
            artifact = str(heat)
        except (KeyError, ValueError, OSError):
            pass
        boxes = sorted(loc.get("boxes") or [], key=lambda b: -b.get("weight", 0))
        if boxes:
            try:
                with Image.open(path) as im:
                    sx, sy = im.width / r.get("width", im.width), im.height / r.get("height", im.height)
            except OSError:
                sx = sy = 1.0
            x1, y1, x2, y2 = boxes[0]["box"]
            location["bbox"] = [round(x1 * sx), round(y1 * sy), round((x2 - x1) * sx), round((y2 - y1) * sy)]
            where = "; covering the region it marks lowers the score most (occlusion map)"
    elif modality == "audio":
        win = [[w["start_s"], w["end_s"], w["score"]] for w in loc.get("windows", [])]
        if win:
            top = max(win, key=lambda w: w[2])
            location.update(start=top[0], end=top[1], windows=win)
            n = sum(w[2] >= thr for w in win)
            where = f"; {n} of {len(win)} two-second windows reach it, the highest at {top[0]:.0f}-{top[1]:.0f} s"
    else:
        segs = [[g["start_s"], g["end_s"], g["score"], g.get("faces_detected", 0)] for g in loc.get("segments", [])]
        if segs:
            top = max(segs, key=lambda g: g[2])
            location.update(t=top[0], end=top[1], segments=segs)
            where = f"; highest in the segment {top[0]:.0f}-{top[1]:.0f} s"
        crops = []
        for n, m in enumerate(loc.get("most_suspicious", [])[:4]):
            b64 = m.get("face_jpeg_base64") or m.get("frame_jpeg_base64")
            if b64:
                p = case.artifact_path(f"{path.stem}_learned_{n}.jpg")
                p.write_bytes(base64.b64decode(b64))
                crops.append({"file": p.name, "score": m.get("score"), "face": "face_jpeg_base64" in m,
                              "t": m.get("frame_index", 0) / r.get("frames_sampled_fps", 2)})
        if crops:
            location["crops"] = crops
            artifact = str(case.artifact_path(crops[0]["file"]))
    return [Indicator(path.name, modality, f"learned_synthetic_{modality}", float(s), "medium", head + where + "." + tail,
                      location, artifact, ALT[modality])]
