"""python test_smoke.py: end-to-end check of vault, image analyzer, fusion, cross-modal, report."""
import io
import numpy as np
from PIL import Image
from truthlens.vault import Case
from truthlens.indicator import Indicator
from truthlens import pipeline, report, fusion, crossmodal
from truthlens.analyzers import audio
import calibrate


def jpeg(arr, q=75):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=q)
    return buf.getvalue()


# Spliced image: smooth JPEG background, uncompressed noise patch pasted in and saved once more at high quality
rng = np.random.default_rng(0)
bg = np.asarray(Image.open(io.BytesIO(jpeg(np.full((256, 256, 3), 120, np.uint8), q=50)))).copy()
bg[96:160, 96:160] = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)

case = Case("smoke_test")
try:
    case.ingest("spliced.jpg", jpeg(bg, q=95))
    res = pipeline.run(case, 'Taken on 2024-05-01. He said "hello".')
    types = {i.type for i in res["indicators"]}
    assert "ela_hotspot" in types, types
    x, y, w, h = next(i for i in res["indicators"] if i.type == "ela_hotspot").location["bbox"]
    assert x <= 96 and y <= 96 and x + w >= 160 and y + h >= 160, (x, y, w, h)  # box covers the splice
    assert all(res["integrity"].values())
    html = report.render(case, res)
    assert "spliced.jpg" in html and "data:image/png;base64," in html  # ELA heatmap embedded
    assert "Inspect spliced.jpg region" in html and res["summary"] in html  # human check + (offline) summary

    # fusion: monotonic, bounded, labelled
    assert fusion.combine([]) == 0 and fusion.label(0) == "No significant indicators"
    strong = [Indicator("f", "image", t, 1.0, "high", "") for t in "xyz"]
    assert 0.99 < fusion.combine(strong) <= 1 and fusion.label(fusion.combine(strong)) == "Strong indicators"
    weak = Indicator("f", "video", "ela_hotspot", 0.3, "low", "")
    assert fusion.combine([weak] * 60) == fusion.combine([weak])  # repeated frames don't saturate

    # cross-modal: aligned A/V flags link, distant ones don't
    v = Indicator("v.mp4", "video", "temporal", 0.6, "low", "", location={"t": 12.0})
    a = Indicator("a.wav", "audio", "splice", 0.6, "low", "", location={"start": 12.3})
    far = Indicator("a.wav", "audio", "splice", 0.6, "low", "", location={"start": 20.0})
    assert len(crossmodal.link_timestamps([v, a, far])) == 1
    # calibration table: per-type counts and case-level catch/false-alarm
    table = calibrate.summarize([
        {"label": "manipulated", "types": {"ela_hotspot", "copy_move"}, "verdict": "Strong indicators"},
        {"label": "manipulated", "types": {"ela_hotspot"}, "verdict": "Weak indicators"},
        {"label": "real", "types": {"ela_hotspot"}, "verdict": "Moderate indicators"},
    ])
    assert "| ela_hotspot | 2/2 | 1/1 | 67% |" in table and "caught 1/2" in table and "false alarms 1/1" in table, table
    assert "'ela_hotspot': 0.1" in table and "'copy_move': 0.3" in table, table  # suggested fusion weights
    fusion.TYPE_WEIGHT["ela_hotspot"] = 0.0
    assert fusion.combine([weak]) == 0  # calibrated override beats the generic reliability weight
    del fusion.TYPE_WEIGHT["ela_hotspot"]

    # audio: noise-floor jump at 4 s and a click in silence are found; a steady floor gives nothing
    sr = 16000
    steady = rng.normal(0, 10 ** (-50 / 20), 8 * sr).astype(np.float32)
    joined = steady.copy()
    joined[4 * sr:] *= 10  # +20 dB floor
    assert audio.splice_detection(steady, sr, "a.wav") == []
    sp = [i for i in audio.splice_detection(joined, sr, "a.wav") if i.type == "audio_splice"]
    assert len(sp) == 1 and sp[0].location["start"] < 4 < sp[0].location["end"], sp
    clicked = steady.copy()
    clicked[2 * sr] += 0.5
    ck = [i for i in audio.splice_detection(clicked, sr, "a.wav") if i.type == "audio_click"]
    assert len(ck) == 1 and ck[0].location["start"] < 2 < ck[0].location["end"], ck

    # cross-modal: claimed date vs EXIF at the claim's precision; quote vs transcript
    ex = Image.Exif()
    ex[306] = "2026:11:02 10:00:00"
    buf = io.BytesIO()
    Image.fromarray(bg).save(buf, "JPEG", exif=ex)
    case.ingest("dated.jpg", buf.getvalue())
    assert crossmodal.claimed_date_vs_metadata({"when": ["2026-11", "2026-11-02", "1/2/24"]}, case) == []
    bad = crossmodal.claimed_date_vs_metadata({"when": ["2026-11-15"]}, case)
    assert [(i.file, i.type) for i in bad] == [("dated.jpg", "claim_date_mismatch")], bad
    case.ingest("clip.wav", b"")
    case.artifact_path("clip_transcript.json").write_text('[{"start":0,"end":2,"text":"We postponed the fest."}]')
    q = crossmodal.quote_vs_transcript({"quotes": ["we postponed the fest", "we cancelled everything"]}, case)
    assert [i.location["span"] for i in q] == ["we cancelled everything"], q
    print("OK", sorted(types), res["assessment"]["case"])
finally:
    case.delete()
