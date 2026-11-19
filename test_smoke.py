"""python test_smoke.py: end-to-end check of vault, image analyzer, fusion, cross-modal, report."""
import io
import numpy as np
from PIL import Image
from truthlens.vault import Case
from truthlens.indicator import Indicator
from truthlens import pipeline, report, fusion, crossmodal


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
    assert "spliced.jpg" in report.render(case, res)

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
    print("OK", sorted(types), res["assessment"]["case"])
finally:
    case.delete()
