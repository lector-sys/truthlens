"""OWNER A. Image forensics. Working: EXIF, ELA. TODO: noise map, copy-move, AI classifier, faces."""
import io
import numpy as np
from PIL import Image, ImageChops, ExifTags
from ..indicator import Indicator

EDIT_SOFTWARE = ("photoshop", "gimp", "lightroom", "snapseed", "canva", "picsart", "stable diffusion", "midjourney", "dall")


def analyze(path, case):
    img = Image.open(path)
    return exif(path, img) + ela(path, img, case)
    # TODO A: + noise_inconsistency(...) + copy_move(...) + ai_generated(...)


def exif(path, img):
    tags = {ExifTags.TAGS.get(k, k): v for k, v in img.getexif().items()}
    out = []
    if not tags:
        out.append(Indicator(path.name, "image", "exif_missing", 0.2, "low",
                             "No EXIF metadata. Common after social-media upload or deliberate stripping.",
                             alt_explanations=["Platform re-upload strips metadata", "Screenshot"]))
    sw = str(tags.get("Software", "")).lower()
    if any(s in sw for s in EDIT_SOFTWARE):
        out.append(Indicator(path.name, "image", "exif_edit_software", 0.6, "medium",
                             f"EXIF Software tag shows an editor/generator: '{tags['Software']}'.",
                             location={"meta": "Software"}, alt_explanations=["Benign edit (crop/colour)"]))
    return out


def ela(path, img, case, quality=90, z=3.0):
    """Error Level Analysis: re-save at known JPEG quality, look for regions that re-compress differently."""
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    diff = np.asarray(ImageChops.difference(rgb, Image.open(buf)), dtype=np.float32).mean(axis=2)

    heat = (255 * diff / max(diff.max(), 1)).astype(np.uint8)
    art = case.artifact_path(f"{path.stem}_ela.png")
    Image.fromarray(heat).save(art)

    # 32px blocks; flag blocks far above the image's typical error level
    b = 32
    h, w = (diff.shape[0] // b) * b, (diff.shape[1] // b) * b
    if not h or not w:
        return []
    blocks = diff[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))
    mu, sd = blocks.mean(), blocks.std() + 1e-6
    ys, xs = np.where(blocks > mu + z * sd)
    if not len(ys):
        return []
    x0, y0, x1, y1 = xs.min() * b, ys.min() * b, (xs.max() + 1) * b, (ys.max() + 1) * b
    score = float(min(1.0, (blocks.max() - mu) / sd / 10))
    return [Indicator(path.name, "image", "ela_hotspot", score, "low",
                      f"{len(ys)} region(s) re-compress differently from the rest of the image (>{z}σ).",
                      location={"bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]}, artifact=str(art),
                      alt_explanations=["High-detail/edge regions", "Multiple re-saves", "Text overlays"])]
