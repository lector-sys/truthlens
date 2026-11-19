"""OWNER A. Image forensics: EXIF, error level analysis, sensor-noise inconsistency, copy-move (ORB self-matching)."""
import io
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageChops, ExifTags, ImageFilter
from ..indicator import Indicator

EDIT_SOFTWARE = ("photoshop", "gimp", "lightroom", "snapseed", "canva", "picsart", "stable diffusion", "midjourney", "dall")


def analyze(path, case):
    img = Image.open(path)
    return exif(path, img) + ela(path, img, case) + noise_inconsistency(path, img, case) + copy_move(path, img, case)


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


def noise_inconsistency(path, img, case, block=32, z=3.0):
    """Sensor-noise map: median-filter residual, std per block; blocks far from the image's typical noise (robust z)."""
    g = img.convert("L")
    res = np.abs(np.asarray(g, np.float32) - np.asarray(g.filter(ImageFilter.MedianFilter(3)), np.float32))
    h, w = (res.shape[0] // block) * block, (res.shape[1] // block) * block
    if h < 4 * block or w < 4 * block:
        return []
    std = res[:h, :w].reshape(h // block, block, w // block, block).std(axis=(1, 3))
    med = np.median(std)
    # ponytail: 0.5 grey-level floor on the spread so flat synthetic regions (MAD = 0) don't flag everything
    dev = np.abs(std - med) / max(1.4826 * np.median(np.abs(std - med)), 0.5)
    mask = (dev > z).astype(np.uint8)
    art = case.artifact_path(f"{path.stem}_noise.png")
    Image.fromarray((255 * res / max(res.max(), 1)).astype(np.uint8)).save(art)
    if not mask.any():
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bx, by, bw, bh, area = stats[1:][stats[1:, 4].argmax()]  # largest connected flagged region
    side = "noisier" if std[mask > 0].mean() > med else "cleaner"
    # ponytail: lone outlier blocks are usually texture; full strength needs a region of >= 8 blocks (256x256 px at 32)
    score = min(1.0, dev.max() / 10) * min(1.0, area / 8)
    return [Indicator(path.name, "image", "noise_inconsistency", float(score), "low",
                      f"{int(mask.sum())} block(s) have {side} sensor noise than the rest of the image "
                      f"(up to {dev.max():.1f}σ), as when content from another source is pasted in.",
                      location={"bbox": [int(bx * block), int(by * block), int(bw * block), int(bh * block)]},
                      artifact=str(art),
                      alt_explanations=["Mixed textures (sky vs foliage)", "Denoise/beauty filter", "Upscaling or resizing"])]


def _box(pts):
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return [int(x0), int(y0), int(x1 - x0) + 1, int(y1 - y0) + 1]


def copy_move(path, img, case, min_matches=10, min_dist=40):
    """Cloned region: ORB keypoints matched against the same image, kept when many pairs share one shift vector."""
    g = np.asarray(img.convert("L"))
    kps, des = cv2.ORB_create(nfeatures=3000).detectAndCompute(g, None)
    if des is None or len(kps) < 3:
        return []
    pairs = []
    for ms in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des, des, k=3):
        ms = [m for m in ms if m.queryIdx != m.trainIdx]
        if len(ms) >= 2 and ms[0].distance < 0.75 * ms[1].distance:
            a, b = np.array(kps[ms[0].queryIdx].pt), np.array(kps[ms[0].trainIdx].pt)
            if np.hypot(*(b - a)) > min_dist:
                pairs.append((a, b) if (b - a)[0] > 0 or ((b - a)[0] == 0 and (b - a)[1] > 0) else (b, a))
    if len(pairs) < min_matches:
        return []
    # one clone = one shared shift; this drops scattered matches from repeating textures
    shifts = Counter(tuple(np.round((b - a) / 8).astype(int)) for a, b in pairs)
    top = shifts.most_common(1)[0][0]
    keep = [(a, b) for a, b in pairs if max(abs(np.round((b - a) / 8).astype(int) - top)) <= 1]
    keep = list({(tuple(a), tuple(b)): (a, b) for a, b in keep}.values())  # a<->b found twice
    if len(keep) < min_matches:
        return []
    src, dst = np.array([a for a, _ in keep]), np.array([b for _, b in keep])
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for a, b in keep:
        cv2.line(vis, tuple(map(int, a)), tuple(map(int, b)), (82, 115, 249), 1)
    art = case.artifact_path(f"{path.stem}_copymove.png")
    cv2.imwrite(str(art), vis)
    dx, dy = (dst - src).mean(axis=0)
    return [Indicator(path.name, "image", "copy_move", float(min(1.0, len(keep) / 50)), "medium",
                      f"{len(keep)} matching keypoint pairs share one shift ({dx:+.0f}, {dy:+.0f}) px: "
                      f"a region appears to be duplicated elsewhere in the image.",
                      location={"bbox": _box(src), "bbox2": _box(dst)}, artifact=str(art),
                      alt_explanations=["Repeating patterns (tiles, windows, fabric)", "Symmetric objects"])]
