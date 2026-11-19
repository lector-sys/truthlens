"""OWNER A. Image forensics. Edited: EXIF, error level analysis, sensor-noise inconsistency, copy-move (ORB self-matching).
Generated: provenance metadata (C2PA / XMP DigitalSourceType / generator names), spectral upsampling traces, visual
artifacts listed by the vision model. Formats Pillow can't read are analysed via a converted PNG copy."""
import io, json, re, shutil, subprocess
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageChops, ExifTags, ImageFilter
from .. import llm
from ..indicator import Indicator
from .media_meta import readable_image

EDIT_SOFTWARE = ("photoshop", "gimp", "lightroom", "snapseed", "canva", "picsart")
GENERATORS = ("midjourney", "dall-e", "dall·e", "dalle", "stable diffusion", "stablediffusion", "sdxl", "comfyui",
              "automatic1111", "novelai", "firefly", "imagen", "gemini", "flux", "leonardo", "ideogram", "chatgpt", "openai",
              "sora", "runway", "nano banana", "grok", "kandinsky", "dreamstudio", "bing image creator", "copilot designer")
CONVERTED_NOTE = "Analysed a PNG converted from {ext}: compression-based checks (ELA) see the conversion, not the original file"


def analyze(path, case):
    src = readable_image(path, case)
    out = synthetic_provenance(path, case)  # raw bytes of the evidence itself, never the converted copy
    if src is None:
        case.log("check_not_run", check="pixel checks", file=path.name, reason=f"{path.suffix} could not be decoded or converted")
        return out
    img = Image.open(src)
    if src != path:  # converted: EXIF did not survive, ELA measures the PNG
        case.log("check_not_run", check="exif", file=path.name, reason=f"metadata not carried over when converting {path.suffix}")
        found = ela(path, img, case)
        for i in found:
            i.alt_explanations.append(CONVERTED_NOTE.format(ext=path.suffix))
        out += found
    else:
        out += exif(path, img) + ela(path, img, case)
    return (out + noise_inconsistency(path, img, case) + copy_move(path, img, case) + spectral_upsampling(path, img, case)
            + synthetic_visual_artifacts(path, src, img.size, case))


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


C2PA_ALT = ["AI-assisted edit of a real photo", "Tool adds the tag by default", "Metadata can be added or copied by anyone"]


def _metadata_text(data, img):
    """Lower-cased metadata a generator could write: XMP packet, PNG text chunks, EXIF strings, C2PA manifest region."""
    parts = [m.group(0) for m in re.finditer(rb"<x:xmpmeta.{0,200000}?</x:xmpmeta>", data, re.S)]
    if (k := data.lower().find(b"c2pa")) >= 0:
        parts.append(data[k:k + 200000])
    text = b" ".join(parts).decode("latin-1")
    if img is not None:
        text += " ".join(v if isinstance(v, str) else v.decode("latin-1") for v in img.info.values() if isinstance(v, (str, bytes)))
        text += " ".join(str(v) for v in img.getexif().values() if isinstance(v, (str, bytes)))
    return text.lower()


def _c2patool(path):
    """(claim generator, digital source types, validation problems) from c2patool, or None if not installed / unreadable."""
    if not shutil.which("c2patool"):
        return None
    r = subprocess.run(["c2patool", str(path)], capture_output=True, text=True)
    try:
        store = json.loads(r.stdout)
    except ValueError:
        return None
    m = store.get("manifests", {}).get(store.get("active_manifest"), {})
    raw = json.dumps(m).lower()
    # older releases: "validation_status" list; newer: "validation_results" -> activeManifest -> failure
    failures = store.get("validation_status") or (store.get("validation_results") or {}).get("activeManifest", {}).get("failure", [])
    problems = [f.get("code", str(f)) if isinstance(f, dict) else str(f) for f in failures]
    return (str(m.get("claim_generator") or m.get("claim_generator_info") or ""),
            sorted(set(re.findall(r"digitalsourcetype/(\w+)", raw))), problems)


def synthetic_provenance(path, case):
    """C2PA / JUMBF manifest, XMP-IPTC DigitalSourceType and generator names in the evidence file's own metadata bytes."""
    data = path.read_bytes()
    low = data.lower()
    try:
        img = Image.open(path)
    except (OSError, ValueError):
        img = None
    c2pa = b"c2pa" in low and (b"jumb" in low or b"cabx" in low)
    tool = _c2patool(path) if c2pa else None
    if tool:
        gen, kinds, problems = tool
        status = "C2PA manifest verified by c2patool" if not problems else f"C2PA manifest present, validation failed ({problems[0]})"
        types = " ".join(kinds)
    else:
        gen, status, types = "", "C2PA manifest present (not verified)", ""
    meta = _metadata_text(data, img)
    types += " " + " ".join(re.findall(r"digitalsourcetype\W{0,60}?(?:[\w:./-]*/)?(\w*algorithmicmedia|compositesynthetic)", meta))
    names = sorted({g for g in GENERATORS if g in meta or g in gen.lower()})
    case.log("provenance_scan", file=path.name, c2pa=c2pa, c2pa_status=status if c2pa else None, generators=names)
    found, score = [], 0.0
    if "compositewithtrainedalgorithmicmedia" in types or "compositesynthetic" in types:
        found.append("metadata declares DigitalSourceType = composite with AI-generated parts")
        score = 0.6
    elif "trainedalgorithmicmedia" in types:
        found.append("metadata declares DigitalSourceType = trainedAlgorithmicMedia (made by a generative model)")
        score = 0.9
    if names:
        found.append(f"metadata names an image generator: {', '.join(names)}")
        score = max(score, 0.7)
    if c2pa:
        found.append(status + ("" if found else "; it declares no AI generation"))
        score = max(score, 0.05)
    if not found:
        return []
    text = "; ".join(found)
    return [Indicator(path.name, "image", "synthetic_provenance", score, "medium", text[0].upper() + text[1:] + ".",
                      location={"meta": "C2PA / XMP DigitalSourceType / generator"}, alt_explanations=C2PA_ALT)]


def spectral_upsampling(path, img, case, size=512, z=6.0):
    """FFT of the high-pass residual (centre crop, Hann window): periodic off-centre peaks, as left by upsampling
    layers, and the residual energy near Nyquist relative to the mid band, which interpolation removes."""
    g = np.asarray(img.convert("L"), np.float32)
    n = min(size, *g.shape) // 2 * 2
    if n < 128:
        return []
    y0, x0 = (g.shape[0] - n) // 2, (g.shape[1] - n) // 2
    g = g[y0:y0 + n, x0:x0 + n]
    res = g - cv2.blur(g, (3, 3))
    spec = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(res * np.outer(np.hanning(n), np.hanning(n))))) ** 2 + 1e-6)
    yy, xx = np.indices(spec.shape) - n // 2
    r = np.hypot(yy, xx) / n  # cycles per pixel
    bins = np.minimum((r * 100).astype(int), 70)  # azimuthal average in 0.01 cycle/px rings
    prof = np.bincount(bins.ravel(), 10 ** spec.ravel()) / np.maximum(np.bincount(bins.ravel()), 1)
    tail = float(prof[38:50].mean() / max(prof[15:25].mean(), 1e-9))
    dev = spec - cv2.blur(spec, (9, 9))  # peaks: far above their 9x9 neighbourhood, off-centre, off the axes
    ok = (r > 0.08) & (np.abs(xx) > 2) & (np.abs(yy) > 2)
    if path.suffix.lower() in (".jpg", ".jpeg"):  # ponytail: JPEG 8 px block-grid peaks (k*n/8) skipped, not judged
        ok &= ~((xx % (n // 8) == 0) & (yy % (n // 8) == 0))
    d = dev[ok]
    zs = (d - np.median(d)) / (1.4826 * np.median(np.abs(d - np.median(d))) + 1e-9)
    peaks = int((zs > z).sum())
    art = case.artifact_path(f"{path.stem}_spectrum.png")
    Image.fromarray((255 * (spec - spec.min()) / max(np.ptp(spec), 1e-9)).astype(np.uint8)).save(art)
    case.log("spectral", file=path.name, peaks=peaks, nyquist_tail=round(tail, 3))
    # ponytail: thresholds set on noise / photo / 2x-upsampled tests; calibrate.py decides how much this type counts
    s_tail = float(np.clip((TAIL_OK - tail) / TAIL_OK, 0, 1))
    s_peaks = float(min(1.0, peaks / 12))
    score = max(s_tail, s_peaks)
    if score < 0.15:
        return []
    why = []
    if s_peaks >= 0.15:
        why.append(f"{peaks} periodic off-centre peaks in the noise spectrum")
    if s_tail >= 0.15:
        why.append(f"little fine-detail energy near the pixel limit ({tail:.2f} of the mid-band level)")
    text = "; ".join(why)
    return [Indicator(path.name, "image", "spectral_upsampling", score, "low",
                      text[0].upper() + text[1:] + ", as upsampling layers in image generators can leave.", artifact=str(art),
                      alt_explanations=["JPEG block grid", "Resizing / upscaling after capture", "Screen capture",
                                        "Halftone print or moiré"])]


TAIL_OK = 0.35
ARTIFACT_KINDS = ["garbled_text", "hand_anomaly", "teeth_anomaly", "shadow_inconsistency", "reflection_inconsistency",
                  "warped_lines", "melted_detail", "repeated_faces"]
ARTIFACT_PROMPT = """You inspect one image for a forensic reviewer and list concrete, locatable visual artifacts only.
Allowed "kind" values (use exactly one of these strings):
- garbled_text: letters, signs or labels that are garbled or nonsensical
- hand_anomaly: wrong number of fingers, fused, bent or melted fingers
- teeth_anomaly: teeth merged into one strip, or the wrong number or shape
- shadow_inconsistency: shadows pointing different ways, or missing where others exist
- reflection_inconsistency: a reflection that does not match its object
- warped_lines: lines that should be straight (door frames, railings, tiles) bend or break
- melted_detail: background detail that smears or melts into shapeless texture
- repeated_faces: the same face repeated in a crowd
For each artifact: "description" = one line saying what is visible and where; "clarity" = "clear" or "subtle";
"bbox" = [x1, y1, x2, y2], four integers from 0 to 1000 relative to the image width and height, (0, 0) = top left.
List at most 5, the clearest first. Only list artifacts you can point to. If none are clearly visible, return {"artifacts": []}: that is the expected answer
for most images. Never judge whether the image as a whole is real, fake, generated or edited, and never say so."""
ARTIFACT_VERSION = "v2"
ARTIFACT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["artifacts"],
                   "properties": {"artifacts": {"type": "array", "items": {
                       "type": "object", "additionalProperties": False, "required": ["kind", "description", "clarity", "bbox"],
                       "properties": {"kind": {"type": "string", "enum": ARTIFACT_KINDS},
                                      "description": {"type": "string"},
                                      "clarity": {"type": "string", "enum": ["clear", "subtle"]},
                                      "bbox": {"type": "array", "items": {"type": "integer"}}}}}}}


def synthetic_visual_artifacts(path, src, size, case):
    """Vision model lists locatable artifacts (garbled text, hands, shadows...) with boxes; it never rates the image.
    Score = how many and how clear: clear counts 1, subtle 0.5; three clear ones reach 1.0."""
    r = llm.ask_json(ARTIFACT_PROMPT, "List the visual artifacts in this image.", [src], schema=ARTIFACT_SCHEMA,
                     name="synthetic_visual_artifacts", version=ARTIFACT_VERSION, case=case)
    if r is None:
        return []
    w, h = size
    arts = [a for a in r["artifacts"] if len(a["bbox"]) == 4][:10]
    case.log("visual_artifacts_check", file=path.name, model=llm.MODEL, artifacts=arts)
    score = float(min(1.0, sum(1.0 if a["clarity"] == "clear" else 0.5 for a in arts) / 3))
    out = []
    for a in arts:
        x1, y1, x2, y2 = (min(1000, max(0, v)) for v in a["bbox"])
        box = [round(min(x1, x2) * w / 1000), round(min(y1, y2) * h / 1000),
               max(1, round(abs(x2 - x1) * w / 1000)), max(1, round(abs(y2 - y1) * h / 1000))]
        out.append(Indicator(path.name, "image", "synthetic_visual_artifacts", score, "low",
                             f"{a['kind'].replace('_', ' ')} ({a['clarity']}): {a['description']}", location={"bbox": box},
                             alt_explanations=["Low resolution or heavy compression", "Motion blur or lens distortion",
                                               "Unusual but real scene (signage, crowd, pose)", "Model misread the image"]))
    return out
