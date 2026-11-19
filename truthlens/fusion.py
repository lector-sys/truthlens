"""OWNER D. Rule-based fusion. Weights per (indicator type, media of the evidence file) come from calibration.json
(written by `python calibrate.py`); a pair it has no data for weighs 0. Only with no calibration at all do detectors
fall back to their general reliability."""
import json
from pathlib import Path
from .indicator import RELIABILITY_WEIGHT

CALIBRATION = Path(__file__).resolve().parent.parent / "calibration.json"


def load_calibration(path=CALIBRATION):
    """The whole calibration record ({} if not run yet). Its "dataset" says SYNTHETIC or REAL."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


TYPE_WEIGHT = load_calibration().get("weights", {})

OF = " of manipulation or synthetic generation"
LEVELS = [(0.75, "Strong indicators" + OF), (0.45, "Moderate indicators" + OF), (0.15, "Weak indicators" + OF),
          (0.0, "No significant indicators" + OF)]
LIMITED = "Limited analysis: no significant indicators"
GENERATED = {"synthetic_provenance", "synthetic_visual_artifacts", "spectral_upsampling", "synthetic_bandlimit",
             "synthetic_silence", "synthetic_prosody", "lipsync_mismatch", "face_region_flicker"}
CONTEXT = {"text_inconsistency", "caption_image_mismatch", "caption_frame_mismatch", "claim_date_mismatch",
           "quote_not_in_transcript", "image_matches_frame"}


def group(i):
    """Evidence type of a finding: "Generated" (signs of synthetic origin), "Context" (caption/claims disagree with the
    media), else "Edited" (signs of local edits, cuts or re-encoding)."""
    return "Generated" if i.type in GENERATED else "Context" if i.type in CONTEXT else "Edited"


def headline(assessment, coverage, file=None):
    """Label to show: "Not analysed: <why>" when no analyzer ran, LIMITED when checks were skipped and nothing was
    found, else the fused label. `file` = one evidence file instead of the whole case."""
    cov = coverage["files"].get(file, {}) if file else coverage
    if not cov.get("analysed", 1):
        return "Not analysed: " + cov.get("why", "no analyzer ran")
    lab = (assessment["files"].get(file) or {"label": LEVELS[-1][1]})["label"] if file else assessment["case"]["label"]
    return LIMITED if lab == LEVELS[-1][1] and coverage["not_run"] else lab


UNCALIBRATED = "uncalibrated (not counted in label)"


def media(file):
    """Media of an evidence file by extension: "image" | "video" | "audio" | "text" ("context" and unknown -> "text")."""
    from .analyzers import BY_EXT  # here: analyzers import much more than fusion needs at import time
    mods = BY_EXT.get(Path(file).suffix.lower())
    return mods[0].__name__.rsplit(".", 1)[1] if mods else "text"


def key(i):
    """Calibration key: indicator type + media of its evidence file, e.g. "noise_inconsistency/video" (an audio flag in
    an .mp4 is "/video"), so rows of one media never move another media's weight."""
    return f"{i.type}/{media(i.file)}"


def calibrated(i, weights=None):
    """Does calibration have a weight for this (type, media) pair? With no calibration at all, reliability defaults."""
    w = TYPE_WEIGHT if weights is None else weights
    return not w or key(i) in w


def weight(i, weights=None):
    """Calibrated weight of the (type, media) pair; 0 for a pair calibration has no data for (shown, not counted);
    reliability default only when nothing is calibrated yet."""
    w = TYPE_WEIGHT if weights is None else weights
    return w.get(key(i), 0.0) if w else RELIABILITY_WEIGHT[i.reliability]


def contributions(indicators, weights=None):
    """{(file, type): strongest score*weight}: what the noisy-OR actually combines (the "why this label" bars)."""
    best = {}
    for i in indicators:
        k = (i.file, i.type)
        best[k] = max(best.get(k, 0.0), i.score * weight(i, weights))
    return best


def combine(indicators, weights=None):
    """Noisy-OR over the strongest indicator per (file, type): 1 - prod(1 - score*weight).
    Max per type first, so 60 frames with the same weak ELA flag count once, not 60 times."""
    p = 1.0
    for v in contributions(indicators, weights).values():
        p *= 1 - v
    return 1 - p


def label(p):
    return next(name for cut, name in LEVELS if p >= cut)


def assess(indicators, weights=None):
    files = sorted({i.file for i in indicators})
    per_file = {f: combine([i for i in indicators if i.file == f], weights) for f in files}
    case_p = combine(indicators, weights)
    return {"case": {"p": case_p, "label": label(case_p)},
            "files": {f: {"p": p, "label": label(p)} for f, p in per_file.items()}}
