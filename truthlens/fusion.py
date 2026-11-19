"""OWNER D. Rule-based fusion. Weights per (indicator type, media of the evidence file) come from calibration.json
(written by `python calibrate.py`, committed); a pair it has no data for weighs 0. With no calibration loaded every
weight is 0 and the dashboard and report show MISSING in red: no silent fallback.

No double counting: an av_aligned_flags finding links a video flag and an audio flag that already count on their
own, so the alignment adds at most ALIGN_BONUS: one event's total is its own flags plus that small bonus."""
import json
from pathlib import Path
CALIBRATION = Path(__file__).resolve().parent.parent / "calibration.json"
MISSING = "No calibration loaded — all weights are 0, labels are not meaningful. Run calibrate.py."


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
ALIGN_BONUS = 0.10
ALIGN_RULE = (f"A/V alignment findings add at most {ALIGN_BONUS:.2f} to the label: the video and audio flags they link "
              "are already counted on their own, so one event is never counted twice.")


ROUTED = {}  # evidence file name -> "audio" for audio-only containers (.mp4 with no picture), set by pipeline routing
# ponytail: keyed by file name across cases; two different files with the same name, one audio-only, would share it.
# Pass the case if that ever happens.


def media(file):
    """Media of an evidence file: "image" | "video" | "audio" | "text" ("context" and unknown -> "text"). By extension,
    except containers routing found to be audio only (they are weighed and worded as audio)."""
    if file in ROUTED:
        return ROUTED[file]
    from .analyzers import BY_EXT  # here: analyzers import much more than fusion needs at import time
    mods = BY_EXT.get(Path(file).suffix.lower())
    return mods[0].__name__.rsplit(".", 1)[1] if mods else "text"


def key(i):
    """Calibration key: indicator type + media of its evidence file, e.g. "noise_inconsistency/video" (an audio flag in
    a video's soundtrack is "/video"; in an audio-only .mp4 "/audio"), so rows of one media never move another's weight."""
    return f"{i.type}/{media(i.file)}"


def missing():
    """True when no calibration weights are loaded: every weight is then 0 and labels mean nothing."""
    return not TYPE_WEIGHT


def calibrated(i, weights=None):
    """Does calibration have a weight for this (type, media) pair?"""
    return key(i) in (TYPE_WEIGHT if weights is None else weights)


def weight(i, weights=None):
    """Calibrated weight of the (type, media) pair; 0 for a pair without data, and for everything when nothing is
    calibrated (MISSING is shown then)."""
    return (TYPE_WEIGHT if weights is None else weights).get(key(i), 0.0)


def contribution(i, weights=None):
    """What one finding can add to the label: score*weight (an A/V alignment at most ALIGN_BONUS)."""
    v = i.score * weight(i, weights)
    return min(v, ALIGN_BONUS) if i.type == "av_aligned_flags" else v


def contributions(indicators, weights=None):
    """{(file, type): strongest contribution}: what the noisy-OR actually combines (the "why this label" bars)."""
    best = {}
    for i in indicators:
        k = (i.file, i.type)
        best[k] = max(best.get(k, 0.0), contribution(i, weights))
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
