"""OWNER D. Rule-based fusion. Per-type weights come from calibration.json (written by `python calibrate.py`);
types it hasn't measured fall back to the detector's general reliability."""
import json
from pathlib import Path
from .indicator import RELIABILITY_WEIGHT

CALIBRATION = Path(__file__).resolve().parent.parent / "calibration.json"


def load_calibration(path=CALIBRATION):
    """The whole calibration record ({} if not run yet). Its "dataset" says SYNTHETIC or REAL."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


TYPE_WEIGHT = load_calibration().get("weights", {})

LEVELS = [(0.75, "Strong indicators"), (0.45, "Moderate indicators"), (0.15, "Weak indicators"), (0.0, "No significant indicators")]


def weight(i, weights=None):
    return (TYPE_WEIGHT if weights is None else weights).get(i.type, RELIABILITY_WEIGHT[i.reliability])


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
