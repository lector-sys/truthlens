"""OWNER D. Rule-based fusion. Weights come from detector reliability; tune with the labelled test set."""
from .indicator import RELIABILITY_WEIGHT

# Per-type weights measured by `python calibrate.py` (paste its TYPE_WEIGHT line here). Types not listed fall back
# to RELIABILITY_WEIGHT[indicator.reliability]. Only paste numbers from the REAL test set, not the ph_ placeholders.
TYPE_WEIGHT = {}

LEVELS = [(0.75, "Strong indicators"), (0.45, "Moderate indicators"), (0.15, "Weak indicators"), (0.0, "No significant indicators")]


def weight(i):
    return TYPE_WEIGHT.get(i.type, RELIABILITY_WEIGHT[i.reliability])


def combine(indicators):
    """Noisy-OR over the strongest indicator per (file, type): 1 - prod(1 - score*weight).
    Max per type first, so 60 frames with the same weak ELA flag count once, not 60 times."""
    best = {}
    for i in indicators:
        k = (i.file, i.type)
        best[k] = max(best.get(k, 0.0), i.score * weight(i))
    p = 1.0
    for v in best.values():
        p *= 1 - v
    return 1 - p


def label(p):
    return next(name for cut, name in LEVELS if p >= cut)


def assess(indicators):
    files = sorted({i.file for i in indicators})
    per_file = {f: combine([i for i in indicators if i.file == f]) for f in files}
    case_p = combine(indicators)
    return {"case": {"p": case_p, "label": label(case_p)},
            "files": {f: {"p": p, "label": label(p)} for f, p in per_file.items()}}
