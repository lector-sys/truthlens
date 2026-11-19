"""python calibrate.py: run the labelled test set, print per-detector accuracy as markdown (goes on a slide).
Reads testset/labels.csv; media in testset/media/. Each row is analyzed as its own case, then deleted."""
import csv
from collections import Counter
from pathlib import Path
from truthlens.vault import Case
from truthlens import pipeline

ROOT = Path(__file__).resolve().parent / "testset"
POSITIVE = ("Moderate indicators", "Strong indicators")


def weight(m, r, n_manip, n_real):
    """Fusion weight from firing rates: 0.9 * (1 - FPR/TPR), Laplace-smoothed so 2 lucky hits don't earn 0.9.
    0 = fires as often on real as on manipulated (no signal)."""
    tpr, fpr = (m + 1) / (n_manip + 2), (r + 1) / (n_real + 2)
    return round(0.9 * max(0.0, 1 - fpr / tpr), 2)


def summarize(results):
    """results: [{"label": "real"|"manipulated", "types": set[str], "verdict": str}] -> markdown."""
    n = Counter(r["label"] for r in results)
    fired = {lab: Counter(t for r in results if r["label"] == lab for t in r["types"]) for lab in ("manipulated", "real")}
    lines = [f"Test set: {n['manipulated']} manipulated, {n['real']} real\n",
             "| Indicator | Fired on manipulated | Fired on real | Precision |", "|---|---|---|---|"]
    for t in sorted(set(fired["manipulated"]) | set(fired["real"])):
        m, r = fired["manipulated"][t], fired["real"][t]
        lines.append(f"| {t} | {m}/{n['manipulated']} | {r}/{n['real']} | {m / (m + r):.0%} |")
    tp = sum(r["label"] == "manipulated" and r["verdict"] in POSITIVE for r in results)
    fp = sum(r["label"] == "real" and r["verdict"] in POSITIVE for r in results)
    lines += ["", f"Case level (Moderate/Strong = flagged): caught {tp}/{n['manipulated']} manipulated, "
                  f"false alarms {fp}/{n['real']} real"]
    w = {t: weight(fired["manipulated"][t], fired["real"][t], n["manipulated"], n["real"])
         for t in sorted(set(fired["manipulated"]) | set(fired["real"]))}
    lines += ["", "Paste into truthlens/fusion.py:", f"TYPE_WEIGHT = {w}"]
    return "\n".join(lines)


def run():
    results = []
    for i, row in enumerate(csv.DictReader(open(ROOT / "labels.csv", encoding="utf-8"))):
        media = ROOT / "media" / row["file"]
        if not media.exists():
            print(f"missing: {media}")
            continue
        case = Case(f"calib_{i:03d}")
        try:
            case.ingest(media.name, media.read_bytes())
            res = pipeline.run(case, row.get("context", ""))
            results.append({"label": row["label"], "types": {x.type for x in res["indicators"]},
                            "verdict": res["assessment"]["case"]["label"]})
            print(f"{row['file']}: {row['label']} -> {results[-1]['verdict']}")
        finally:
            case.delete()
    print("\n" + summarize(results))


if __name__ == "__main__":
    run()
