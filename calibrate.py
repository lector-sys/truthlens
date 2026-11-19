"""python calibrate.py: run the labelled test set, print per-detector accuracy as markdown (goes on a slide).
Reads testset/labels.csv; media in testset/media/. Each row is analyzed as its own case, then deleted."""
import csv
from collections import Counter
from pathlib import Path
from truthlens.vault import Case
from truthlens import pipeline

ROOT = Path(__file__).resolve().parent / "testset"
POSITIVE = ("Moderate indicators", "Strong indicators")


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
