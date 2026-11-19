"""python calibrate.py [--allow-placeholder]: validate testset/labels.csv, run every row as its own case, fit per-type
fusion weights on a calibration split and report case-level hit / false-alarm rates on the held-out split, with Wilson
95% intervals. Writes calibration.json (fusion.py and the dashboard read it) and prints a markdown table for the slide.
Rows whose file starts with ph_ are synthetic placeholders: refused unless --allow-placeholder, then stamped SYNTHETIC."""
import csv, datetime, json, math, random, subprocess, sys
from collections import Counter
from pathlib import Path
from PIL import Image
from truthlens.vault import Case
from truthlens import pipeline, fusion, analyzers

ROOT = Path(__file__).resolve().parent / "testset"
HISTORY = Path(__file__).resolve().parent / "calibration" / "history"  # one dated copy per run
POSITIVE = tuple(name for _, name in fusion.LEVELS[:2])  # Strong, Moderate
FIRE = 0.25  # a type "fired" on a file when its strongest flag reaches this score; fainter flags are shown, not counted
LABELS = ("real", "manipulated")  # TASKS.md §5 column values; shown as "untouched" / "manipulated"
GENERATED = ("ai_generated", "cloned_voice", "ai_video")  # rows must name the tool in the `generator` column
MANIPULATIONS = {"none", "splice", "copy_move", "ai_generated", "cloned_voice", "ai_video", "video_cut", "voice_clone", "audio_splice",
                 "wrong_caption", "wrong_date", "wrong_quote"}
CASES = {"", "demo1", "demo2", "demo3"}
# Smoke-test-only placeholders: they trip older detectors for synthetic reasons (drawn talking cartoons: only the mouth
# moves -> temporal_spike / ela_hotspot; digitally gated cloned voices: hard gate steps -> audio_click). They exercise the
# lip-sync / synthetic-speech code paths in test_smoke.py and are never calibration data.
SMOKE_ONLY = ("ph_vid_face_real_", "ph_vid_ai_", "ph_aud_clone_")


def smoke_only(rows):
    """-> (calibration rows, excluded smoke-test-only rows)."""
    return [r for r in rows if not r["file"].startswith(SMOKE_ONLY)], [r for r in rows if r["file"].startswith(SMOKE_ONLY)]


IMAGES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".avif", ".heic", ".heif", ".jxl", ".jp2")


def media_of(name):
    mods = analyzers.BY_EXT.get(Path(name).suffix.lower())
    return mods[0].__name__.rsplit(".", 1)[1] if mods else None


def check(rows, media_dir=ROOT / "media"):
    """-> list of problems; also used to skip bad rows."""
    probs = []
    for n, r in enumerate(rows, 2):
        if not (media_dir / r["file"]).exists():
            probs.append(f"line {n}: {r['file']} not found in {media_dir}")
        if not media_of(r["file"]):
            probs.append(f"line {n}: {r['file']} has an unsupported extension")
        if r["label"] not in LABELS:
            probs.append(f"line {n}: label '{r['label']}' not in {LABELS}")
        if r["manipulation"] not in MANIPULATIONS or (r["label"] == "real") != (r["manipulation"] == "none"):
            probs.append(f"line {n}: manipulation '{r['manipulation']}' invalid for label '{r['label']}'")
        if r.get("case", "") not in CASES:
            probs.append(f"line {n}: case '{r['case']}' not in {sorted(CASES)}")
        if r["manipulation"] in GENERATED and not r.get("generator", "").strip():
            probs.append(f"line {n}: {r['manipulation']} row needs the tool name in the 'generator' column")
    return probs


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n -> (p, lo, hi). Honest at n = 15-30, unlike k/n alone."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def weight(m, r, n_manip, n_real):
    """Fusion weight from firing rates: 0.9 * (1 - FPR/TPR), Laplace-smoothed so 2 lucky hits don't earn 0.9.
    0 = fires as often on untouched as on manipulated files (no signal)."""
    tpr, fpr = (m + 1) / (n_manip + 2), (r + 1) / (n_real + 2)
    return round(0.9 * max(0.0, 1 - fpr / tpr), 2)


def split(results, frac=0.6, seed=0):
    """Stratified by (media, label), fixed seed -> (calibration, held_out)."""
    rnd, cal, held = random.Random(seed), [], []
    for key in sorted({(r["media"], r["label"]) for r in results}):
        group = [r for r in results if (r["media"], r["label"]) == key]
        rnd.shuffle(group)
        k = max(1, round(len(group) * frac)) if len(group) > 1 else 1
        cal += group[:k]
        held += group[k:]
    return cal, held


def fit(results):
    """Per (indicator type, media) pair, counted only on rows of that media: hits, false alarms, precision with CI,
    weight. Keys "type/media" match fusion.key(), so e.g. synthetic video rows can't lower an image detector's weight."""
    out = {}
    for med in sorted({r["media"] for r in results}):
        rows = [r for r in results if r["media"] == med]
        n = Counter(r["label"] for r in rows)
        fired = {lab: Counter(t for r in rows if r["label"] == lab for t in {i.type for i in r["indicators"] if i.score >= FIRE})
                 for lab in LABELS}
        for t in sorted(set(fired["manipulated"]) | set(fired["real"])):
            m, f = fired["manipulated"][t], fired["real"][t]
            out[f"{t}/{med}"] = {"hits": m, "false_alarms": f, "n_manipulated": n["manipulated"], "n_untouched": n["real"],
                                 "precision": wilson(m, m + f), "hit_rate": wilson(m, n["manipulated"]),
                                 "weight": weight(m, f, n["manipulated"], n["real"])}
    return dict(sorted(out.items()))


def evaluate(results, weights):
    """Case-level confusion matrix with these weights (Moderate/Strong = flagged)."""
    flag = [(r["label"], fusion.assess(r["indicators"], weights)["case"]["label"] in POSITIVE) for r in results]
    tp = sum(l == "manipulated" and f for l, f in flag)
    fn = sum(l == "manipulated" and not f for l, f in flag)
    fp = sum(l == "real" and f for l, f in flag)
    tn = sum(l == "real" and not f for l, f in flag)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "hit_rate": wilson(tp, tp + fn), "false_alarm_rate": wilson(fp, fp + tn)}


def pct(ci):
    return f"{ci[0]:.0%} ({ci[1]:.0%}-{ci[2]:.0%})"


def table(rec):
    s, h = rec["detectors"], rec["held_out"]
    lines = [f"**{rec['dataset']}** test set, n={rec['n']}, calibration split n={rec['n_calibration']}, "
             f"held-out n={rec['n_held_out']}, {rec['date']}, git {rec['git']}\n",
             "| Indicator / media | Fired on manipulated | Fired on untouched | Precision (95% CI) | Weight |", "|---|---|---|---|---|"]
    for t, d in s.items():
        lines.append(f"| {t} | {d['hits']}/{d['n_manipulated']} | {d['false_alarms']}/{d['n_untouched']} | "
                     f"{pct(d['precision'])} | {d['weight']} |")
    lines += ["", f"Held-out case level (Moderate/Strong = flagged): caught {h['tp']}/{h['tp'] + h['fn']} manipulated "
                  f"= {pct(h['hit_rate'])}; false alarms {h['fp']}/{h['fp'] + h['tn']} untouched = {pct(h['false_alarm_rate'])}"]
    return "\n".join(lines)


def run(argv=()):
    rows, skipped = smoke_only(list(csv.DictReader(open(ROOT / "labels.csv", encoding="utf-8"))))
    if skipped:
        print(f"excluded {len(skipped)} smoke-test-only placeholder rows (trip older detectors for synthetic reasons, "
              f"see SMOKE_ONLY): {', '.join(r['file'] for r in skipped)}")
    for p in check(rows):
        print("label problem:", p)
    bad = {r["file"] for r in rows if check([r])}
    rows = [r for r in rows if r["file"] not in bad]
    synthetic = any(r["file"].startswith("ph_") for r in rows)
    if synthetic and "--allow-placeholder" not in argv:
        sys.exit("ph_ placeholder rows present: their numbers are not detector accuracy. Remove them, "
                 "or pass --allow-placeholder to get a SYNTHETIC-stamped calibration.")
    counts = Counter((media_of(r["file"]), r["label"]) for r in rows)
    print("counts:", {f"{m}/{'untouched' if l == 'real' else l}": c for (m, l), c in sorted(counts.items())})
    for (m, l), c in sorted(counts.items()):
        if c < 15:
            print(f"warning: only {c} {m} {'untouched' if l == 'real' else l} files (goal >= 15)")
    for kind in GENERATED:
        got = [r for r in rows if r["manipulation"] == kind]
        gens = {r.get("generator", "").strip().lower() for r in got} - {""}
        if len(got) < 15 or len(gens) < 3:
            print(f"warning: {len(got)} {kind} files from {len(gens)} tool(s) (goal >= 15 from >= 3 tools)")
    ai = [r for r in rows if r["manipulation"] == "ai_generated"]
    for r in ai:
        try:
            with Image.open(ROOT / "media" / r["file"]) as im:
                if max(im.size) < 768:
                    print(f"warning: {r['file']} is {im.size[0]}x{im.size[1]}: looks like a thumbnail, use the original download")
        except OSError:
            pass  # missing (reported above) or HEIC etc. (the pipeline converts it; size not checked here)

    results = []
    for i, row in enumerate(rows):
        media = ROOT / "media" / row["file"]
        case = Case(f"calib_{i:03d}")
        try:
            case.ingest(media.name, media.read_bytes())
            res = pipeline.run(case, row.get("context", ""))
            results.append({"file": row["file"], "label": row["label"], "media": media_of(row["file"]),
                            "indicators": res["indicators"]})
            print(f"{row['file']}: {row['label']} -> {res['assessment']['case']['label']}")
        finally:
            case.delete()

    cal, held = split(results)
    det = fit(cal)
    weights = {t: d["weight"] for t, d in det.items()}
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    rec = {"dataset": "SYNTHETIC" if synthetic else "REAL", "n": len(results), "n_calibration": len(cal),
           "n_held_out": len(held), "date": datetime.date.today().isoformat(), "git": git or "?",
           "excluded_smoke_only": [r["file"] for r in skipped],
           "detectors": det, "weights": weights, "held_out": evaluate(held, weights)}
    fusion.CALIBRATION.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    hist = HISTORY / f"calibration_{datetime.datetime.now():%Y-%m-%d_%H%M%S}_{rec['git']}.json"
    hist.parent.mkdir(parents=True, exist_ok=True)
    hist.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"Copy kept in {hist.relative_to(ROOT.parent)} (commit both).")
    print("\n" + table(rec) + f"\n\nWrote {fusion.CALIBRATION.name}; fusion.py loads its weights automatically.")
    print(f"TYPE_WEIGHT = {weights}")


if __name__ == "__main__":
    run(sys.argv[1:])
