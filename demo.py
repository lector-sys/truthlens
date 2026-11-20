"""python demo.py [--offline] [--fresh] [--no-browser] [--prewarm]
Seeds cases demo1-demo3 from testset/labels.csv (rows with a `case`; team rows win over ph_ placeholders), then serves
the dashboard and opens it on demo2's cross-media view. One command, one process.
--offline  no API calls: Luna/transcript results come only from cache/ (run once online first to fill it)
--fresh    re-run the analysis even if the demo case already exists, asking the models again (cache bypassed)
--prewarm  run the demo cases and every testset file so cache/ holds every model answer (only missing ones are asked
           for, all of them with --fresh), list cases that still have a check not run, then exit"""
import csv, os, subprocess, sys, threading, time, webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "http://localhost:8501/?case=demo2&screen=Cross-media"


def seed(rerun=False):
    from truthlens.vault import Case, CASES
    from truthlens import pipeline
    rows = [r for r in csv.DictReader(open(ROOT / "testset" / "labels.csv", encoding="utf-8")) if r.get("case")]
    for cid in sorted({r["case"] for r in rows}):
        mine = [r for r in rows if r["case"] == cid]
        mine = [r for r in mine if not r["file"].startswith("ph_")] or mine
        if (CASES / cid / "result.json").exists() and not rerun:
            print(f"{cid}: kept (use --fresh to re-run)")
            continue
        if (CASES / cid).exists():
            Case(cid).delete()
        case = Case(cid)
        for r in mine:
            media = ROOT / "testset" / "media" / r["file"]
            if media.exists():
                case.ingest(media.name, media.read_bytes())
            else:
                print(f"{cid}: missing {media}")
        res = pipeline.run(case, mine[0]["context"])
        print(f"{cid}: {', '.join(r['file'] for r in mine)} -> {res['assessment']['case']['label']}")


def prewarm():
    """Demo cases + every labelled testset file through the pipeline; the cache keeps each good reply."""
    from truthlens.vault import Case
    from truthlens import pipeline
    seed(rerun=True)
    missing = {cid: _not_run(Case(cid)) for cid in ("demo1", "demo2", "demo3")}
    for r in csv.DictReader(open(ROOT / "testset" / "labels.csv", encoding="utf-8")):
        media = ROOT / "testset" / "media" / r["file"]
        if not media.exists():
            missing[r["file"]] = ["file missing"]
            continue
        case = Case(f"probe_{media.stem}")
        try:
            case.ingest(media.name, media.read_bytes())
            pipeline.run(case, r["context"])
            missing[r["file"]] = _not_run(case)
        finally:
            case.delete()
    bad = {k: v for k, v in missing.items() if v}
    print(f"prewarm: {len(missing) - len(bad)}/{len(missing)} cases fully cached")
    for k, v in bad.items():
        print(f"  not fully cached: {k}: {', '.join(v)}")


def _not_run(case):
    return sorted({e["check"] for e in case.entries() if e["action"] == "check_not_run"})


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--offline" in args:  # no server URLs or keys: every model answer comes from cache/
        os.environ.update(OLLAMA_URL="", WHISPER_URL="", OPENAI_API_KEY="", GROQ_API_KEY="")
    from truthlens import llm
    llm.FRESH = "--fresh" in args
    if "--prewarm" in args:
        sys.exit(prewarm())
    seed("--fresh" in args)
    if "--no-browser" not in args:
        threading.Thread(target=lambda: (time.sleep(3), webbrowser.open(URL)), daemon=True).start()
    print(f"Dashboard: {URL}")
    public = os.getenv("TRUTHLENS_PUBLIC") == "1"  # public: 50 MB uploads (config.toml) and no error details on screen
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
                    "--server.port", "8501", *(["--client.showErrorDetails", "none"] if public else ["--server.maxUploadSize", "200"])],
                   env=os.environ)
