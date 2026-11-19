"""python demo.py [--offline] [--fresh] [--no-browser]
Seeds cases demo1-demo3 from testset/labels.csv (rows with a `case`; team rows win over ph_ placeholders), then serves
the dashboard and opens it on demo2's cross-media view. One command, one process.
--offline  no API calls: Luna/transcript results come only from cache/ (run once online first to fill it)
--fresh    re-run the analysis even if the demo case already exists"""
import csv, os, subprocess, sys, threading, time, webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "http://localhost:8501/?case=demo2&screen=Cross-media"


def seed(fresh=False):
    from truthlens.vault import Case, CASES
    from truthlens import pipeline
    rows = [r for r in csv.DictReader(open(ROOT / "testset" / "labels.csv", encoding="utf-8")) if r.get("case")]
    for cid in sorted({r["case"] for r in rows}):
        mine = [r for r in rows if r["case"] == cid]
        mine = [r for r in mine if not r["file"].startswith("ph_")] or mine
        if (CASES / cid / "result.json").exists() and not fresh:
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


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--offline" in args:
        os.environ["OPENAI_API_KEY"] = ""  # llm.client() then stays None; cached responses still load
    seed("--fresh" in args)
    if "--no-browser" not in args:
        threading.Thread(target=lambda: (time.sleep(3), webbrowser.open(URL)), daemon=True).start()
    print(f"Dashboard: {URL}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
                    "--server.port", "8501"], env=os.environ)
