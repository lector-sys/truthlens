"""TruthLens forensic workbench. `python demo.py` (seeds the demo cases) or `streamlit run app.py`.
Screens: intake, overview, evidence viewer, cross-media lanes, findings, calibration, chain of custody."""
import hashlib, html, json, time
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from rapidfuzz import fuzz
from truthlens.vault import Case, CASES, sha256
from truthlens import pipeline, report, fusion, llm
from truthlens.analyzers import IMAGE_EXTS, TEXT_EXTS
from truthlens.analyzers.media_meta import readable_image

IMAGE = IMAGE_EXTS
VIDEO = (".mp4", ".mov", ".mkv", ".webm", ".avi")
AUDIO = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
LEVELS = [name for _, name in reversed(fusion.LEVELS)]  # No, Weak, Moderate, Strong
LEVEL = dict(zip(LEVELS, [("#64748B", "○", "No"), ("#A3B18A", "◔", "Weak"), ("#F2B544", "◑", "Moderate"),
                          ("#F97352", "●", "Strong")]))
GREY = ("#64748B", "–", "Not analysed")  # "Not analysed: ..." and "Limited analysis: ..." headlines
GROUPS = {"Edited": "signs of local edits, cuts or re-encoding", "Generated": "signs the content was made by a generator",
          "Context": "the caption or claims disagree with the media"}
LAYER = {"ELA": ("_ela.png", (56, 189, 248)), "Noise map": ("_noise.png", (242, 181, 68))}
SCREENS = ["Overview", "Evidence", "Cross-media", "Findings", "Calibration", "Custody"]
HIDDEN = ("smoke_test", "calib_", "probe_")
e = html.escape

st.set_page_config("TruthLens", "🔎", layout="wide")
st.markdown("""<style>
:root{--bg:#0A0E13;--panel:#10161D;--raised:#161E27;--line:#1F2A36;--text:#E6EDF3;--muted:#8B98A5;--accent:#38BDF8}
html,body,[class*="css"]{font-family:Inter,"Segoe UI",system-ui,sans-serif;font-variant-numeric:tabular-nums}
h1,h2,h3,h4{font-family:"Space Grotesk",Inter,"Segoe UI",sans-serif;letter-spacing:-.01em}
.mono,code{font-family:"JetBrains Mono",Consolas,ui-monospace,monospace!important;font-variant-numeric:tabular-nums}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:12px}
.muted{color:var(--muted)} .small{font-size:.82rem}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:600;font-size:.85rem;border:1px solid;vertical-align:middle}
.badge{display:inline-block;padding:3px 10px;border-radius:6px;font:600 .78rem "JetBrains Mono",Consolas,monospace;letter-spacing:.04em}
.card{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--c,#38BDF8);border-radius:10px;
      padding:10px 14px;margin-bottom:8px;animation:fade .25s ease-out}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px;min-height:96px}
.tile.off{opacity:.45}
.tile b{font-size:1.6rem;font-family:"JetBrains Mono",Consolas,monospace}
mark{border-radius:4px;padding:0 3px;color:#0A0E13}
.tl{border-left:2px solid var(--line);margin-left:6px;padding-left:16px}
.tl div{position:relative;margin-bottom:10px}
.tl div:before{content:"";position:absolute;left:-22px;top:6px;width:10px;height:10px;border-radius:50%;background:var(--c,#38BDF8)}
.scan{height:4px;border-radius:2px;background:linear-gradient(90deg,transparent,#38BDF8,transparent);
      background-size:40% 100%;background-repeat:no-repeat;animation:scan 1.2s linear infinite}
@keyframes scan{from{background-position:-40% 0}to{background-position:140% 0}}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>""", unsafe_allow_html=True)


def md(s):
    st.markdown(s, unsafe_allow_html=True)


def pill(label):
    c, icon, _ = LEVEL.get(label, GREY)
    return f'<span class="pill" style="color:{c};border-color:{c}">{icon} {e(label)}</span>'


def where(loc):
    if "bbox" in loc:
        x, y, w, h = loc["bbox"]
        s = f"x{x} y{y} · {w}×{h}px"
        if "bbox2" in loc:
            s += f" ↔ x{loc['bbox2'][0]} y{loc['bbox2'][1]}"
        return s + (f" @ {loc['t']:.1f}s" if "t" in loc else "")
    if "t" in loc:
        return f"{loc['t']:.2f}s" + (f" · frame {loc['frame']}" if "frame" in loc else "")
    if "start" in loc:
        return f"{loc['start']:.2f}–{loc['end']:.2f}s"
    if "span" in loc:
        return f"“{loc['span']}”"
    if "meta" in loc:
        return f"metadata: {loc['meta']}"
    return "whole file"


def _runs(mask):
    """[(start, end)] index ranges where a boolean array is True."""
    d = np.diff(np.concatenate([[0], np.asarray(mask, np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def card(i):
    c = LEVEL[fusion.label(i.score * fusion.weight(i))][0]
    rel = " ↔ " + e(", ".join(i.related)) if i.related else ""
    md(f'<div class="card" style="--c:{c}"><b>{e(i.type)}</b> <span class="muted small">· {e(i.modality)} · {e(i.file)}{rel}</span>'
       f'<span class="mono small" style="float:right">score {i.score:.2f} '
       + (f'× weight {fusion.weight(i):.2f}' if fusion.calibrated(i) else f'· {fusion.UNCALIBRATED}') + '</span><br>'
       f'<span class="mono small muted">{e(where(i.location))}</span><br>{e(i.explanation)}<br>'
       f'<span class="muted small">Could also be: {e("; ".join(i.alt_explanations))}</span><br>'
       f'<span class="small">✔ Check: {e("; ".join(report.recommended(i)))}</span></div>')


def dataset_badge(cal):
    if not cal:
        return '<span class="badge" style="background:#1F2A36;color:#8B98A5">UNCALIBRATED · reliability defaults</span>'
    measured = cal["dataset"] == "REAL"
    col = "#38BDF8" if measured else "#F2B544"
    held = f" · held-out n={cal['n_held_out']}" if measured else ""
    return f'<span class="badge" style="background:{col};color:#0A0E13">{e(cal["dataset"])} · n={cal["n"]}{held}</span>'


def pct(ci):
    return f"{ci[0]:.0%} ({ci[1]:.0%}–{ci[2]:.0%})"


def case_dirs():
    if not CASES.exists():
        return []
    ds = [d for d in CASES.iterdir() if (d / "result.json").exists() and not d.name.startswith(HIDDEN)]
    demos = sorted(d for d in ds if d.name.startswith("demo"))
    return demos + sorted((d for d in ds if d not in demos), key=lambda d: -d.stat().st_mtime)


@st.cache_data(show_spinner=False)
def load(case_id, mtime):
    return pipeline.load(Case(case_id))


def mtime(case_id):
    return (CASES / case_id / "result.json").stat().st_mtime


cal = fusion.load_calibration()
qp = st.query_params


@st.fragment(run_every=30)
def server_pill(fallback=()):
    """GPU server status, checked at startup and every 30 s: grey when offline, amber when this case's answers came
    from a fallback provider, else green."""
    if "ollama" not in llm.PROVIDERS:
        return
    up, ms = llm.ping()
    text, c = (("offline", "#64748B") if not up else (f"online · {ms} ms · fallback used: {', '.join(fallback)}", "#F2B544")
               if fallback else (f"online · {ms} ms", "#4ADE80"))
    md(f'<div style="text-align:right"><span class="pill" style="color:{c};border-color:{c}">GPU server: '
       f'{e(llm.host("ollama"))} · {e(text)}</span></div>')


# ---------- sidebar: case list ----------
with st.sidebar:
    md('<h2 style="margin:0">🔎 TruthLens</h2><div class="muted small">Multimodal forensic workbench</div>')
    md(dataset_badge(cal))
    ids = ["＋ New case"] + [d.name for d in case_dirs()]
    cur = qp.get("case", ids[1] if len(ids) > 1 else ids[0])
    pick = st.radio("Cases", ids, index=ids.index(cur) if cur in ids else 0,
                    format_func=lambda c: c if c.startswith("＋") else
                    f"{LEVEL.get(load(c, mtime(c))['headline'], GREY)[1]}  {c}")
    if pick != cur:
        qp["case"] = pick
        qp.pop("file", None)
        st.rerun()
    st.caption("Automated indicators, not proof. A person reviews every finding.")

server_pill(() if pick.startswith("＋") else tuple(llm.fallback_used(Case(pick).entries())))

# ---------- intake ----------
if pick.startswith("＋"):
    st.title("New case")
    md('<div class="muted">Evidence is hashed (SHA-256) on arrival, stored read-only, and every step is logged.</div>')
    files = st.file_uploader("Evidence: images, videos, audio, text (.txt / .md)", accept_multiple_files=True)
    context = st.text_area("Caption / claims that came with the media (optional)",
                           placeholder='e.g. Flood at the campus library on 2026-11-02. She said "we cancelled the fest".')
    pasted = st.text_area("Text evidence (optional): a message, post or article to examine as evidence",
                          placeholder="Stored as pasted_text.txt with its own hash; its claims are checked against the media.")
    if files:
        rows = []
        for f in files:
            n = f.name.lower()
            kind = ("image" if n.endswith(IMAGE) else "video" if n.endswith(VIDEO) else "audio" if n.endswith(AUDIO)
                    else "text" if n.endswith(TEXT_EXTS) else "unsupported")
            rows.append(f'<div class="mono small">{e(f.name)} <span class="badge" style="background:#161E27;color:#8B98A5">'
                        f'{kind}</span> {hashlib.sha256(f.getvalue()).hexdigest()}</div>')
        md('<div class="panel"><div class="muted small">SHA-256 of each upload</div>' + "".join(rows) + "</div>")
    b1, b2 = st.columns([1, 6])
    empty = not files and not pasted.strip()
    go = b1.button("Analyze", type="primary", disabled=empty)
    fresh = b2.button("Analyze fresh", disabled=empty, help="Ask the models again instead of reusing cached answers")
    if go or fresh:
        llm.FRESH = fresh
        case = Case()
        uploads = {f.name: f.getvalue() for f in files or []}
        if pasted.strip():
            uploads["pasted_text.txt"] = pasted.strip().encode("utf-8")
        ok = {n: sha256(case.ingest(n, b)) == hashlib.sha256(b).hexdigest() for n, b in uploads.items()}
        md("".join(f'<div class="mono small">{"✓" if v else "✗"} {e(n)}: stored copy {"matches" if v else "DOES NOT match"} '
                   f'the upload hash</div>' for n, v in ok.items()))
        md('<div class="scan"></div>')
        with st.status("Analyzing…", expanded=True) as status:
            t0 = {}

            def progress(step, state):
                if state == "running":
                    t0[step] = time.time()
                    status.update(label=f"Running {step}…")
                else:
                    st.write(f"{'✓' if state == 'done' else '⚠'} {step} · {time.time() - t0.get(step, time.time()):.1f}s")
            try:
                pipeline.run(case, context, progress)
            finally:
                llm.FRESH = False
            status.update(label="Analysis complete", state="complete")
        qp["case"] = case.id
        st.rerun()
    st.stop()

# ---------- open case ----------
case = Case(pick)
res = load(pick, mtime(pick))
inds, a, log = res["indicators"], res["assessment"], case.entries()
not_run = {x["check"]: x["reason"] for x in log if x["action"] == "check_not_run"}
llm_calls = [x for x in log if x["action"] == "llm_call"]

top_l, top_r = st.columns([2, 3])
top_l.markdown(f"## {e(case.id)} {pill(res['headline'])}", unsafe_allow_html=True)
cov_line = pipeline.coverage_line(res["coverage"])
for w in llm.server_warnings(log):
    st.warning(w, icon="⏳" if w.startswith("GPU server busy") else "📡")
want = qp.get("screen", "Overview")
screen = top_r.radio("Screen", SCREENS, horizontal=True, label_visibility="collapsed",
                     index=SCREENS.index(want) if want in SCREENS else 0)
qp["screen"] = screen


def gauge(label, p):
    k = LEVELS.index(label)
    segs = []
    for n in range(4):
        a0, a1 = np.pi - n * np.pi / 4 - 0.03, np.pi - (n + 1) * np.pi / 4 + 0.03
        x0, y0, x1, y1 = 150 + 110 * np.cos(a0), 140 - 110 * np.sin(a0), 150 + 110 * np.cos(a1), 140 - 110 * np.sin(a1)
        segs.append(f'<path d="M{x0:.1f},{y0:.1f} A110,110 0 0 1 {x1:.1f},{y1:.1f}" stroke="{LEVEL[LEVELS[n]][0]}" '
                    f'stroke-width="22" fill="none" opacity="{1 if n == k else 0.16}"/>')
    ang = np.pi - min(p, 1) * np.pi
    nx, ny = 150 + 84 * np.cos(ang), 140 - 84 * np.sin(ang)
    c, icon, short = LEVEL[label]
    return (f'<svg viewBox="0 0 300 178" width="100%" style="max-width:360px" role="img" aria-label="{e(label)}">'
            + "".join(segs) + f'<line x1="150" y1="140" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#E6EDF3" stroke-width="3" '
            f'stroke-linecap="round"/><circle cx="150" cy="140" r="6" fill="#E6EDF3"/>'
            f'<text x="150" y="172" text-anchor="middle" fill="{c}" font-size="17" font-weight="700">{icon} {short} indicators</text>'
            f'<text x="30" y="160" fill="#8B98A5" font-size="10">No</text><text x="250" y="160" fill="#8B98A5" font-size="10">Strong</text></svg>')


def bars(items, width=480):
    """[(label, value 0..1, note)] -> horizontal SVG bars coloured by the level that value alone would reach."""
    rows = []
    for n, (lab, v, note) in enumerate(items):
        y = n * 34
        rows.append(f'<text x="0" y="{y + 13}" fill="#E6EDF3" font-size="12">{e(lab)}</text>'
                    f'<rect x="250" y="{y + 2}" width="{width - 300}" height="14" rx="3" fill="#161E27"/>'
                    f'<rect x="250" y="{y + 2}" width="{max(2, v * (width - 300)):.0f}" height="14" rx="3" fill="{LEVEL[fusion.label(v)][0]}"/>'
                    f'<text x="{width - 42}" y="{y + 13}" fill="#8B98A5" font-size="12" font-family="Consolas,monospace">{e(note)}</text>')
    return f'<svg viewBox="0 0 {width} {max(34, 34 * len(items))}" width="100%">{"".join(rows)}</svg>'


def tt(i):
    return i.location.get("t", i.location.get("start"))


# ---------- overview ----------
if screen == "Overview":
    c1, c2 = st.columns([2, 3])
    with c1:
        if res["headline"].startswith("Not analysed"):
            md(f'<div class="panel" style="text-align:center;color:#8B98A5"><div style="font-size:1.3rem">– {e(res["headline"])}</div>'
               '<div style="margin-top:8px">No analyzer ran on this evidence, so no label is given.</div></div>')
        else:
            md('<div class="panel" style="text-align:center">' + gauge(a["case"]["label"], a["case"]["p"]) +
               f'<div class="mono small muted">fused strength {a["case"]["p"]:.2f}</div>'
               + (f'<div style="margin-top:6px;color:#8B98A5"><b>{e(res["headline"])}</b></div>' if res["headline"] == fusion.LIMITED else "")
               + '<div style="margin-top:8px">Automated indicators, not proof. Review each finding below.</div></div>')
        if cov_line:
            md(f'<div class="panel small" style="border-color:#64748B">{e(cov_line)}</div>')
        if cal:
            h = cal["held_out"]
            md(f'<div class="panel small">{dataset_badge(cal)}<br><span class="muted">How sure is this label? On the held-out '
               f'split this setup flagged</span> {pct(h["hit_rate"])} <span class="muted">of manipulated files and</span> '
               f'{pct(h["false_alarm_rate"])} <span class="muted">of untouched ones (95% intervals).</span></div>')
        else:
            md('<div class="panel small muted">Not calibrated yet: weights are reliability defaults. '
               'Run <code>python calibrate.py</code> on the labelled test set.</div>')
    with c2:
        contrib = [kv for kv in sorted(fusion.contributions(inds).items(), key=lambda kv: -kv[1])[:3] if kv[1] > 0]
        md('<div class="panel"><b>Why this label</b> <span class="muted small">strongest contributions · score × calibrated weight</span>'
           + (bars([(lab if len(lab := f"{t} · {f}") <= 36 else lab[:35] + "…", v, f"{v:.2f}") for (f, t), v in contrib]) if contrib
              else '<div class="muted">Nothing contributes: no indicator carries weight in this setup.</div>') + "</div>")
        mods = [("Image", "image", IMAGE), ("Video", "video", VIDEO), ("Audio", "audio", AUDIO + VIDEO),
                ("Text", "text", None), ("Cross-media", "cross", None)]
        for col, (name, m, exts) in zip(st.columns(len(mods)), mods):
            present = exts is None or any(p.suffix.lower() in exts for p in case.evidence())
            col.markdown(f'<div class="tile{"" if present else " off"}"><div class="muted small">{name}</div>'
                         f'<b>{sum(i.modality == m for i in inds)}</b><div class="muted small">'
                         f'{"flags" if present else "no such evidence"}</div></div>', unsafe_allow_html=True)
        if not_run:
            md('<div class="panel small" style="opacity:.75;margin-top:12px"><b>Semantic checks not run</b><br>' + "<br>".join(
                f'<span class="mono">{e(k)}</span>: {e(v)}' for k, v in not_run.items()) + "</div>")
        semantic = [x for x in llm_calls if x["check"] != "transcribe"]
        if semantic:
            vers = sorted({f'{x["model"]}, prompt {x.get("prompt_version", "?")}' for x in semantic})
            md(f'<div class="muted small" style="margin-top:12px">Semantic checks: {e("; ".join(vers))}'
               f'{" (from cache)" if all(x.get("cached") for x in semantic) else ""}</div>')
        if prov := llm.provenance(log):
            md(f'<div class="muted small">{e(prov)} Model answers: {e(res.get("source") or llm.source(log))}.</div>')
        bad = [f for f, ok in res["integrity"].items() if not ok]
        md('<div class="panel small" style="margin-top:12px">' + (f"✗ Integrity check FAILED: {e(', '.join(bad))}" if bad else
           f"✓ Integrity: all {len(res['integrity'])} evidence file(s) match their intake SHA-256.") + "</div>")
        md('<div class="panel"><b>Summary</b><br>' + e(report.summary(res, case)) + '<div style="margin-top:8px"><b>Recommended '
           'human checks</b><ol style="margin:4px 0 0 18px">' + "".join(f"<li>{e(c)}</li>" for c in report.human_checks(inds)[:5])
           + "</ol></div></div>")

# ---------- evidence ----------
elif screen == "Evidence":
    names = [p.name for p in case.evidence()]
    fname = st.selectbox("Evidence file", names, index=names.index(qp["file"]) if qp.get("file") in names else 0)
    qp["file"] = fname
    path = case.dir / "evidence" / fname
    ext, stem = path.suffix.lower(), path.stem
    mine = sorted((i for i in inds if i.file == fname), key=lambda i: -i.score)
    digest = next((x["sha256"] for x in log if x["action"] == "ingest" and x["file"] == fname), "?")
    md(f'{pill(fusion.headline(a, res["coverage"], fname))} <span class="mono small muted">sha256 {e(digest)}</span>')
    if conv := next((x for x in log if x["action"] == "converted" and x["file"] == fname), None):
        md(f'<div class="muted small">Shown and analysed via {e(conv["to"])} (converted with {e(conv["tool"])}): '
           'compression-based checks (ELA) see the conversion, not the original. Evidence hash above is the original file.</div>')
    if ext in IMAGE:
        c1, c2 = st.columns([1, 4])
        with c1:
            avail = [k for k, (suf, _) in LAYER.items() if case.artifact_path(stem + suf).exists()]
            layers = st.multiselect("Overlay layers", avail, default=avail[1:2] or avail[:1])
            opacity = st.slider("Opacity", 0.0, 1.0, 0.6)
            boxes = st.toggle("Flagged boxes", True)
            model_boxes = any(i.type == "synthetic_visual_artifacts" for i in mine) and st.toggle("Model-listed artifacts", True)
            cm = case.artifact_path(stem + "_copymove.png")
            show_cm = cm.exists() and st.toggle("Copy-move pairs", True)
        base = Image.open(readable_image(path, case) or path).convert("RGBA")
        out = base.copy()
        for k in layers:
            suf, rgb = LAYER[k]
            h = np.asarray(Image.open(case.artifact_path(stem + suf)).convert("L").resize(base.size), np.float32)
            h = np.clip(h / max(np.percentile(h, 99.5), 1), 0, 1)
            lay = Image.new("RGBA", base.size, rgb + (0,))
            lay.putalpha(Image.fromarray((h * 255 * opacity).astype(np.uint8)))
            out = Image.alpha_composite(out, lay)
        if boxes:
            d = ImageDraw.Draw(out)
            for i in mine:
                if i.type == "synthetic_visual_artifacts":
                    continue
                for key in ("bbox", "bbox2"):
                    if key in i.location:
                        x, y, w, hh = i.location[key]
                        d.rectangle([x, y, x + w, y + hh], outline="#F2B544", width=max(2, base.width // 300))
                        d.text((x + 4, y + 4), i.type, fill="#F2B544")
        if model_boxes:
            d = ImageDraw.Draw(out)
            for i in mine:
                if i.type == "synthetic_visual_artifacts":
                    x, y, w, hh = i.location["bbox"]
                    d.rectangle([x, y, x + w, y + hh], outline="#C4B5FD", width=max(2, base.width // 300))
                    d.text((x + 4, y + 4), i.explanation.split(":")[0], fill="#C4B5FD")
        with c2:
            v1, v2 = st.columns(2)
            v1.image(out.convert("RGB"), caption="Analysis view: overlays + flagged regions (violet: model-listed artifacts)",
                     width="stretch")
            v2.image(Image.open(cm) if show_cm else base.convert("RGB"),
                     caption="Copy-move pairs" if show_cm else "Original, as ingested", width="stretch")
    if ext in VIDEO:
        timed = sorted((i for i in mine if tt(i) is not None), key=tt)
        vc, tc = st.columns([3, 2])
        jump = tc.selectbox("Jump to flag", ["start"] + [f"{tt(i):.2f}s · {i.type}" for i in timed])
        vc.video(str(path), start_time=0 if jump == "start" else int(max(0.0, float(jump.split("s")[0]) - 0.5)))
        frames = sorted(case.artifact_path("").glob(f"{stem}_f[0-9][0-9][0-9][0-9].jpg"))
        dur = max([len(frames), 1] + [i.location.get("end", tt(i)) + 1 for i in timed])
        X = lambda t: 20 + 960 * t / dur
        marks = "".join(
            f'<g><title>{e(i.type)} {e(where(i.location))}</title><rect x="{X(tt(i)):.1f}" y="{14 if i.modality == "video" else 40}" '
            f'width="{max(3, X(i.location["end"]) - X(i.location["start"])) if "end" in i.location else 3:.1f}" height="18" rx="2" '
            f'fill="{"#38BDF8" if i.modality == "video" else "#F2B544"}"/></g>' for i in timed if i.modality in ("video", "audio"))
        fj, lj = case.artifact_path(f"{stem}_faces.json"), case.artifact_path(f"{stem}_lipsync.json")
        track = json.loads(fj.read_text(encoding="utf-8")) if fj.exists() else {"rows": [], "face_ratio": 0}
        lip = json.loads(lj.read_text(encoding="utf-8")) if lj.exists() else None
        has = np.array(["box" in r for r in track["rows"]])
        ts = [r["t"] for r in track["rows"]]
        faces = "".join(f'<rect x="{X(ts[a]):.1f}" y="66" width="{max(2, X(ts[b - 1]) - X(ts[a])):.1f}" height="14" rx="2" '
                        f'fill="#A3B18A"/>' for a, b in _runs(has)) if len(has) else ""

        def curve(vals, color):
            v = np.nan_to_num(np.asarray(vals, float))
            v = (v - v.min()) / max(np.ptp(v), 1e-9)
            return (f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="'
                    + " ".join(f"{X(t):.1f},{126 - 34 * y:.1f}" for t, y in zip(lip["t"], v)) + '"/>')
        lips = (curve(lip["mouth"], "#38BDF8") + curve(lip["audio_db"], "#F2B544")
                + f'<text x="985" y="100" fill="#8B98A5" font-size="11" text-anchor="end">best lag {lip["lag_ms"]:+d} ms · '
                  f'r {lip["r"]:.2f}</text>') if lip else ""
        ticks = "".join(f'<text x="{X(s):.0f}" y="146" fill="#8B98A5" font-size="13">{s:.0f}s</text>' for s in np.linspace(0, dur, 9))
        md(f'<div class="panel"><svg viewBox="0 0 1000 152" width="100%" role="img" aria-label="Flag timeline">'
           f'<text x="0" y="27" fill="#8B98A5" font-size="13">V</text><text x="0" y="53" fill="#8B98A5" font-size="13">A</text>'
           f'<text x="0" y="78" fill="#8B98A5" font-size="11">face</text><text x="0" y="115" fill="#8B98A5" font-size="11">lips</text>'
           f'<rect x="20" y="14" width="960" height="18" fill="#161E27"/><rect x="20" y="40" width="960" height="18" fill="#161E27"/>'
           f'<rect x="20" y="66" width="960" height="14" fill="#161E27"/><rect x="20" y="90" width="960" height="40" fill="#10161D"/>'
           f'{marks}{faces}{lips}{ticks}</svg><span class="muted small">Cyan: video flags · amber: audio flags · green: face '
           f'visible ({track["face_ratio"]:.0%} of frames) · lips: mouth motion (cyan) vs speech loudness (amber)'
           + ("" if lip else " · lip-sync not run (no face or no speech)") + ' · pick a flag to seek.</span></div>')
        if frames:
            show = [frames[k] for k in sorted(set(np.linspace(0, len(frames) - 1, 8).astype(int)))]
            with tc:
                for row in (show[:4], show[4:]):
                    for col, f in zip(st.columns(4), row):
                        t = int(f.stem[-4:]) - 1
                        im = Image.open(f).convert("RGB")
                        near = min((r for r in track["rows"] if "box" in r), key=lambda r: abs(r["t"] - t), default=None)
                        if near and abs(near["t"] - t) < 0.5:
                            x, y, w, h = near["box"]
                            ImageDraw.Draw(im).rectangle([x, y, x + w, y + h], outline="#A3B18A", width=max(2, im.width // 160))
                        col.image(im, caption=f"{t}s", width="stretch")
    if ext in TEXT_EXTS:
        md('<div class="panel mono small" style="white-space:pre-wrap">'
           + e(path.read_text(encoding="utf-8", errors="replace")[:20000]) + "</div>")
    if ext in AUDIO or ext in VIDEO:
        spec = case.artifact_path(f"{stem}_spec.png")
        aflags = sorted((i for i in mine if i.modality == "audio" and "start" in i.location), key=tt)
        if spec.exists():
            st.image(str(spec), caption="Spectrogram · low frequencies at the bottom · red: splice/click span · blue: "
                     "too-clean pauses · amber: flat-prosody sentences · cyan line: band-limit cutoff", width="stretch")
        sel = st.selectbox("Play from a flagged span (0.5 s before)", ["start"] + [f'{i.location["start"]:.2f}s · {i.type}' for i in aflags])
        wav = case.artifact_path(f"{stem}_16k.wav")
        st.audio(str(wav if wav.exists() else path), start_time=0 if sel == "start" else int(max(0, float(sel.split("s")[0]) - 0.5)))
    spec = case.artifact_path(f"{stem}_spectrum.png")
    if ext in IMAGE and spec.exists():
        with st.expander("Noise spectrum (spectral_upsampling)"):
            st.image(str(spec), caption="Log power of the high-pass residual, centre = low frequencies. Bright isolated dots "
                     "off the centre are periodic patterns; a dark outer ring is missing fine detail.", width=360)
    md("<h4>Findings in this file</h4>")
    for g, what in GROUPS.items():
        found = [i for i in mine if fusion.group(i) == g]
        if found:
            md(f'<div class="muted small" style="margin-top:8px"><b>{g}</b> · {what}</div>')
            for i in found:
                card(i)
    if not mine:
        md('<div class="panel muted">No indicators for this file. That does not show it is unaltered.</div>')
    claims = next((x for x in reversed(log) if x["action"] == "text_claims"), {})
    if res.get("context"):
        marked = e(res["context"])
        colors = {"when": "#38BDF8", "where": "#A3B18A", "who": "#C4B5FD", "quotes": "#F2B544"}
        for k, c in colors.items():
            for v in claims.get(k, []):
                if v and e(str(v)) in marked:
                    marked = marked.replace(e(str(v)), f'<mark style="background:{c}" title="{k}">{e(str(v))}</mark>', 1)
        md('<h4>Context text</h4><div class="panel">' + marked + '<div class="muted small" style="margin-top:6px">claims: '
           + " ".join(f'<mark style="background:{c}">{k}</mark>' for k, c in colors.items()) + "</div></div>")
    tp = case.artifact_path(f"{stem}_transcript.json")
    if tp.exists():
        said = " ".join(s["text"] for s in json.loads(tp.read_text(encoding="utf-8")))
        found = [(q, fuzz.partial_ratio(str(q).lower(), said.lower())) for q in claims.get("quotes", [])]
        md('<h4>Transcript</h4><div class="panel mono small">' + e(said) + "</div>" + "".join(
            f'<div class="small">{"✓ heard" if r >= 70 else "✗ not found"} “{e(str(q))}” <span class="muted mono">match {r:.0f}/100</span></div>'
            for q, r in found))

# ---------- cross-media lanes ----------
elif screen == "Cross-media":
    md('<div class="muted">One lane per source on a shared time axis. Curves join flags that a cross-media check found to '
       'line up (video and audio within 0.5 s) or to contradict each other (claim vs metadata or transcript).</div>')
    lanes = {"Video": 30, "Audio": 90, "Text / claims": 150, "Metadata": 210}
    timed = [i for i in inds if tt(i) is not None and i.modality in ("video", "audio")]
    dur = max([1.0] + [tt(i) + 1 for i in timed])
    X = lambda t: 200 + 780 * t / dur
    els = [f'<text x="0" y="{y + 4}" fill="#8B98A5" font-size="14">{n}</text><line x1="200" y1="{y}" x2="980" y2="{y}" '
           f'stroke="#1F2A36" stroke-width="18" stroke-linecap="round"/>' for n, y in lanes.items()]
    els.append('<text x="125" y="250" fill="#8B98A5" text-anchor="middle" font-size="12">untimed</text>')
    els += [f'<text x="{X(s):.0f}" y="250" fill="#8B98A5" font-size="12">{s:.0f}s</text>' for s in np.linspace(0, dur, 7)]
    for i in timed:
        y, c = (lanes["Video"], "#38BDF8") if i.modality == "video" else (lanes["Audio"], "#F2B544")
        els.append(f'<g><title>{e(i.file)}: {e(i.type)} {e(where(i.location))}</title><circle cx="{X(tt(i)):.1f}" cy="{y}" r="6" fill="{c}"/></g>')
    for n, i in enumerate(i for i in inds if "meta" in i.location):
        els.append(f'<g><title>{e(i.file)}: {e(i.type)}</title><rect x="{205 + 12 * n}" y="{lanes["Metadata"] - 5}" width="8" height="10" fill="#64748B"/></g>')
    cross = [i for i in inds if i.modality == "cross"]
    for n, i in enumerate(cross):
        if i.type == "av_aligned_flags":
            x = X(i.location["t"])
            x2 = X(min((tt(j) for j in timed if j.modality == "audio" and j.file in i.related),
                       key=lambda t: abs(t - i.location["t"]), default=i.location["t"]))
            els.append(f'<g><title>{e(i.explanation)}</title><path d="M{x:.1f},{lanes["Video"]} C{x:.1f},60 {x2:.1f},60 {x2:.1f},{lanes["Audio"]}" '
                       f'stroke="#F97352" stroke-width="2.5" fill="none"/></g>')
        else:
            y1, y2 = lanes["Text / claims"], lanes[{"claim_date_mismatch": "Metadata", "quote_not_in_transcript": "Audio"}.get(i.type, "Video")]
            x = 150 + 14 * (n % 4)
            els.append(f'<g><title>{e(i.explanation)}</title><path d="M{x},{y1} C{x - 50},{y1} {x - 50},{y2} {x},{y2}" stroke="#F97352" '
                       f'stroke-width="2.5" fill="none"/><circle cx="{x}" cy="{y1}" r="5" fill="#F97352"/><circle cx="{x}" cy="{y2}" r="5" fill="#F97352"/></g>')
    md(f'<div class="panel"><svg viewBox="0 0 1000 262" width="100%" role="img" aria-label="Cross-media lanes">{"".join(els)}</svg>'
       '<span class="muted small">Cyan: video flags · amber: audio flags · grey: metadata flags · coral curves: cross-media checks '
       '(hover for details)</span></div>')
    for i in cross:
        card(i)
    if not cross:
        md('<div class="panel muted">No cross-media links or contradictions were found'
           + (" (some semantic checks were not run: see Overview)." if not_run else ".") + "</div>")

# ---------- findings ----------
elif screen == "Findings":
    rows = [{"evidence type": fusion.group(i), "indicator": i.type, "medium": i.modality, "file": i.file, "location": where(i.location),
             "score": round(i.score, 2), "weight": round(fusion.weight(i), 2), "contribution": round(i.score * fusion.weight(i), 3),
             "counted": "yes" if fusion.calibrated(i) else fusion.UNCALIBRATED,
             "reliability": i.reliability, "why": i.explanation, "could also be": "; ".join(i.alt_explanations),
             "artifact": Path(i.artifact).name if i.artifact else ""} for i in inds]
    if rows:
        for g, what in GROUPS.items():
            mine = sorted((r for r in rows if r["evidence type"] == g), key=lambda r: -r["contribution"])
            if mine:
                md(f"<h4>{g} <span class='muted small'>· {what}</span></h4>")
                st.dataframe(mine, width="stretch", hide_index=True, height=min(440, 38 + 35 * len(mine)))
        files = sorted({i.file for i in inds if i.file != "context"})
        if files:
            c1, c2 = st.columns([3, 1])
            go = c1.selectbox("Jump to evidence", files)
            if c2.button("Open in evidence viewer"):
                qp["file"], qp["screen"] = go, "Evidence"
                st.rerun()
    else:
        md('<div class="panel muted">No indicators were raised for this case.</div>')
    checks = report.human_checks(inds)
    if checks:
        md("<h4>Recommended human checks</h4>" + "".join(f"<div>{n}. {e(c)}</div>" for n, c in enumerate(checks, 1)))

# ---------- calibration ----------
elif screen == "Calibration":
    if not cal:
        md('<div class="panel muted">No calibration yet. Run <code>python calibrate.py</code> on the labelled test set; '
           "until then fusion uses each detector's general reliability.</div>")
    else:
        md(f'<div style="font-size:1.3rem">{dataset_badge(cal)}</div><div class="muted small mono">{e(cal["date"])} · git {e(cal["git"])} · '
           f'fitted on n={cal["n_calibration"]}, evaluated on held-out n={cal["n_held_out"]}</div>')
        if cal["dataset"] != "REAL":
            md('<div class="panel small" style="border-color:#F2B544;margin-top:8px">These numbers come from SYNTHETIC placeholder '
               'media. They show the system runs end to end; they are not detector accuracy.</div>')
        det = cal["detectors"]
        rows = []
        for n, (t, d) in enumerate(det.items()):
            p, lo, hi = d["precision"]
            y = n * 30 + 12
            X = lambda v: 220 + v * 580
            rows.append(f'<text x="0" y="{y + 4}" fill="#E6EDF3" font-size="12">{e(t)}</text>'
                        f'<line x1="{X(lo):.0f}" y1="{y}" x2="{X(hi):.0f}" y2="{y}" stroke="#38BDF8" stroke-width="2"/>'
                        f'<line x1="{X(lo):.0f}" y1="{y - 5}" x2="{X(lo):.0f}" y2="{y + 5}" stroke="#38BDF8" stroke-width="2"/>'
                        f'<line x1="{X(hi):.0f}" y1="{y - 5}" x2="{X(hi):.0f}" y2="{y + 5}" stroke="#38BDF8" stroke-width="2"/>'
                        f'<circle cx="{X(p):.0f}" cy="{y}" r="5" fill="#E6EDF3"/>'
                        f'<text x="820" y="{y + 4}" fill="#8B98A5" font-size="11" font-family="Consolas,monospace">'
                        f'{d["hits"]}/{d["n_manipulated"]} vs {d["false_alarms"]}/{d["n_untouched"]} · w {d["weight"]}</text>')
        axis = "".join(f'<text x="{220 + v * 580:.0f}" y="{len(det) * 30 + 24}" fill="#8B98A5" font-size="10" text-anchor="middle">{v:.0%}</text>'
                       for v in (0, .25, .5, .75, 1))
        md('<div class="panel" style="margin-top:12px"><b>Precision per detector</b> <span class="muted small">dot = measured · '
           'bar = 95% Wilson interval · right: fired on manipulated vs untouched · fusion weight</span>'
           f'<svg viewBox="0 0 1000 {len(det) * 30 + 30}" width="100%">{"".join(rows)}{axis}</svg></div>')
        h = cal["held_out"]
        td = 'style="padding:6px 14px"'
        md(f'''<div class="panel"><b>Held-out confusion matrix</b> <span class="muted small">Moderate/Strong = flagged</span>
<table style="margin-top:8px;border-collapse:collapse" class="mono"><tr><td></td><td class="muted" {td}>flagged</td>
<td class="muted" {td}>not flagged</td></tr><tr><td class="muted" {td}>manipulated</td><td {td}>{h["tp"]}</td><td {td}>{h["fn"]}</td></tr>
<tr><td class="muted" {td}>untouched</td><td {td}>{h["fp"]}</td><td {td}>{h["tn"]}</td></tr></table>
<div class="small" style="margin-top:6px">Hit rate {pct(h["hit_rate"])} · false-alarm rate {pct(h["false_alarm_rate"])}</div></div>''')
    md('''<div class="panel small"><b>Limitations</b><ul>
<li>The test set is small and team-made, so every rate has a wide interval: read the bars, not the dots.</li>
<li>Platform recompression and resizing can both create and erase these indicators.</li>
<li>Fully generated media without provenance metadata or visible artifacts may show no indicators at all.</li>
<li>No indicators does not mean unaltered. The language model reads text, compares captions with pictures and lists visible
artifacts with their location; it is never asked whether media is real or generated.</li></ul></div>''')

# ---------- custody ----------
elif screen == "Custody":
    c1, c2 = st.columns([3, 2])
    with c1:
        md("<h4>Chain of custody</h4>")
        colors = {"ingest": "#38BDF8", "analyzed": "#A3B18A", "llm_call": "#C4B5FD", "check_not_run": "#64748B",
                  "assessed": "#F2B544", "analyzer_error": "#F97352", "verify": "#38BDF8", "report_export": "#38BDF8"}
        items = []
        for x in log:
            txt = ", ".join(f"{k}={v}" for k, v in x.items() if k not in ("ts", "action"))[:220]
            items.append(f'<div style="--c:{colors.get(x["action"], "#1F2A36")}"><span class="mono small muted">'
                         f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x["ts"]))}</span> <b>{e(x["action"])}</b>'
                         f'<br><span class="mono small muted">{e(txt)}</span></div>')
        md('<div class="tl">' + "".join(items) + "</div>")
    with c2:
        md("<h4>Integrity & export</h4>")
        if st.button("Verify integrity", type="primary"):
            v = case.verify()
            case.log("verify", **{k: ("ok" if ok else "MISMATCH") for k, ok in v.items()})
            md("".join(f'<div class="mono small">{"✓" if ok else "✗"} {e(k)} · {sha256(case.dir / "evidence" / k)[:16]}…</div>'
                       for k, ok in v.items()))
        if st.download_button("Export report (HTML)", report.render(case, res), f"{case.id}.html", "text/html"):
            case.log("report_export", format="html")
        st.caption("For PDF: open the HTML report and print → Save as PDF (the report's print view is light).")
