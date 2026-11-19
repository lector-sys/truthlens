"""TruthLens forensic workbench. `python demo.py` (seeds the demo cases) or `streamlit run app.py`.
One flow: Upload -> Analyzing -> Results (overview, then Evidence / Cross-media / All findings / Custody / Calibration).
Theme "Evidence Light Table": ui/theme.css + .streamlit/config.toml, fonts bundled in static/fonts. Display only:
nothing here changes an analysis result, weight or label."""
import base64, hashlib, html, io, json, logging, threading, time, traceback
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rapidfuzz import fuzz
from truthlens.vault import Case, CASES, sha256
from truthlens import pipeline, report, fusion, llm, learned
from calibrate import MIN_OWN
from truthlens.analyzers import IMAGE_EXTS, TEXT_EXTS
from truthlens.analyzers.media_meta import readable_image

ROOT = Path(__file__).resolve().parent
IMAGE = IMAGE_EXTS
VIDEO = (".mp4", ".mov", ".mkv", ".webm", ".avi")
AUDIO = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
LEVELS = [name for _, name in reversed(fusion.LEVELS)]  # No, Weak, Moderate, Strong
LEVEL = dict(zip(LEVELS, [("#64748B", "○", "No"), ("#A3B18A", "◔", "Weak"), ("#F2B544", "◑", "Moderate"),
                          ("#F97352", "●", "Strong")]))
GREY = ("#64748B", "–", "Not analysed")
AMBER = "#F2B544"
GROUP_C = {"Edited": "#7AA2F7", "Generated": "#BB9AF7", "Context": "#E0AF68"}
GROUPS = {"Edited": "signs of local edits, cuts or re-encoding", "Generated": "signs the content was made by a generator",
          "Context": "the caption or claims disagree with the media"}
ICON = {"image": "🖼️", "video": "🎞️", "audio": "🎙️", "text": "📝", "cross": "🔗"}
LAYER = {"ELA": ("_ela.png", (79, 209, 197)), "Noise map": ("_noise.png", (242, 181, 68)),
         "Trained detector": ("_learned.png", (187, 154, 247))}
TABS = ["Evidence", "Cross-media", "All findings", "Custody", "Calibration"]
HIDDEN = ("smoke_", "calib_", "probe_", "diag_", "pv_", "tmp_")
TERMS = {  # indicator type -> (plain name, what it means); used for tooltips and the sidebar glossary
    "exif_missing": ("missing EXIF", "No camera metadata (EXIF) in the file. Messengers and social sites usually strip it."),
    "exif_edit_software": ("editor in EXIF", "The camera metadata names editing software that saved the file."),
    "ela_hotspot": ("ELA hotspot", "ELA: re-saves the image and looks for regions that compress differently, a sign they came from elsewhere."),
    "noise_inconsistency": ("noise mismatch", "Sensor noise is uneven across the picture; pasted-in content often carries different noise."),
    "copy_move": ("copy-move", "The same patch appears twice, as when an area is cloned to hide or duplicate something."),
    "reencoded": ("re-encoded", "The file was written by a converter (e.g. FFmpeg) after capture."),
    "no_creation_time": ("no capture time", "The container has no capture timestamp."),
    "temporal_spike": ("temporal spike", "A sudden picture change between frames that is not a scene cut, as at an edit point."),
    "face_texture_jump": ("face texture jump", "The texture of a face changes abruptly between frames."),
    "synthetic_provenance": ("provenance metadata", "Metadata or signed Content Credentials (C2PA) that name an AI tool or declare AI generation."),
    "spectral_upsampling": ("upsampling trace", "Regular patterns in the fine-detail spectrum, as upscaling layers in image generators leave."),
    "synthetic_visual_artifacts": ("visual artifacts", "Located defects the vision model listed, such as garbled text or warped hands."),
    "face_region_flicker": ("face flicker", "Brightness or texture of the face jumps between frames while the rest of the picture stays steady."),
    "audio_splice": ("audio splice", "The background noise level changes abruptly, as when two recordings are joined."),
    "audio_click": ("audio click", "A sample-level jump in a quiet passage, typical of a hard cut."),
    "uniform_pauses": ("uniform pauses", "Pauses between words are unusually regular."),
    "synthetic_bandlimit": ("band limit", "The audio stops sharply above one frequency, as many speech synthesizers do."),
    "synthetic_silence": ("too-clean pauses", "Pauses are digitally silent or have no breaths, unlike a microphone in a room."),
    "synthetic_prosody": ("flat prosody", "Pitch and loudness barely change within sentences."),
    "text_inconsistency": ("contradicting text", "Statements in the text contradict each other."),
    "av_aligned_flags": ("A/V alignment", "A video flag and an audio flag at the same moment (within 0.5 s)."),
    "caption_image_mismatch": ("caption vs image", "The vision model found the caption does not match what the image shows."),
    "caption_frame_mismatch": ("caption vs frames", "The vision model found the caption does not match the video's keyframes."),
    "claim_date_mismatch": ("date vs metadata", "A date in the caption disagrees with the file's capture date."),
    "quote_not_in_transcript": ("quote not heard", "A quote in the caption is not in the transcript of the recording."),
    "image_matches_frame": ("image = frame", "An image matches a frame of the video."),
    "lipsync_mismatch": ("lip-sync", "Mouth movement and speech loudness are offset or do not follow each other."),
    **{f"learned_synthetic_{m}": (f"trained detector ({m})", f"A neural network trained to tell AI-generated {m} from "
                                  "camera or microphone recordings scored this file at or above its threshold. Not counted in the label "
                                  "until calibrated on our own media; model card on the Calibration page.")
       for m in ("image", "video", "audio")},
}
CHECK_NAMES = {"face_region_flicker": "face flicker", "lipsync": "lip-sync", "claims from the caption": "claims",
               "caption ↔ image / frames": "caption vs image", "visual artifacts (model)": "visual artifacts",
               "transcript": "transcript", "claimed date ↔ metadata": "date vs metadata", "exif": "EXIF"}
STEP_NAMES = {"trained detector": "Trained detector", "image": "Image checks", "video": "Video checks", "audio": "Audio checks", "text": "Claims from the caption",
              "cross-media checks": "Cross-media checks", "summary": "Summary of the findings"}
HINTS = [  # reason fragment -> what to do next
    ("detector server", "The trained-detector server didn't answer, so the trained detectors were skipped. Check it, then Analyze fresh."),
    ("detectors_url", "No trained-detector server is configured. Set DETECTORS_URL in .env, then Analyze fresh."),
    ("no face", "No faces found, so lip-sync and face checks couldn't run. Try a clip where the speaker's face is visible."),
    ("no caption", "No caption was given. Add the caption or claims that came with the media to check them against it."),
    ("timed out", "A check took too long and the watchdog stopped it; the rest of the case finished. Try again, or a shorter clip."),
    ("no audio", "No audio track was found, so the audio checks couldn't run."),
    ("no video stream", "The file has no picture track, so it was analysed as audio."),
    ("not configured", "The model server isn't configured, so model checks were skipped. Set it up in .env, then Analyze fresh."),
    ("unreachable", "The model server didn't answer, so model checks were skipped. Check the server, then Analyze fresh."),
]
e = html.escape

st.set_page_config("TruthLens", "🔎", layout="wide")
st.markdown("<style>" + (ROOT / "ui" / "theme.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
if fusion.missing():
    st.error(fusion.MISSING, icon="⛔")
qp, ss = st.query_params, st.session_state
cal = fusion.load_calibration()


# ---------- small helpers ----------
def md(s):
    st.markdown(s, unsafe_allow_html=True)


def level(headline):
    """(colour, icon, short word) for a headline: the four levels, else grey (not analysed / limited / failed)."""
    if headline.startswith("Failed"):
        return ("#64748B", "–", "failed")
    if headline.startswith("Limited"):
        return ("#64748B", "–", "Limited")
    return LEVEL.get(headline, GREY)


def pill(headline, short=False):
    c, icon, word = level(headline)
    return f'<span class="pill" style="--c:{c}"><i>{icon}</i>{e(word if short else headline)}</span>'


def term(t):
    return TERMS.get(t, (t.replace("_", " "), t))


where = report.where


def at(loc):
    """Short "where" for a sentence: "at 0.5 s", "in region x120 y40", "in the metadata"."""
    t = loc.get("t", loc.get("start"))
    if t is not None:
        return f"at {t:.1f} s"
    if "bbox" in loc:
        return f"in region x{loc['bbox'][0]} y{loc['bbox'][1]}"
    return "in the metadata" if "meta" in loc else "in the text" if "span" in loc else "in the file"


def tt(i):
    return i.location.get("t", i.location.get("start"))


def _runs(mask):
    """[(start, end)] index ranges where a boolean array is True."""
    d = np.diff(np.concatenate([[0], np.asarray(mask, np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def img_uri(im, width=1400, q=85):
    """PIL image -> JPEG data URI, at most `width` px wide (display copy; evidence files are never touched)."""
    im = im.convert("RGB")
    if im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def light_table(inner, caption=""):
    md(f'<div class="lt">{inner}' + (f'<div class="cap small muted">{caption}</div>' if caption else "") + "</div>")


def safe(fn, what):
    """Render a panel; on an unexpected error show a friendly note and the details folded away, never a raw traceback."""
    try:
        fn()
    except Exception:
        logging.getLogger("truthlens.app").exception("panel failed: %s", what)  # full traceback in the dashboard log
        st.info(f"This {what} couldn't be shown. The analysis itself is unaffected.", icon="ℹ️")
        with st.expander("Details"):
            st.code(traceback.format_exc()[-3000:])


def steps(k):
    names = ["Upload", "Analyzing", "Results"]
    md('<div class="steps" role="list" aria-label="Progress">' + "".join(
        f'<div role="listitem" class="{"on" if n == k else "done" if n < k else ""}"><b>{"✓" if n < k else n + 1}</b>{s}</div>'
        for n, s in enumerate(names)) + "</div>")


def hint_for(reason):
    return next((h for k, h in HINTS if k in reason.lower()), None)


def hints(reasons):
    """Distinct next-step hints for a set of not-run reasons."""
    return list(dict.fromkeys(h for h in map(hint_for, reasons) if h))


def short_reason(r):
    r = r.lower()
    for k, s in (("no face", "no face visible"), ("no caption", "no caption given"), ("timed out", "timed out"),
                 ("no audio", "no audio track"), ("not configured", "model server not set up"),
                 ("unreachable", "model server offline")):
        if k in r:
            return s
    return r.split(" (")[0][:60]


def check_name(c):
    if " analysis · " in c:
        kind, f = c.split(" analysis · ", 1)
        return f"{kind} analysis of {f}"
    return CHECK_NAMES.get(c, c.replace("_", " "))


def coverage_html(cov):
    if not cov["not_run"]:
        return f"Checks run: {len(cov['ran'])} of {cov['total']} · all checks ran"
    return (f"Checks run: {len(cov['ran'])} of {cov['total']} · Not run: "
            + ", ".join(f"{check_name(c)} ({short_reason(w)})" for c, w in cov["not_run"].items()))


# ---------- components ----------
def tag_card(t, i, open_details=False):
    """A finding as an evidence tag: punched hole, tag number, media icon, plain name (tooltip), group chip, score."""
    w, cal_ok = fusion.weight(i), fusion.calibrated(i)
    c = LEVEL[fusion.label(fusion.contribution(i))][0] if cal_ok else "#64748B"
    name, tip = term(i.type)
    g = fusion.group(i)
    rel = " ↔ " + e(", ".join(i.related)) if i.related else ""
    score = (f"{i.score:.2f} × w {w:.2f} = {fusion.contribution(i):.2f}" if cal_ok else f"{i.score:.2f}" + (f" ≥ threshold {i.location['threshold']:.2f}" if "threshold" in i.location else "")
             + f" · {fusion.UNCALIBRATED}")
    md(f'<div class="tag{"" if cal_ok else " uncal"}" style="--c:{c}"><div class="head">'
       f'<span class="tagno">{t}</span><span aria-hidden="true">{ICON.get(i.modality, "")}</span>'
       f'<span class="name" title="{e(tip)}">{e(name)}</span><span class="chip" style="--c:{GROUP_C[g]}">{g}</span>'
       f'<span class="score">{e(score)}</span></div>'
       f'<div class="mono tiny muted">{e(i.file)}{rel} · {e(where(i.location))}</div><div class="why">{e(i.explanation)}</div>'
       f'<details{" open" if open_details else ""}><summary>Could also be · what to check</summary>'
       f'<div class="small muted">Could also be: {e("; ".join(i.alt_explanations))}</div>'
       f'<div class="small">✔ {e("; ".join(report.recommended(i)))}</div></details></div>')


def film_strip(frames, hits, focus_t=None, n=8):
    """Keyframes as a film strip: timestamps under each, flagged frames outlined in their indicator colour."""
    show = [frames[k] for k in sorted(set(np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)))]
    cells = []
    for f in show:
        t = int(f.stem[-4:]) - 1
        c = hits.get(t) or (AMBER if focus_t is not None and abs(focus_t - t) < 0.5 else None)
        cells.append(f'<figure class="{"hit" if c else ""}" style="--c:{c or "transparent"}"><img src="{img_uri(Image.open(f), 240, 75)}" '
                     f'alt="Frame at {t} s"><figcaption>{t}s{" ⚑" if c else ""}</figcaption></figure>')
    return '<div class="film" role="list" aria-label="Keyframes">' + "".join(cells) + "</div>"


def custody_stamp(case, log, integrity):
    ingests = [x for x in log if x["action"] == "ingest"]
    t0 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ingests[0]["ts"])) if ingests else "–"
    ok = bool(integrity) and all(integrity.values())
    hashes = "".join(f'<div class="v h" title="{e(x["sha256"])}">{e(x["file"][:40])} · sha256 {x["sha256"][:12]}…{x["sha256"][-4:]}</div>'
                     for x in ingests)
    md(f'<div class="stamp" role="group" aria-label="Custody stamp"><div class="k">case</div><div class="v">{e(case.id)}</div>'
       f'<div class="k" style="margin-top:6px">evidence</div>{hashes}<div class="k" style="margin-top:6px">intake</div>'
       f'<div class="v">{t0}</div><div class="{"ok" if ok else "no"}">{"INTEGRITY VERIFIED ✓" if ok else "NOT VERIFIED ✗"}</div></div>')


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
    return (f'<svg viewBox="0 0 300 168" width="100%" role="img" aria-label="Gauge: {e(label)}">' + "".join(segs)
            + f'<line x1="150" y1="140" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#E6EDF3" stroke-width="3" stroke-linecap="round"/>'
            f'<circle cx="150" cy="140" r="6" fill="#E6EDF3"/><text x="28" y="162" fill="#9AA7B4" font-size="11">No</text>'
            f'<text x="236" y="162" fill="#9AA7B4" font-size="11">Strong</text></svg>')


def label_hero(res, inds, a):
    """Gauge, label in words, what it mostly comes from, how sure the setup is."""
    head = res["headline"]
    if head.startswith(("Not analysed", "Failed")):
        why = "The analysis did not finish" if res.get("failed") else "No analyzer could run on this evidence"
        md(f'<div class="panel"><div class="hero"><div class="t"><div class="lab"><i style="color:#64748B">–</i>{e(head)}</div>'
           f'<div class="muted">{why}, so no label is given. The evidence files and their custody log are kept.</div></div></div></div>')
        return
    lab, p = a["case"]["label"], a["case"]["p"]
    c, icon, word = LEVEL[lab]
    contrib = sorted(fusion.contributions(inds).items(), key=lambda kv: -kv[1])
    if contrib and contrib[0][1] > 0:
        (f, ty), _ = contrib[0]
        best = max((i for i in inds if i.file == f and i.type == ty), key=fusion.contribution)
        mostly = (f'Mostly from: <b>{fusion.group(best)}</b> — <span title="{e(term(ty)[1])}">{e(term(ty)[0])}</span> '
                  f'{e(at(best.location))}' + (f' <span class="muted">({e(f)})</span>' if len(res["integrity"]) > 1 else ""))
    else:
        mostly = "Nothing counted toward the label: no finding carries a calibrated weight."
    limited = f'<div class="muted small">{e(head)}</div>' if head == fusion.LIMITED else ""
    sure = ""
    if cal:
        h = cal["held_out"]
        sure = (f'<div class="small muted" style="margin-top:8px">How sure? On the held-out test split this setup flagged '
                f'{h["hit_rate"][0]:.0%} of manipulated files and {h["false_alarm_rate"][0]:.0%} of untouched ones '
                f'({e(cal["dataset"])} test set, n={cal["n"]}).</div>')
    creds = f'<div class="small muted" style="margin-top:6px">{e(report.CREDENTIALS_NOTE)}</div>' if ss.get("_creds") else ""
    md(f'<div class="panel"><div class="hero"><div class="g">{gauge(lab, p)}</div><div class="t">'
       f'<div class="lab" style="--c:{c}"><i>{icon}</i>{word} indicators</div>{limited}'
       f'<div class="mostly">{mostly}</div>'
       f'<div>Automated indicators, not proof. Review the evidence tags below.</div>'
       f'<div class="mono tiny muted" style="margin-top:6px">fused strength {p:.2f} · label of manipulation or synthetic generation</div>'
       f'{creds}{sure}</div></div></div>')


# ---------- case list ----------
def case_dirs():
    if not CASES.exists():
        return []
    ds = [d for d in CASES.iterdir() if (d / "result.json").exists() and not d.name.startswith(HIDDEN)]
    return sorted(ds, key=lambda d: -d.stat().st_mtime)


def mtime(case_id):
    return (CASES / case_id / "result.json").stat().st_mtime


@st.cache_data(show_spinner=False)
def load(case_id, mtime):
    return pipeline.load(Case(case_id))


@st.cache_data(show_spinner=False)
def case_meta(case_id, mtime):
    """Sidebar entry: file names, headline, intake time, a small thumbnail (bytes or None)."""
    c, r = Case(case_id), load(case_id, mtime)
    ev = c.evidence()
    ing = next((x["ts"] for x in c.entries() if x["action"] == "ingest"), mtime)
    thumb = None
    for p in ev:
        cands = ([p, c.artifact_path(p.stem + ".png")] if p.suffix.lower() in IMAGE else
                 [c.artifact_path(f"{p.stem}_f0001.jpg"), c.artifact_path(f"{p.stem}_spec.png")])
        for q in cands:
            try:
                im = ImageOps.fit(Image.open(q).convert("RGB"), (96, 96))  # square, so every list entry is the same height
                b = io.BytesIO()
                im.save(b, "JPEG", quality=80)
                thumb = b.getvalue()
                break
            except (OSError, ValueError):
                continue
        if thumb:
            break
    kind = next((("🎞️" if p.suffix.lower() in VIDEO else "🎙️" if p.suffix.lower() in AUDIO else "📝" if p.suffix.lower() in TEXT_EXTS
                  else "🖼️") for p in ev), "📁")
    return {"files": [p.name for p in ev], "headline": r["headline"], "ts": ing, "thumb": thumb, "kind": kind}


@st.cache_data(show_spinner=False)
def report_html(case_id, mtime, log_size):
    c = Case(case_id)
    return report.render(c, load(case_id, mtime))


@st.fragment(run_every=30)
def server_pill(fallback=()):
    """GPU server status, checked at startup and every 30 s: grey when offline, amber when this case's answers came
    from a fallback provider, else cyan."""
    if "ollama" not in llm.PROVIDERS:
        md('<span class="pill" style="--c:#64748B"><i>○</i>GPU server: not configured</span>')
        return
    up, ms = llm.ping()
    text, c, icon = (("offline", "#64748B", "○") if not up else (f"online · {ms} ms · fallback used: {', '.join(fallback)}",
                     "#F2B544", "◑") if fallback else (f"online · {ms} ms", "#4FD1C5", "●"))
    md(f'<span class="pill" style="--c:{c}" title="Model server for the semantic checks"><i>{icon}</i>GPU {e(llm.host("ollama"))} · '
       f'{e(text)}</span>')


def when(ts):
    """Intake time for the case list: "23:20" today, else "18 Nov"."""
    lt = time.localtime(ts)
    return time.strftime("%H:%M" if lt[:3] == time.localtime()[:3] else "%d %b", lt)


def open_case(cid):
    qp.clear()
    qp["case"] = cid


known = [d.name for d in case_dirs()]
pick = qp.get("case") or (known[0] if known else "new")
if pick != "new" and not (CASES / pick / "result.json").exists() and not (ss.get("job", {}).get("case") == pick):
    pick = "new"

with st.sidebar:
    job = ss.get("job")
    server_pill(() if pick == "new" or pick not in known else tuple(llm.fallback_used(Case(pick).entries())))
    md('<h2 style="margin:10px 0 0">🔎 TruthLens</h2><div class="muted small" style="margin-bottom:6px">Multimodal forensic workbench</div>')
    if st.button("＋ New case", type="primary" if pick == "new" else "secondary", width="stretch"):
        open_case("new")
        st.rerun()
    if job and not job.get("done"):
        md(f'<div class="small" style="margin:6px 0">⏳ Analyzing {e(job["case"])}…</div>')
        if pick != job["case"] and st.button("Back to the running analysis", width="stretch"):
            open_case(job["case"])
            st.rerun()
    md('<div class="muted tiny" style="margin:14px 0 6px;letter-spacing:.08em">CASES · NEWEST FIRST</div>')
    if not known:
        md('<div class="muted small">No cases yet. Start with “New case”.</div>')
    for cid in known:
        m = case_meta(cid, mtime(cid))
        c1, c2 = st.columns([1, 3], vertical_alignment="center")
        if m["thumb"]:
            c1.image(m["thumb"], width="stretch")
        else:
            c1.markdown(f'<div class="thumb-ph" aria-hidden="true">{m["kind"]}</div>', unsafe_allow_html=True)
        name = m["files"][0] if m["files"] else cid
        name = (name if len(name) <= 24 else name[:22] + "…") + (f" +{len(m['files']) - 1}" if len(m["files"]) > 1 else "")
        if c2.button(name, key=f"open_{cid}", type="primary" if cid == pick else "tertiary", help=cid, width="stretch"):
            open_case(cid)
            st.rerun()
        c2.markdown(f'<div class="case-meta">{pill(m["headline"], short=True)}<span class="mono tiny muted">'
                    f'{when(m["ts"])}</span></div>', unsafe_allow_html=True)
    with st.expander("What do these terms mean?"):
        md("".join(f'<div class="small" style="margin-bottom:6px"><b>{e(n)}</b> · <span class="muted">{e(d)}</span></div>'
                   for n, d in sorted(TERMS.values())))
    st.caption("Automated indicators, not proof. A person reviews every finding.")


# ---------- 1. upload ----------
def upload_page():
    steps(0)
    st.title("New case")
    md('<div class="muted" style="margin:-6px 0 12px">Evidence is hashed (SHA-256) on arrival, stored read-only, and every step is logged.</div>')
    files = st.file_uploader("Drop evidence files here (several at once is fine)", accept_multiple_files=True)
    md('<div class="muted tiny" style="margin-top:-8px">Images: JPG, PNG, WebP, TIFF, BMP, GIF, AVIF, HEIC, JXL · Video: MP4, MOV, MKV, '
       'WebM, AVI · Audio: WAV, MP3, M4A, FLAC, OGG · Text: TXT, MD</div>')
    context = st.text_area("Caption / claims (optional)", height=90,
                           placeholder='e.g. Flood at the campus library on 2026-11-02. She said "we cancelled the fest".')
    with st.expander("Paste text evidence (optional): a message, post or article to examine"):
        pasted = st.text_area("Text evidence", label_visibility="collapsed",
                              placeholder="Stored as pasted_text.txt with its own hash; its claims are checked against the media.")
    if files:
        rows = []
        for f in files:
            n = f.name.lower()
            kind = ("image" if n.endswith(IMAGE) else "video" if n.endswith(VIDEO) else "audio" if n.endswith(AUDIO)
                    else "text" if n.endswith(TEXT_EXTS) else "unsupported")
            rows.append(f'<div class="mono small" style="overflow-wrap:anywhere">{ICON.get(kind, "⚠️")} {e(f.name)} '
                        f'<span class="muted">· {kind} · sha256 {hashlib.sha256(f.getvalue()).hexdigest()[:16]}…</span></div>')
        md('<div class="panel">' + "".join(rows) + "</div>")
    empty = not files and not pasted.strip()
    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    fresh = c2.checkbox("Ask the models again (don't reuse cached answers)")
    if c1.button("Analyze", type="primary", disabled=empty, width="stretch"):
        case = Case()
        uploads = {f.name: f.getvalue() for f in files or []}
        if pasted.strip():
            uploads["pasted_text.txt"] = pasted.strip().encode("utf-8")
        for n, b in uploads.items():
            if sha256(case.ingest(n, b)) != hashlib.sha256(b).hexdigest():
                st.error(f"The stored copy of {n} does not match the upload. Please upload it again.")
                st.stop()
        job = {"case": case.id, "context": context, "fresh": fresh, "ev": [], "t0": time.time()}
        threading.Thread(target=work, args=(job,), daemon=True).start()
        ss["job"] = job
        open_case(case.id)
        st.rerun()
    if empty:
        md('<div class="muted small">Add at least one file (or paste text) to start.</div>')


def work(job):
    """The analysis, off the page thread, so the page can show live progress and elapsed time."""
    llm.FRESH = job["fresh"]
    try:
        pipeline.run(Case(job["case"]), job["context"], lambda step, state: job["ev"].append((time.time(), step, state)))
    except Exception:
        job["error"] = traceback.format_exc()
    finally:
        llm.FRESH = False
        job["done"] = time.time()


# ---------- 2. analyzing ----------
def progress_html(job, case):
    rows, state, t_on, t_off = [], {}, {}, {}
    for ts, step, s in job["ev"]:
        state[step] = s
        if s == "running":
            t_on[step] = ts
        elif s != "queued":
            t_off[step] = ts
    errs = {f'{x["analyzer"].rsplit(".", 1)[-1]} · {x["file"]}': x["error"] for x in case.entries() if x["action"] == "analyzer_error"}
    now = job.get("done") or time.time()
    look = {"queued": ("#9AA7B4", "queued"), "running": ("#4FD1C5", "running…"), "done": ("#A3B18A", "✓ done"),
            "error": ("#F97352", "⚠ failed")}
    for step, s in state.items():
        c, word = look.get(s, ("#9AA7B4", s))
        secs = (t_off.get(step, now) - t_on[step]) if step in t_on else None
        note = ""
        if s == "error":
            err = errs.get(step, "")
            note = (f' <span class="muted small">— {e(err)}: stopped by the watchdog, the case continues</span>' if "timed out" in err
                    else f' <span class="muted small">— the case continues without it</span>')
        kind, _, f = step.partition(" · ")
        name = STEP_NAMES.get(kind, kind) + (f" · {f}" if f else "")
        rows.append(f'<div><span class="st" style="color:{c}">{word}</span><span class="nm">{e(name)}{note}</span>'
                    f'<span class="tm">{"" if secs is None else f"{secs:.1f}s"}</span></div>')
    for x in case.entries():  # checks that could not run, as soon as they are logged
        if x["action"] == "check_not_run":
            rows.append(f'<div><span class="st" style="color:#9AA7B4">– not run</span><span class="nm">{e(check_name(x["check"]))}'
                        f' <span class="muted small">— {e(short_reason(x["reason"]))}</span></span><span class="tm"></span></div>')
    return ('<div class="panel prog" role="status" aria-live="polite"><div class="head" style="display:flex;justify-content:space-between">'
            f'<b>{"Analysis complete" if job.get("done") else "Analyzing…"}</b><span class="mono small">elapsed {now - job["t0"]:.0f}s</span></div>'
            + ("".join(rows) or '<div class="muted">Reading the files…</div>') + "</div>")


def analyzing_page(job):
    steps(1)
    case = Case(job["case"])
    st.title(case.id)
    ev = case.evidence()
    c1, c2 = st.columns([1, 2])
    with c1:
        p = next((p for p in ev if p.suffix.lower() in IMAGE), None)
        try:
            inner = f'<img class="ev" src="{img_uri(Image.open(p), 600)}" alt="{e(p.name)}">' if p else \
                f'<div style="font-size:3rem;text-align:center;padding:40px 0" aria-hidden="true">{"🎞️" if any(q.suffix.lower() in VIDEO for q in ev) else "🎙️"}</div>'
        except OSError:
            inner = '<div style="font-size:3rem;text-align:center;padding:40px 0" aria-hidden="true">🖼️</div>'
        md(f'<div class="lt"><div class="{"" if job.get("done") else "lamp"}">{inner}</div><div class="cap small muted">'
           + " · ".join(e(q.name) for q in ev) + "</div></div>")
    with c2:
        live_progress()


@st.fragment(run_every=0.5)
def live_progress():
    """Only this box re-runs while the analysis works, so the rest of the page stays put; switches to Results when done."""
    job = ss.get("job")
    if not job:
        return
    case = Case(job["case"])
    md(progress_html(job, case))
    if not job.get("done"):
        return
    if not job.get("error"):
        ss.pop("job", None)
        st.rerun(scope="app")
    st.warning("The analysis stopped unexpectedly. The case and its evidence are kept, and it is marked failed.", icon="⚠️")
    with st.expander("Details"):
        st.code(job["error"][-3000:])
    if st.button("Open the case", type="primary"):
        ss.pop("job", None)
        st.rerun(scope="app")


# ---------- 3. results ----------
def overview(case, res, inds, a, tags):
    creds = report.content_credentials(case, inds)
    ss["_creds"] = bool(creds)
    for c in creds:
        md(f'<div class="prov" role="note">🔏 {e(c)}</div>')
    label_hero(res, inds, a)
    cov = res["coverage"]
    if not res["headline"].startswith(("Not analysed", "Failed")):
        md(f'<div class="panel small cov">{e(coverage_html(cov))}'
           + "".join(f'<div class="muted" style="margin-top:4px">→ {e(h)}</div>' for h in hints(cov["not_run"].values())) + "</div>")
    if res["headline"].startswith(("Not analysed", "Failed")):
        md('<div class="panel small">→ Next step: start a <b>New case</b> with the same file'
           + (f' (the cause is {e(res["failed"].split("(bug ")[-1].rstrip(")"))}).' if "(bug fixed" in (res.get("failed") or "") else ".")
           + " Its evidence and custody log stay here.</div>")
        with st.expander("Details"):
            st.code(res.get("failed") or cov.get("why") or "")
        return
    top = [(t, i) for t, i in tags if fusion.contribution(i) > 0][:3]
    md(f'<h3 style="margin:18px 0 8px">Top evidence tags</h3>')
    if not top:
        md('<div class="panel muted">No finding counts toward the label. Findings that were observed but not counted are under '
           '<b>All findings</b>. No indicators does not show the media is unaltered.</div>')
    for t, i in top:
        c1, c2 = st.columns([7, 1], vertical_alignment="center")
        with c1:
            tag_card(t, i)
        if c2.button("Show me", key=f"show_{t}", help=f"Open {t} in the evidence viewer", width="stretch"):
            ss[f"tab_{case.id}"] = "Evidence"
            ss["focus"] = {"case": case.id, "tag": t, "file": i.file if i.file != "context" else None}
            ss["_scroll"] = True
            st.rerun()
    c1, c2 = st.columns([3, 2])
    with c1:
        checks = report.human_checks(inds)[:5]
        md('<div class="panel"><b>Recommended human checks</b><ol style="margin:6px 0 0 18px">'
           + ("".join(f"<li>{e(c)}</li>" for c in checks) or "<li>Spot-check the original files yourself.</li>") + "</ol></div>")
    with c2:
        md('<div class="panel"><b>Summary</b> <span class="muted tiny">written from the findings only</span><div class="small" '
           f'style="margin-top:6px">{e(report.summary(res, case))}</div></div>')
    bad = [f for f, ok in res["integrity"].items() if not ok]
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.markdown(f'<div class="small {"" if not bad else "panel bad"}">' + (f"✗ Integrity check FAILED: {e(', '.join(bad))}" if bad else
                f"✓ Integrity: all {len(res['integrity'])} evidence file(s) match their intake SHA-256.") + "</div>", unsafe_allow_html=True)
    log_size = case.log_path.stat().st_size if case.log_path.exists() else 0
    if c2.download_button("Export report", report_html(case.id, mtime(case.id), log_size), f"{case.id}.html", "text/html",
                          type="primary", width="stretch", help="Self-contained HTML; print it to PDF from the browser"):
        case.log("report_export", format="html")


def evidence_tab(case, res, inds, tags, log):
    names = [p.name for p in case.evidence()]
    if not names:
        md('<div class="panel muted">This case has no evidence files.</div>')
        return
    focus = ss.get("focus") if (ss.get("focus") or {}).get("case") == case.id else None
    key = f"file_{case.id}"
    if focus and focus.get("file") in names:
        ss[key] = focus["file"]
    elif key not in ss:
        ss[key] = qp.get("file") if qp.get("file") in names else names[0]
    fname = st.selectbox("Evidence file", names, key=key) if len(names) > 1 else names[0]
    path = case.dir / "evidence" / fname
    ext, stem = path.suffix.lower(), path.stem
    tag_of = {id(i): t for t, i in tags}
    mine = [(t, i) for t, i in tags if i.file == fname]
    fi = next((i for t, i in tags if focus and t == focus["tag"]), None)
    digest = next((x["sha256"] for x in log if x["action"] == "ingest" and x["file"] == fname), "?")
    md(f'<div style="margin:4px 0 10px">{pill(fusion.headline(res["assessment"], res["coverage"], fname))} '
       f'<span class="mono tiny muted" title="{e(digest)}">sha256 {digest[:12]}…{digest[-4:]}</span>'
       + (f' <span class="small">· focused on <b class="mono">{focus["tag"]}</b></span>' if fi is not None and fi.file == fname else "") + "</div>")
    if conv := next((x for x in log if x["action"] == "converted" and x["file"] == fname), None):
        md(f'<div class="muted small">Shown and analysed via {e(conv["to"])} (converted with {e(conv["tool"])}): '
           'compression-based checks (ELA) see the conversion, not the original. The hash above is the original file.</div>')
    if ext in IMAGE:
        image_view(case, path, stem, mine, fi)
    if ext in VIDEO:
        video_view(case, path, stem, mine, fi)
    if ext in TEXT_EXTS:
        md('<div class="panel mono small" style="white-space:pre-wrap;overflow-wrap:anywhere">'
           + e(path.read_text(encoding="utf-8", errors="replace")[:20000]) + "</div>")
    if ext in AUDIO or ext in VIDEO:
        audio_view(case, path, stem, mine, fi, log)
    md("<h4>Evidence tags in this file</h4>")
    for g, what in GROUPS.items():
        found = [(t, i) for t, i in mine if fusion.group(i) == g]
        if found:
            md(f'<div class="small" style="margin:8px 0 4px"><span class="chip" style="--c:{GROUP_C[g]}">{g}</span> '
               f'<span class="muted">{what}</span></div>')
            for t, i in found:
                tag_card(t, i, open_details=fi is i)
    if not mine:
        md('<div class="panel muted">No findings for this file. That does not show it is unaltered: review it yourself.</div>')
    context_view(res, log, stem, case)


def brackets(d, box, color, label, width, size):
    """Crosshair marker: thin corner brackets (camera focus box) with the tag number above them."""
    x, y, w, h = box
    k = max(6, min(w, h) // 4)
    for cx, cy, sx, sy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        d.line([cx, cy, cx + sx * k, cy], fill=color, width=width)
        d.line([cx, cy, cx, cy + sy * k], fill=color, width=width)
    font = ImageFont.load_default(size)
    l, t, r, b = d.textbbox((0, 0), label, font=font)
    top = max(0, y - (b - t) - 8)
    d.rectangle([x, top, x + r - l + 8, top + b - t + 6], fill=(13, 17, 23, 230))
    d.text((x + 4 - l, top + 3 - t), label, fill=color, font=font)


def image_view(case, path, stem, mine, fi):
    k1, k2, k3 = st.columns([2, 2, 2], vertical_alignment="bottom")
    avail = [k for k, (suf, _) in LAYER.items() if case.artifact_path(stem + suf).exists()]
    layers = k1.multiselect("Overlay layers", avail, default=avail[1:2] or avail[:1], key=f"lay_{case.id}_{stem}")
    opacity = k2.slider("Overlay opacity", 0.0, 1.0, 0.6, key=f"op_{case.id}_{stem}")
    boxes = k3.toggle("Flagged regions", True, key=f"bx_{case.id}_{stem}")
    cm = case.artifact_path(stem + "_copymove.png")
    show_cm = cm.exists() and k3.toggle("Copy-move pairs", True, key=f"cm_{case.id}_{stem}")
    c2 = st.container()
    try:
        base = Image.open(readable_image(path, case) or path).convert("RGBA")
    except (OSError, ValueError):
        c2.info("This image can't be displayed here (unreadable format). Its findings are listed below.", icon="ℹ️")
        return
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
        lw = max(2, base.width // 400)
        for t, i in mine:
            col = "#BB9AF7" if i.type == "synthetic_visual_artifacts" else AMBER
            for kb in ("bbox", "bbox2"):
                if kb in i.location:
                    brackets(d, i.location[kb], "#4FD1C5" if fi is i else col, t, lw * (2 if fi is i else 1),
                             max(13, base.width // 45))
    with c2:
        v1, v2 = st.columns(2)
        with v1:
            light_table(f'<img class="ev" src="{img_uri(out)}" alt="Analysis view of {e(path.name)} with flagged regions">',
                        "Analysis view · corner brackets = flagged regions, labelled by tag · violet = model-listed artifacts")
        with v2:
            other = Image.open(cm) if show_cm else base
            light_table(f'<img class="ev" src="{img_uri(other)}" alt="{"Copy-move pairs" if show_cm else "Original"} of {e(path.name)}">',
                        "Copy-move pairs" if show_cm else "Original, as ingested")
        if fi is not None and fi.file == path.name and "bbox" in fi.location:
            x, y, w, h = fi.location["bbox"]
            pad = max(w, h) // 2
            crop = base.crop((max(0, x - pad), max(0, y - pad), min(base.width, x + w + pad), min(base.height, y + h + pad)))
            light_table(f'<img class="ev" style="max-height:320px" src="{img_uri(crop, 800)}" alt="Close-up of the focused region">',
                        f"Close-up of {e(ss['focus']['tag'])} · region x{x} y{y}, {w}×{h}px")
    spec = case.artifact_path(f"{stem}_spectrum.png")
    if spec.exists():
        with st.expander("Noise spectrum (upsampling trace)"):
            light_table(f'<img class="ev" style="max-height:360px" src="{img_uri(Image.open(spec), 700)}" alt="Noise spectrum">',
                        "Log power of the high-pass residual, centre = low frequencies. Bright isolated dots off the centre are "
                        "periodic patterns; a dark outer ring is missing fine detail.")


def video_view(case, path, stem, mine, fi):
    timed = sorted((i for _, i in mine if tt(i) is not None), key=tt)
    start = int(max(0.0, tt(fi) - 0.5)) if fi is not None and fi.file == path.name and tt(fi) is not None else 0
    vc, tc = st.columns([3, 2])
    vc.video(str(path), start_time=start)
    with tc:
        flagged = [(t, i) for t, i in mine if tt(i) is not None and fusion.contribution(i) >= fusion.LEVELS[2][0]]
        md('<div class="panel small"><b>Flagged moments</b>' + ("".join(
            f'<div style="margin-top:4px"><span class="mono" style="color:#4FD1C5">{t}</span> <span class="mono">{tt(i):.2f}s</span> '
            f'{e(term(i.type)[0])}</div>' for t, i in sorted(flagged, key=lambda ti: tt(ti[1]))[:10])
            or '<div class="muted" style="margin-top:4px">No timed finding reaches the Weak level.</div>') + "</div>")
        if start:
            md(f'<div class="small">Player starts at {start}s (0.5 s before {e(ss["focus"]["tag"])}).</div>')
    frames = sorted(case.artifact_path("").glob(f"{stem}_f[0-9][0-9][0-9][0-9].jpg"))
    if not frames:
        md('<div class="panel muted small">No frames could be read from this video, so the per-frame checks could not run.</div>')
    else:
        hits = {}
        for i in timed:
            if i.modality == "video" and fusion.contribution(i) >= fusion.LEVELS[2][0]:  # Weak or above
                hits.setdefault(round(tt(i)), LEVEL[fusion.label(fusion.contribution(i))][0])
        md('<div class="small muted" style="margin:6px 0 4px">Keyframes, 1 per second · ⚑ = a counted flag of Weak level or above</div>'
           + film_strip(frames, hits, tt(fi) if fi is not None and fi.file == path.name else None))
    dur = max([len(frames), 1] + [i.location.get("end", tt(i)) + 1 for i in timed])
    X = lambda t: 40 + 940 * t / dur
    marks = "".join(
        f'<g><title>{e(term(i.type)[0])} {e(where(i.location))}</title><rect x="{X(tt(i)):.1f}" y="{14 if i.modality == "video" else 40}" '
        f'width="{max(3, X(i.location["end"]) - X(tt(i))) if "end" in i.location else 3:.1f}" height="18" rx="2" '
        f'fill="{"#4FD1C5" if i.modality == "video" else "#F2B544"}"/></g>' for i in timed if i.modality in ("video", "audio"))
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
    lips = (curve(lip["mouth"], "#4FD1C5") + curve(lip["audio_db"], "#F2B544")
            + f'<text x="985" y="100" fill="#9AA7B4" font-size="11" text-anchor="end">best lag {lip["lag_ms"]:+d} ms · '
              f'r {lip["r"]:.2f}</text>') if lip else ""
    ticks = "".join(f'<text x="{X(s):.0f}" y="148" fill="#9AA7B4" font-size="12" font-family="JetBrains Mono,monospace">{s:.0f}s</text>'
                    for s in np.linspace(0, dur, 9))
    light_table(f'<svg viewBox="0 0 1000 154" width="100%" role="img" aria-label="Timeline of flags">'
                f'<text x="0" y="27" fill="#9AA7B4" font-size="12">video</text><text x="0" y="53" fill="#9AA7B4" font-size="12">audio</text>'
                f'<text x="0" y="78" fill="#9AA7B4" font-size="11">face</text><text x="0" y="115" fill="#9AA7B4" font-size="11">lips</text>'
                f'<rect x="40" y="14" width="940" height="18" fill="#0D1117"/><rect x="40" y="40" width="940" height="18" fill="#0D1117"/>'
                f'<rect x="40" y="66" width="940" height="14" fill="#0D1117"/><rect x="40" y="90" width="940" height="40" fill="#0D1117"/>'
                f'{marks}{faces}{lips}{ticks}</svg>',
                f'Cyan: video flags · amber: audio flags · green: face visible ({track["face_ratio"]:.0%} of frames) · lips: mouth motion '
                f'(cyan) vs speech loudness (amber)' + ("" if lip else " · lip-sync not run (no face or no speech)"))
    for _, i in mine:
        if i.type == "learned_synthetic_video" and i.location.get("segments"):
            score_strip(i, i.location["segments"], "two-second segment")
            crops = [c for c in i.location.get("crops", []) if case.artifact_path(c["file"]).exists()]
            if crops:
                md('<div class="film" role="list" aria-label="Frames the trained detector scored highest">' + "".join(
                    f'<figure class="hit" style="--c:#BB9AF7"><img src="{img_uri(Image.open(case.artifact_path(c["file"])), 240, 80)}" '
                    f'alt="{"Face" if c["face"] else "Frame"} at {c["t"]:.1f} s"><figcaption>{c["t"]:.1f}s · '
                    f'{"face" if c["face"] else "frame"} · {c["score"]:.2f}</figcaption></figure>' for c in crops) + "</div>")


def score_strip(i, cells, what):
    """Trained-detector scores per time window as bars (threshold dashed), under the spectrogram or video timeline."""
    dur = max(c[1] for c in cells)
    X = lambda t: 40 + 940 * t / dur
    thr = i.location.get("threshold", 0.5)
    bars = "".join(f'<rect x="{X(c[0]):.1f}" y="{70 - 56 * c[2]:.1f}" width="{max(2.0, X(c[1]) - X(c[0]) - 1):.1f}" '
                   f'height="{56 * c[2]:.1f}" fill="{"#BB9AF7" if c[2] >= thr else "#3B4252"}" fill-opacity="0.75">'
                   f'<title>{c[0]:.1f}–{c[1]:.1f}s · {c[2]:.2f}</title></rect>' for c in cells)
    ticks = "".join(f'<text x="{X(s):.0f}" y="86" fill="#9AA7B4" font-size="11" font-family="JetBrains Mono,monospace">{s:.0f}s</text>'
                    for s in np.linspace(0, dur, 9))
    light_table(f'<svg viewBox="0 0 1000 90" width="100%" role="img" aria-label="Trained detector score per {what}">'
                f'<rect x="40" y="14" width="940" height="56" fill="#0D1117"/>{bars}'
                f'<line x1="40" x2="980" y1="{70 - 56 * thr:.1f}" y2="{70 - 56 * thr:.1f}" stroke="#E6EDF3" stroke-dasharray="4 3"/>'
                f'<text x="0" y="46" fill="#9AA7B4" font-size="11">score</text>{ticks}</svg>',
                f"Trained detector ({e(i.location.get('model_version', '?'))}) score per {what} · violet = at or above its "
                f"threshold {thr:.2f} (dashed) · indicative: the threshold was set on whole files")


def audio_view(case, path, stem, mine, fi, log):
    spec = case.artifact_path(f"{stem}_spec.png")
    if spec.exists():
        light_table(f'<img class="ev" style="width:100%;max-height:300px;object-fit:fill" src="{img_uri(Image.open(spec), 1600)}" '
                    f'alt="Spectrogram of {e(path.name)}">',
                    "Spectrogram · low frequencies at the bottom · red: splice/click span · blue: too-clean pauses · amber: "
                    "flat-prosody sentences · cyan line: band-limit cutoff")
    elif any(x["action"] == "no_audio" and x["file"] == path.name for x in log):
        md('<div class="panel muted small">No audio track was found in this file, so the audio checks couldn\'t run.</div>')
        return
    start = 0
    if fi is not None and fi.file == path.name and fi.modality == "audio" and "start" in fi.location:
        start = int(max(0, fi.location["start"] - 0.5))
    for _, i in mine:
        if i.type == "learned_synthetic_audio" and i.location.get("windows"):
            score_strip(i, i.location["windows"], "two-second window")
    wav = case.artifact_path(f"{stem}_16k.wav")
    st.audio(str(wav if wav.exists() else path), start_time=start)
    if start:
        md(f'<div class="small">Player starts at {start}s (0.5 s before {e(ss["focus"]["tag"])}).</div>')


def context_view(res, log, stem, case):
    claims = next((x for x in reversed(log) if x["action"] == "text_claims"), {})
    if res.get("context"):
        marked = e(res["context"])
        colors = {"when": "#4FD1C5", "where": "#A3B18A", "who": "#BB9AF7", "quotes": "#F2B544"}
        for k, c in colors.items():
            for v in claims.get(k, []):
                if v and e(str(v)) in marked:
                    marked = marked.replace(e(str(v)), f'<mark style="background:{c}" title="{k}">{e(str(v))}</mark>', 1)
        md('<h4>Caption / claims</h4><div class="panel">' + marked + '<div class="muted small" style="margin-top:6px">claims: '
           + " ".join(f'<mark style="background:{c}">{k}</mark>' for k, c in colors.items()) + "</div></div>")
    tp = case.artifact_path(f"{stem}_transcript.json")
    if tp.exists():
        said = " ".join(s["text"] for s in json.loads(tp.read_text(encoding="utf-8")))
        found = [(q, fuzz.partial_ratio(str(q).lower(), said.lower())) for q in claims.get("quotes", [])]
        md('<h4>Transcript</h4><div class="panel mono small">' + (e(said) or '<span class="muted">(no speech recognised)</span>')
           + "</div>" + "".join(f'<div class="small">{"✓ heard" if r >= 70 else "✗ not found"} “{e(str(q))}” '
                                f'<span class="muted mono">match {r:.0f}/100</span></div>' for q, r in found))


def cross_tab(inds, tags, not_run):
    md('<div class="muted small" style="margin-bottom:8px">One lane per source on a shared time axis. Curves join flags that a '
       'cross-media check found to line up (video and audio within 0.5 s) or to contradict each other (claim vs metadata or '
       'transcript).</div>')
    lanes = {"Video": 30, "Audio": 90, "Text / claims": 150, "Metadata": 210}
    timed = [i for i in inds if tt(i) is not None and i.modality in ("video", "audio")]
    dur = max([1.0] + [tt(i) + 1 for i in timed])
    X = lambda t: 200 + 780 * t / dur
    els = [f'<text x="0" y="{y + 4}" fill="#9AA7B4" font-size="14">{n}</text><line x1="200" y1="{y}" x2="980" y2="{y}" '
           f'stroke="#263040" stroke-width="18" stroke-linecap="round"/>' for n, y in lanes.items()]
    els.append('<text x="125" y="252" fill="#9AA7B4" text-anchor="middle" font-size="12">untimed</text>')
    els += [f'<text x="{X(s):.0f}" y="252" fill="#9AA7B4" font-size="12" font-family="JetBrains Mono,monospace">{s:.0f}s</text>'
            for s in np.linspace(0, dur, 7)]
    for i in timed:
        y, c = (lanes["Video"], "#4FD1C5") if i.modality == "video" else (lanes["Audio"], "#F2B544")
        els.append(f'<g><title>{e(i.file)}: {e(term(i.type)[0])} {e(where(i.location))}</title><circle cx="{X(tt(i)):.1f}" cy="{y}" r="6" fill="{c}"/></g>')
    for n, i in enumerate(i for i in inds if "meta" in i.location):
        els.append(f'<g><title>{e(i.file)}: {e(term(i.type)[0])}</title><rect x="{205 + 12 * n}" y="{lanes["Metadata"] - 5}" width="8" height="10" fill="#9AA7B4"/></g>')
    cross = [(t, i) for t, i in tags if i.modality == "cross"]
    for n, (t, i) in enumerate(cross):
        if i.type == "av_aligned_flags":
            x = X(i.location["t"])
            x2 = X(min((tt(j) for j in timed if j.modality == "audio" and j.file in i.related),
                       key=lambda v: abs(v - i.location["t"]), default=i.location["t"]))
            els.append(f'<g><title>{t}: {e(i.explanation)}</title><path d="M{x:.1f},{lanes["Video"]} C{x:.1f},60 {x2:.1f},60 {x2:.1f},{lanes["Audio"]}" '
                       f'stroke="#F97352" stroke-width="2.5" fill="none"/></g>')
        else:
            y1, y2 = lanes["Text / claims"], lanes[{"claim_date_mismatch": "Metadata", "quote_not_in_transcript": "Audio"}.get(i.type, "Video")]
            x = 150 + 14 * (n % 4)
            els.append(f'<g><title>{t}: {e(i.explanation)}</title><path d="M{x},{y1} C{x - 50},{y1} {x - 50},{y2} {x},{y2}" stroke="#F97352" '
                       f'stroke-width="2.5" fill="none"/><circle cx="{x}" cy="{y1}" r="5" fill="#F97352"/><circle cx="{x}" cy="{y2}" r="5" fill="#F97352"/></g>')
    light_table(f'<svg viewBox="0 0 1000 262" width="100%" role="img" aria-label="Cross-media lanes">{"".join(els)}</svg>',
                "Cyan: video flags · amber: audio flags · grey: metadata flags · coral curves: cross-media checks (hover for details)")
    for t, i in cross:
        tag_card(t, i)
    if not cross:
        md('<div class="panel muted">No cross-media links or contradictions were found.'
           + "".join(f"<div class='small' style='margin-top:4px'>→ {e(h)}</div>" for h in hints(not_run.values())) + "</div>")


def findings_tab(case, inds, tags, cov):
    c1, c3 = st.columns(2)
    c2 = st.container()
    groups = c1.pills("Evidence type", list(GROUPS), selection_mode="multi", default=list(GROUPS), key=f"fg_{case.id}")
    medias = sorted({i.modality for i in inds}) or ["image"]
    media = c2.pills("Media", medias, selection_mode="multi", default=medias, key=f"fm_{case.id}",
                     format_func=lambda m: f"{ICON.get(m, '')} {m}")
    status = c3.pills("Status", ["counted", "uncalibrated", "not run"], selection_mode="multi",
                      default=["counted", "uncalibrated", "not run"], key=f"fs_{case.id}")
    keep = [(t, i) for t, i in tags if fusion.group(i) in groups and i.modality in media]
    counted = [(t, i) for t, i in keep if fusion.calibrated(i) and fusion.contribution(i) > 0]
    other = [(t, i) for t, i in keep if (t, i) not in counted]
    if "counted" in status:
        md(f'<h4>Counted in the label <span class="muted small">· {len(counted)} · sorted by contribution</span></h4>')
        for t, i in counted:
            tag_card(t, i)
        if not counted:
            md('<div class="panel muted small">Nothing here matches these filters.</div>')
    shown_other = [(t, i) for t, i in other if "uncalibrated" in status]
    nr = cov["not_run"] if "not run" in status else {}
    if shown_other or nr:
        with st.expander(f"Also observed (not counted in label) · {len(shown_other)} findings · {len(nr)} checks not run"):
            for t, i in shown_other:
                tag_card(t, i)
            for c, w in nr.items():
                md(f'<div class="small" style="margin:4px 0"><b>{e(check_name(c))}</b> <span class="muted">· not run: {e(w)}</span></div>')
    if not inds:
        md('<div class="panel muted">No indicators were raised for this case. That does not show the media is unaltered.</div>')
    with st.expander("Table view (all columns)"):
        rows = [{"tag": t, "evidence type": fusion.group(i), "indicator": i.type, "medium": i.modality, "file": i.file,
                 "location": where(i.location), "score": round(i.score, 2), "weight": round(fusion.weight(i), 2),
                 "contribution": round(fusion.contribution(i), 3), "counted": "yes" if fusion.calibrated(i) else fusion.UNCALIBRATED,
                 "why": i.explanation, "could also be": "; ".join(i.alt_explanations)} for t, i in tags]
        st.dataframe(rows, width="stretch", hide_index=True) if rows else st.caption("No findings.")
    md(f'<div class="muted tiny" style="margin-top:8px">{e(fusion.ALIGN_RULE)}</div>')


def custody_tab(case, res, log):
    custody_stamp(case, log, res["integrity"])
    c1, c2 = st.columns([3, 2])
    with c1:
        md("<h4>Chain of custody</h4>")
        colors = {"ingest": "#4FD1C5", "analyzed": "#A3B18A", "llm_call": "#BB9AF7", "check_not_run": "#64748B",
                  "assessed": "#F2B544", "analyzer_error": "#F97352", "analysis_failed": "#F97352", "verify": "#4FD1C5",
                  "report_export": "#4FD1C5"}
        items = []
        for x in log:
            txt = ", ".join(f"{k}={v}" for k, v in x.items() if k not in ("ts", "action"))[:220]
            items.append(f'<div style="--c:{colors.get(x["action"], "#263040")}"><span class="mono tiny muted">'
                         f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x["ts"]))}</span> <b>{e(x["action"])}</b>'
                         f'<br><span class="mono tiny muted" style="overflow-wrap:anywhere">{e(txt)}</span></div>')
        md('<div class="tl">' + "".join(items) + "</div>")
    with c2:
        md("<h4>Integrity</h4>")
        if st.button("Verify integrity now", type="primary"):
            v = case.verify()
            case.log("verify", **{k: ("ok" if ok else "MISMATCH") for k, ok in v.items()})
            md("".join(f'<div class="mono small">{"✓" if ok else "✗"} {e(k)} · {sha256(case.dir / "evidence" / k)[:16]}…</div>'
                       for k, ok in v.items()))
        with st.expander("Full SHA-256 hashes (copy)"):
            for x in log:
                if x["action"] == "ingest":
                    st.code(f'{x["sha256"]}  {x["file"]}', language=None)
        st.caption("Export the report from the overview. For PDF: open the HTML report and print → Save as PDF.")


def calibration_tab():
    if not cal:
        md('<div class="panel bad">No calibration loaded. Run <code>python calibrate.py</code> on the labelled test set; until '
           "then every weight is 0 and labels are not meaningful.</div>")
        return
    col = "#4FD1C5" if cal["dataset"] == "REAL" else "#F2B544"
    md(f'<span class="badge" style="--c:{col}">{e(cal["dataset"])} · n={cal["n"]}</span> <span class="muted small mono">'
       f'{e(cal["date"])} · git {e(cal["git"])} · fitted on n={cal["n_calibration"]}, held-out n={cal["n_held_out"]}</span>')
    if cal["dataset"] != "REAL":
        md('<div class="panel small warn" style="margin-top:8px">These numbers come from SYNTHETIC placeholder media. They show '
           'the system runs end to end; they are not detector accuracy.</div>')
    det = cal["detectors"]
    rows = []
    for n, (t, d) in enumerate(det.items()):
        p, lo, hi = d["precision"]
        y = n * 30 + 12
        X = lambda v: 260 + v * 540
        rows.append(f'<text x="0" y="{y + 4}" fill="#E6EDF3" font-size="12">{e(t)}</text>'
                    f'<line x1="{X(lo):.0f}" y1="{y}" x2="{X(hi):.0f}" y2="{y}" stroke="#4FD1C5" stroke-width="2"/>'
                    f'<line x1="{X(lo):.0f}" y1="{y - 5}" x2="{X(lo):.0f}" y2="{y + 5}" stroke="#4FD1C5" stroke-width="2"/>'
                    f'<line x1="{X(hi):.0f}" y1="{y - 5}" x2="{X(hi):.0f}" y2="{y + 5}" stroke="#4FD1C5" stroke-width="2"/>'
                    f'<circle cx="{X(p):.0f}" cy="{y}" r="5" fill="#E6EDF3"/>'
                    f'<text x="820" y="{y + 4}" fill="#9AA7B4" font-size="11" font-family="JetBrains Mono,monospace">'
                    f'{d["hits"]}/{d["n_manipulated"]} vs {d["false_alarms"]}/{d["n_untouched"]} · w {d["weight"]}</text>')
    axis = "".join(f'<text x="{260 + v * 540:.0f}" y="{len(det) * 30 + 24}" fill="#9AA7B4" font-size="10" text-anchor="middle">{v:.0%}</text>'
                   for v in (0, .25, .5, .75, 1))
    md('<div class="panel" style="margin-top:12px"><b>Precision per detector</b> <span class="muted small">dot = measured · '
       'bar = 95% Wilson interval · right: fired on manipulated vs untouched · fusion weight</span>'
       f'<svg viewBox="0 0 1000 {len(det) * 30 + 30}" width="100%" role="img" aria-label="Precision per detector">{"".join(rows)}{axis}</svg></div>')
    h = cal["held_out"]
    td = 'style="padding:6px 14px"'
    md(f'''<div class="panel"><b>Held-out confusion matrix</b> <span class="muted small">Moderate/Strong = flagged</span>
<table style="margin-top:8px;border-collapse:collapse" class="mono"><tr><td></td><td class="muted" {td}>flagged</td>
<td class="muted" {td}>not flagged</td></tr><tr><td class="muted" {td}>manipulated</td><td {td}>{h["tp"]}</td><td {td}>{h["fn"]}</td></tr>
<tr><td class="muted" {td}>untouched</td><td {td}>{h["fp"]}</td><td {td}>{h["tn"]}</td></tr></table>
<div class="small" style="margin-top:6px">Hit rate {h["hit_rate"][0]:.0%} ({h["hit_rate"][1]:.0%}–{h["hit_rate"][2]:.0%}) · false-alarm rate
{h["false_alarm_rate"][0]:.0%} ({h["false_alarm_rate"][1]:.0%}–{h["false_alarm_rate"][2]:.0%})</div></div>''')
    md(f'''<div class="panel small"><b>Limitations</b><ul>
<li>The test set is small and team-made, so every rate has a wide interval: read the bars, not the dots.</li>
<li>Platform recompression and resizing can both create and erase these indicators.</li>
<li>{e(fusion.ALIGN_RULE)}</li>
<li>Trained detectors: {e(learned.WEAK)} {e(learned.FIRST_RESULT)}</li>
<li>Fully generated media without provenance metadata or visible artifacts may show no indicators at all.</li>
<li>No indicators does not mean unaltered. The language model reads text, compares captions with pictures and lists visible
artifacts with their location; it is never asked whether media is synthetic.</li></ul></div>''')
    if lw := cal.get("learned"):
        td = 'style="padding:4px 12px"'
        md('<div class="panel small"><b>Trained detectors on our own media</b> <span class="muted">weight 0 (uncalibrated, not '
           f"counted in the label) until a media type has {MIN_OWN} untouched and {MIN_OWN} AI files of our own</span>"
           f'<table class="mono" style="margin-top:6px;border-collapse:collapse"><tr>'
           + "".join(f'<td class="muted" {td}>{k}</td>' for k in ("detector", "untouched (phone hold-out)", "false alarms", "AI files",
                                                                 "weight", "still needed")) + "</tr>"
           + "".join(f'<tr><td {td}>{e(t)}</td><td {td}>{v["n_untouched"]} ({v["phone_holdout"]})</td>'
                     f'<td {td}>{v["false_alarms"]}/{v["n_untouched"]}</td><td {td}>{v["n_manipulated"]}</td>'
                     f'<td {td}>{"uncalibrated" if v["weight"] is None else v["weight"]}</td>'
                     f'<td {td}>{v["need"]["untouched"]} untouched, {v["need"]["ai"]} AI</td></tr>' for t, v in lw.items())
           + "</table></div>")
    card = server_model_card()
    with st.expander("Model card: trained detectors (from the detector server)"):
        if card:
            st.markdown(card)
        else:
            md('<div class="muted small">Model card: not yet available from the detector server</div>')


@st.cache_data(ttl=300, show_spinner=False)
def server_model_card():
    return llm.model_card()


def results_page(case):
    res = load(case.id, mtime(case.id))
    inds, a, log = res["indicators"], res["assessment"], case.entries()
    tags = report.tagged(inds)
    not_run = {x["check"]: x["reason"] for x in log if x["action"] == "check_not_run"}
    steps(2)
    files = [p.name for p in case.evidence()]
    md(f'<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:4px"><h1 style="margin:0;font-size:1.9rem">'
       f'{e(case.id)}</h1>{pill(res["headline"])}</div><div class="muted small" style="margin-bottom:12px;overflow-wrap:anywhere">'
       + " · ".join(e(f) for f in files) + "</div>")
    for w in llm.server_warnings(log):
        st.warning(w, icon="⏳" if w.startswith("GPU server busy") else "📡")
    safe(lambda: overview(case, res, inds, a, tags), "overview")
    md('<div id="tabs" style="height:4px"></div>')
    key = f"tab_{case.id}"
    if key not in ss:
        ss[key] = qp.get("tab") if qp.get("tab") in TABS else "Evidence"
    tab = st.segmented_control("View", TABS, key=key, label_visibility="collapsed") or "Evidence"
    if ss.pop("_scroll", False):
        st.iframe("<script>window.parent.document.getElementById('tabs').scrollIntoView({behavior:'smooth'})</script>", height=1)
    {"Evidence": lambda: safe(lambda: evidence_tab(case, res, inds, tags, log), "evidence view"),
     "Cross-media": lambda: safe(lambda: cross_tab(inds, tags, not_run), "cross-media view"),
     "All findings": lambda: safe(lambda: findings_tab(case, inds, tags, res["coverage"]), "findings list"),
     "Custody": lambda: safe(lambda: custody_tab(case, res, log), "custody view"),
     "Calibration": lambda: safe(calibration_tab, "calibration view")}[tab]()


job = ss.get("job")
if pick == "new":
    upload_page()
elif job and job["case"] == pick:
    analyzing_page(job)
else:
    results_page(Case(pick))
