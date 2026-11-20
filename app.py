"""TruthLens forensic workbench. `python demo.py` (seeds the demo cases) or `streamlit run app.py`.
Pages: Overview · Analysis · Evidence viewer · Calibration · Chain of custody (plus New analysis and the live progress view).
Design system: ui/tokens.css (values) + ui/theme.css (styles) + ui/components.py (HTML components); see docs/UI_GUIDE.md.
Display only: nothing here changes an analysis result, weight, label or custody entry beyond the explicit Verify/Export logs."""
import base64, hashlib, io, json, logging, os, re, sys, threading, time, traceback
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ui import components as ui  # noqa: E402
from ui.components import e  # noqa: E402
from truthlens.vault import Case, CASES, case_dir, sha256  # noqa: E402
from truthlens import pipeline, report, fusion, llm, learned, public  # noqa: E402
from calibrate import MIN_OWN  # noqa: E402
from truthlens.analyzers import IMAGE_EXTS, TEXT_EXTS  # noqa: E402
from truthlens.analyzers.media_meta import readable_image  # noqa: E402

IMAGE = IMAGE_EXTS
VIDEO = (".mp4", ".mov", ".mkv", ".webm", ".avi")
AUDIO = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
LEVELS = [name for _, name in reversed(fusion.LEVELS)]  # No, Weak, Moderate, Strong (label strings)
WORD = dict(zip(LEVELS, ("No", "Weak", "Moderate", "Strong")))
LVL_TOKEN = {"No": "lvl-no", "Weak": "lvl-weak", "Moderate": "lvl-moderate", "Strong": "lvl-strong"}
GROUPS = {"Edited": "signs of local edits, cuts or re-encoding", "Generated": "signs the content was made by a generator",
          "Context": "the caption or claims disagree with the media"}
MEDIA_ICON = {"image": "image", "video": "movie", "audio": "graphic_eq", "text": "description", "cross": "link", "unsupported": "help"}
LAYER = {"ELA": ("_ela.png", "mark-video"), "Noise map": ("_noise.png", "mark-audio"), "Trained detector": ("_learned.png", "mark-detector")}
PAGES = ["Overview", "Analysis", "Evidence viewer", "Calibration", "Chain of custody"]
NAV_ICON = {"Overview": ":material/dashboard:", "Analysis": ":material/fact_check:", "Evidence viewer": ":material/center_focus_strong:",
            "Calibration": ":material/tune:", "Chain of custody": ":material/link:"}
PUBLIC = public.PUBLIC
MEDIA_ROWS = {"image": "Image", "video": "Video", "audio": "Audio", "cross": "Cross-media"}
HIDDEN = ("smoke_", "calib_", "probe_", "diag_", "pv_", "tmp_")
CASE_LIST_MAX = 8
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
    ("detector server", "The trained-detector server didn't answer, so the trained detectors were skipped. Check it, then analyse the case again."),
    ("detectors_url", "No trained-detector server is configured. Set DETECTORS_URL in .env, then analyse the case again."),
    ("no face", "No faces found, so lip-sync and face checks couldn't run. Try a clip where the speaker's face is visible."),
    ("no caption", "No caption was given. Add the caption or claims that came with the media to check them against it."),
    ("timed out", "A check took too long and the watchdog stopped it; the rest of the case finished. Try again, or a shorter clip."),
    ("no audio", "No audio track was found, so the audio checks couldn't run."),
    ("no video stream", "The file has no picture track, so it was analysed as audio."),
    ("not configured", "The model server isn't configured, so model checks were skipped. Set it up in .env, then analyse the case again."),
    ("unreachable", "The model server didn't answer, so model checks were skipped. Check the server, then analyse the case again."),
]

st.set_page_config("TruthLens", ":material/policy:", layout="wide")
st.markdown("<style>" + (ROOT / "ui" / "tokens.css").read_text(encoding="utf-8") + (ROOT / "ui" / "theme.css").read_text(encoding="utf-8")
            + "</style>", unsafe_allow_html=True)
qp, ss = st.query_params, st.session_state
cal = fusion.load_calibration()


# ---------- small helpers ----------
def md(s):
    st.markdown(public.scrub(s) if PUBLIC else s, unsafe_allow_html=True)


def oops(what, tb):
    """Error state: a plain message; the traceback folded under Details (public mode: a reference id, traceback in the log)."""
    if PUBLIC:
        ref = hashlib.sha256(f"{time.time()}{tb}".encode()).hexdigest()[:8]
        logging.getLogger("truthlens.app").error("ref %s: %s failed\n%s", ref, what, tb)
        md(ui.notice(f"Something went wrong (reference {ref}).", "danger"))
        return
    md(ui.notice(f"This {e(what)} couldn't be shown. The analysis itself is unaffected.", "danger"))
    with st.expander("Details"):
        st.code(tb[-3000:])


if fusion.missing():
    md(ui.notice(e(fusion.MISSING), "danger"))


@st.cache_resource
def _cleanup_started():
    public.start_cleanup()
    return True


if PUBLIC:
    _cleanup_started()
    sid = ss.setdefault("sid", public.new_session_id())
    ip = st.context.ip_address or "unknown"  # real visitor behind Tailscale Funnel (it sets X-Real-Ip/X-Forwarded-For; forged ones are overwritten, tested)
    if public.ACCESS_CODE and not ss.get("authed"):
        with st.container(key="gate"):
            md('<div class="brand"><span class="logo" aria-hidden="true">T</span><div><b>TruthLens</b>'
               '<div class="sub">Forensic workbench · public demo</div></div></div>')
            md(ui.notice(e(public.NOTICE), "info"))
            with st.form("gate_form", border=False):
                code = st.text_input("Access code", type="password")
                if st.form_submit_button("Enter", type="primary", width="stretch"):
                    ok, msg = public.check_code(code, ip)
                    if ok:
                        ss["authed"] = True
                        st.rerun()
                    md(ui.notice(e(msg), "danger"))
        st.stop()


def headline_word(headline):
    """Short word for a headline: the four levels, else Limited / Failed / Not analysed (no level colour)."""
    if headline.startswith("Failed"):
        return None, "Failed"
    if headline.startswith("Limited"):
        return None, "Limited"
    return (WORD[headline], WORD[headline]) if headline in WORD else (None, "Not analysed")


def level_html(headline, short=False, size="sm"):
    w, word = headline_word(headline)
    text = word if short else ("Limited analysis" if word == "Limited" else headline)
    return ui.level_label(w, text, size)


def term(t):
    return TERMS.get(t, (t.replace("_", " "), t))


def cap(t):
    return t[:1].upper() + t[1:]


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


def light_table(inner, caption="", legend=None):
    md(f'<div class="lt">{inner}' + (ui.legend(legend) if legend else "")
       + (f'<div class="cap">{caption}</div>' if caption else "") + "</div>")


def safe(fn, what):
    """Render a panel; on an unexpected error show the error state, never a raw traceback."""
    try:
        fn()
    except Exception:
        logging.getLogger("truthlens.app").exception("panel failed: %s", what)  # full traceback in the dashboard log
        oops(what, traceback.format_exc())


def hint_for(reason):
    return next((h for k, h in HINTS if k in reason.lower()), None)


def hints(reasons):
    """Distinct next-step hints for a set of not-run reasons."""
    return list(dict.fromkeys(h for h in map(hint_for, reasons) if h))


def short_reason(r):
    r = r.lower()
    for k, s in (("no face", "no face visible"), ("no caption", "no caption given"), ("timed out", "timed out"),
                 ("no audio", "no audio track"), ("not configured", "model server not set up"),
                 ("unreachable", "model server offline"), ("detector server", "detector server offline")):
        if k in r:
            return s
    return r.split(" (")[0][:60]


def check_name(c):
    if " analysis · " in c:
        kind, f = c.split(" analysis · ", 1)
        return f"{kind} checks on {f}"
    return CHECK_NAMES.get(c, c.replace("_", " "))


def ticks(dur, n=8):
    """Distinct, round tick values for a time axis of `dur` seconds."""
    step = next((s for s in (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300) if dur / s <= n), 600)
    return [k * step for k in range(int(dur // step) + 1)]


def tick_label(v):
    return f"{v:g} s"


def first_sentence(t):
    """Up to the first ". " that is outside quotes (a quoted caption may contain full stops)."""
    for m in re.finditer(r"\. ", t):
        head = t[:m.start()]
        if head.count('"') % 2 == 0 and head.count("'") % 2 == 0:
            return head + "."
    return t


def page_head(title, desc="", actions=None):
    """Page header: title + one-line description on the left, the page's actions on the right (breadcrumb is in the top bar)."""
    with st.container(key="pagehead", horizontal=True, vertical_alignment="bottom"):
        md(ui.page_header(title, desc))
        if actions:
            with st.container(horizontal=True, key="pageactions", gap="small"):
                actions()


def footer():
    md('<div class="foot"><span>TruthLens · Team UNEX · InnovEdge 2026</span><span>Automated indicators, not proof.</span></div>')


# ---------- findings ----------
def state_of(i):
    if not fusion.calibrated(i):
        return "uncalibrated"
    return "counted" if fusion.contribution(i) > 0 else "zero"


def finding_level(i):
    return WORD[fusion.label(fusion.contribution(i))] if state_of(i) == "counted" else None


def score_text(i):
    if fusion.calibrated(i):
        return f"score {i.score:.2f} × weight {fusion.weight(i):.2f} = {fusion.contribution(i):.2f}"
    thr = f" ≥ threshold {i.location['threshold']:.2f}" if "threshold" in i.location else ""
    return f"score {i.score:.2f}{thr} · {fusion.UNCALIBRATED}"


def finding_html(t, i, focused=False, short=False):
    name, tip = term(i.type)
    g = fusion.group(i)
    rel = " ↔ " + ", ".join(i.related) if i.related else ""
    det = (f"<p><b>Where</b> {e(i.file)}{e(rel)} · {e(where(i.location))}</p><p><b>Score</b> <span class='mono'>{e(score_text(i))}</span></p>"
           f"<p><b>Type</b> {g}: {GROUPS[g]}</p><p><b>Could also be</b> {e('; '.join(i.alt_explanations))}</p>"
           f"<p><b>What to check</b> {e('; '.join(report.recommended(i)))}</p>")
    return ui.finding_row(t, cap(name), i.modality, first_sentence(i.explanation) if short else i.explanation, state_of(i),
                          finding_level(i), details="" if short else det, focused=focused, title_tip=tip)


def not_run_html(c, why):
    return ui.finding_row("–", cap(check_name(c)), check_media(c), cap(short_reason(why)), "not_run")


# ---------- components that draw evidence ----------
def film_strip(frames, hits, focus_t=None, n=8):
    """Keyframes as a film strip: timestamps under each, frames with a finding outlined in its level colour."""
    show = [frames[k] for k in sorted(set(np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)))]
    cells = []
    for f in show:
        t = int(f.stem[-4:]) - 1
        tok = hits.get(t) or ("accent" if focus_t is not None and abs(focus_t - t) < 0.5 else None)
        cells.append(f'<figure class="{"hit" if tok else ""}" style="--fc:var(--{tok or "border"})"><img src="{img_uri(Image.open(f), 240, 75)}" '
                     f'alt="Frame at {t} s"><figcaption>{t} s{" · finding" if tok else ""}</figcaption></figure>')
    return '<div class="film" role="list" aria-label="Keyframes">' + "".join(cells) + "</div>"


def custody_stamp(case, log, res):
    ingests = [x for x in log if x["action"] == "ingest"]
    t0 = ui.fdt(ingests[0]["ts"]) if ingests else "–"
    ok, bad = verified(case, res, log), not all(res["integrity"].values())
    hashes = "".join(f'<div class="v">{e(x["file"][:48])} · {ui.hash_short(x["sha256"])}</div>' for x in ingests)
    state = (ui.chip("Integrity verified", "success", "verified") if ok else ui.chip("Hash mismatch", "danger", "error") if bad
             else ui.chip("Not yet verified", "neutral", "radio_button_unchecked"))
    md(f'<div class="stamp" role="group" aria-label="Custody stamp"><div class="eyebrow">Case</div><div class="v">{e(case.id)}</div>'
       f'<div class="eyebrow">Evidence</div>{hashes}<div class="eyebrow">Intake</div><div class="v">{t0}</div>'
       f'<div style="margin-top:var(--s-3)">{state}</div></div>')


def gauge(label, p):
    k = LEVELS.index(label)
    segs = []
    for n in range(4):
        a0, a1 = np.pi - n * np.pi / 4 - 0.03, np.pi - (n + 1) * np.pi / 4 + 0.03
        x0, y0, x1, y1 = 150 + 110 * np.cos(a0), 140 - 110 * np.sin(a0), 150 + 110 * np.cos(a1), 140 - 110 * np.sin(a1)
        segs.append(f'<path d="M{x0:.1f},{y0:.1f} A110,110 0 0 1 {x1:.1f},{y1:.1f}" style="stroke:var(--{LVL_TOKEN[WORD[LEVELS[n]]]})" '
                    f'stroke-width="22" fill="none" opacity="{1 if n == k else 0.16}"/>')
    ang = np.pi - min(p, 1) * np.pi
    nx, ny = 150 + 84 * np.cos(ang), 140 - 84 * np.sin(ang)
    return (f'<svg viewBox="0 0 300 168" width="100%" role="img" aria-label="Gauge: {e(label)}">' + "".join(segs)
            + f'<line x1="150" y1="140" x2="{nx:.1f}" y2="{ny:.1f}" style="stroke:var(--text)" stroke-width="3" stroke-linecap="round"/>'
            f'<circle cx="150" cy="140" r="6" style="fill:var(--text)"/><text x="28" y="164" class="t-muted">No</text>'
            f'<text x="230" y="164" class="t-muted">Strong</text></svg>')


def label_hero(res, inds, a):
    """Gauge, the label in words, what it mostly comes from, how sure the setup is."""
    head = res["headline"]
    if head.startswith(("Not analysed", "Failed")):
        why = "The analysis did not complete" if res.get("failed") else "No check could run on this evidence"
        md(f'<div class="hero"><div class="t">{ui.level_label(None, "Analysis did not complete" if res.get("failed") else "Not analysed", "lg")}'
           f'<div>{why}, so no label is given. The evidence files and their custody log are kept.</div></div></div>')
        return
    lab, p = a["case"]["label"], a["case"]["p"]
    contrib = sorted(fusion.contributions(inds).items(), key=lambda kv: -kv[1])
    if contrib and contrib[0][1] > 0:
        (f, ty), _ = contrib[0]
        best = max((i for i in inds if i.file == f and i.type == ty), key=fusion.contribution)
        mostly = (f'Mostly from <b>{fusion.group(best)}</b>: <span title="{e(term(ty)[1])}">{e(term(ty)[0])}</span> '
                  f'{e(at(best.location))}' + (f' <span class="muted">({e(f)})</span>' if len(res["integrity"]) > 1 else ""))
    else:
        mostly = "Nothing counted toward the label: no finding carries a calibrated weight."
    limited = f'<div class="muted">{e(head)}</div>' if head == fusion.LIMITED else ""
    sure = ""
    if cal:
        h = cal["held_out"]
        sure = (f'<div class="meta">How sure? On the held-out test split this setup flagged {ui.fpct(h["hit_rate"][0])} of manipulated '
                f'files and {ui.fpct(h["false_alarm_rate"][0])} of untouched ones ({e(cal["dataset"])} test set, n={cal["n"]}).</div>')
    creds = f'<div class="meta">{e(report.CREDENTIALS_NOTE)}</div>' if ss.get("_creds") else ""
    md(f'<div class="hero"><div class="g">{gauge(lab, p)}</div><div class="t">'
       f'{ui.level_label(WORD[lab], WORD[lab] + " indicators", "lg")}{limited}<div class="mostly">{mostly}</div>'
       + ui.notice("Automated indicators, not proof. Review the underlying evidence and benign explanations.", "warning")
       + f'<div class="meta mono">Fused strength {p:.2f} · label of manipulation or synthetic generation</div>{creds}{sure}</div></div>')


# ---------- case list ----------
def case_dirs():
    if PUBLIC:  # the demo cases and this browser session's own cases only
        return [case_dir(c) for c in public.session_cases(ss["sid"]) + list(public.DEMOS) if (case_dir(c) / "result.json").exists()]
    if not CASES.exists():
        return []
    ds = [d for d in CASES.iterdir() if (d / "result.json").exists() and not d.name.startswith(HIDDEN)]
    return sorted(ds, key=lambda d: -d.stat().st_mtime)


def mtime(case_id):
    return (case_dir(case_id) / "result.json").stat().st_mtime


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
    kind = next((("video" if p.suffix.lower() in VIDEO else "audio" if p.suffix.lower() in AUDIO else "text"
                  if p.suffix.lower() in TEXT_EXTS else "image") for p in ev), "unsupported")
    return {"files": [p.name for p in ev], "headline": r["headline"], "ts": ing, "thumb": thumb, "kind": kind}


@st.cache_data(show_spinner=False)
def report_html(case_id, mtime, log_size):
    c = Case(case_id)
    out = report.render(c, load(case_id, mtime))
    return public.scrub(out) if PUBLIC else out


@st.cache_data(ttl=30, show_spinner=False)
def services():
    return llm.services()


@st.fragment(run_every=30)
def status_pill():
    """System online when every configured service answers, degraded when some do, offline when none; words and icon carry
    the state, not only colour. Host names only outside public mode."""
    sv = services()
    up = [k for k, v in sv.items() if v]
    if sv and len(up) == len(sv):
        kind, ico, text = "success", "check_circle", "System online"
    elif up:
        kind, ico, text = "warning", "warning", "System degraded: " + ", ".join(k for k, v in sv.items() if not v) + " offline"
    else:
        kind, ico, text = "neutral", "cloud_off", "System offline" + ("" if sv else ": no services configured")
    tip = "; ".join(f"{k}: {'online' if v else 'offline'}" for k, v in sv.items())
    if not PUBLIC and "ollama" in llm.PROVIDERS:
        tip += f" · GPU host {llm.host('ollama')}"
    md(ui.pill(text, kind, ico, tip))


def topbar(page):
    with st.container(key="topbar", horizontal=True, vertical_alignment="center"):
        md(f'<nav class="crumb" aria-label="Breadcrumb">Workspace / <b>{e(page)}</b></nav>')
        status_pill()
        if PUBLIC:
            md(ui.pill("Demo environment", "info", "public"))


def open_case(cid):
    qp.clear()
    qp["case"] = cid
    ss["page"] = "Overview"


known = [d.name for d in case_dirs()]
pick = qp.get("case") or (known[0] if known else "new")
if PUBLIC and pick != "new" and not public.visible(pick, ss["sid"]):
    pick = "new"
if pick != "new" and not (case_dir(pick) / "result.json").exists() and not (ss.get("job", {}).get("case") == pick):
    pick = "new"


def nav_page():
    p = ss.get("page") or qp.get("page")
    return p if p in PAGES else "Overview"


with st.sidebar:
    job = ss.get("job")
    md('<div class="brand"><span class="logo" aria-hidden="true">T</span><div><b>TruthLens</b>'
       '<div class="sub">Forensic workbench</div></div></div>')
    md('<div class="eyebrow side-h">Workspace</div>')
    cur = "Calibration" if nav_page() == "Calibration" else "New analysis" if pick == "new" else nav_page()
    with st.container(key="nav"):
        for n in PAGES:
            off = pick == "new" and n not in ("Overview", "Calibration") and not known
            if st.button(n, key=f"nav_{n}", icon=NAV_ICON[n], type="primary" if n == cur else "tertiary", width="stretch",
                         disabled=off, help="Open a case first" if off else None):
                if pick == "new" and n != "Calibration" and known:
                    open_case(known[0])
                ss["page"] = n
                qp["page"] = n
                st.rerun()
    if pick != "new":
        running = bool(job and job["case"] == pick and not job.get("done"))
        m = (case_meta(pick, mtime(pick)) if pick in known else
             {"files": [p.name for p in Case(pick).evidence()], "headline": "", "ts": time.time()})
        state = (ui.chip("Analysing", "info", "progress_activity") if running else ui.chip("Failed", "danger", "error")
                 if m["headline"].startswith("Failed") else ui.chip("Active case", "success", "check"))
        name = m["files"][0] if m["files"] else pick
        md(f'<div class="eyebrow side-h">Current case</div><div class="casecard"><div>{state}</div>'
           f'<div class="nm">{e(name)}</div><div class="mono muted">{e(pick)}</div>'
           f'<div class="muted">{len(m["files"])} media {"file" if len(m["files"]) == 1 else "files"} · {ui.fdate(m["ts"])}</div></div>')
    if job and not job.get("done") and pick != job["case"]:
        md(ui.notice(f"Analysing {e(job['case'])}…", "info", "progress_activity"))
        if st.button("Back to the running analysis", width="stretch", icon=":material/arrow_back:"):
            open_case(job["case"])
            st.rerun()
    if st.button("New analysis", key="nav_new", icon=":material/add:", type="secondary", width="stretch"):
        open_case("new")
        st.rerun()
    md(f'<div class="eyebrow side-h">Cases · {len(known)}</div>')
    if not known:
        md(ui.empty_state("folder_open", "No cases yet", "Start with New analysis."))
    show_all = ss.get("all_cases", False)
    for cid in known if show_all else known[:CASE_LIST_MAX]:
        m = case_meta(cid, mtime(cid))
        c1, c2 = st.columns([1, 3], vertical_alignment="center", gap="small")
        if m["thumb"]:
            c1.image(m["thumb"], width="stretch")
        else:
            c1.markdown(f'<div class="thumb-ph">{ui.icon(MEDIA_ICON[m["kind"]])}</div>', unsafe_allow_html=True)
        name = m["files"][0] if m["files"] else cid
        name = (name if len(name) <= 24 else name[:22] + "…") + (f" +{len(m['files']) - 1}" if len(m["files"]) > 1 else "")
        if c2.button(name, key=f"open_{cid}", type="primary" if cid == pick else "tertiary", help=cid, width="stretch"):
            open_case(cid)
            st.rerun()
        c2.markdown(f'<div class="caseitem">{level_html(m["headline"], short=True)}<span class="mono">{ui.fshort(m["ts"])}</span></div>',
                    unsafe_allow_html=True)
    if len(known) > CASE_LIST_MAX and st.button("Show fewer cases" if show_all else f"Show all {len(known)} cases", key="cases_all",
                                                type="tertiary", width="stretch"):
        ss["all_cases"] = not show_all
        st.rerun()
    md(f'<div class="side-foot"><b>Team UNEX · InnovEdge 2026</b><div>{"Public demo" if PUBLIC else "Local workspace"}</div></div>')


# ---------- 1. new analysis ----------
def upload_page():
    page_head("New analysis", "Evidence is hashed (SHA-256) on arrival, stored read-only, and every step is logged.")
    if PUBLIC:
        md(ui.notice(e(public.NOTICE), "warning"))
    with st.container(key="card_upload"):
        files = st.file_uploader("Evidence files (several at once is fine)", accept_multiple_files=True)
        md('<div class="muted" style="font-size:var(--fs-xs)">'
           + (e(public.LIMITS) + "<br>" if PUBLIC else "")
           + "Images: JPG, PNG, WebP, TIFF, BMP, GIF, AVIF, HEIC, JXL · Video: MP4, MOV, MKV, WebM, AVI · Audio: WAV, MP3, M4A, FLAC, "
           "OGG · Text: TXT, MD</div>")
        if files:
            rows = []
            for f in files:
                n = f.name.lower()
                kind = ("image" if n.endswith(IMAGE) else "video" if n.endswith(VIDEO) else "audio" if n.endswith(AUDIO)
                        else "text" if n.endswith(TEXT_EXTS) else "unsupported")
                rows.append([f'{ui.icon(MEDIA_ICON[kind])} {e(f.name)}', e(kind), ui.hash_short(hashlib.sha256(f.getvalue()).hexdigest())])
            md(ui.table(["File", "Type", "SHA-256"], rows))
        context = st.text_area("Caption or claims (optional)", height=96,
                               placeholder='e.g. Flood at the campus library on 2026-11-02. She said "we cancelled the fest".')
        with st.expander("Paste text evidence (optional): a message, post or article to examine"):
            pasted = st.text_area("Text evidence", label_visibility="collapsed",
                                  placeholder="Stored as pasted_text.txt with its own hash; its claims are checked against the media.")
        empty = not files and not pasted.strip()
        fresh = not PUBLIC and st.checkbox("Ask the models again (ignore cached answers)")
        go_ = st.button("Start analysis", type="primary", disabled=empty, icon=":material/play_arrow:")
        if empty:
            md('<div class="muted">Add at least one file, or paste text, to start.</div>')
    if go_:
        uploads = {f.name: f.getvalue() for f in files or []}
        if pasted.strip():
            uploads["pasted_text.txt"] = pasted.strip().encode("utf-8")
        if PUBLIC:
            if problems := public.upload_problems(list(uploads.items()), ss["sid"], ip):
                for x in problems:
                    md(ui.notice(e(x), "danger"))
                st.stop()
            public.count_case(ss["sid"], ip)
        case = Case(public.new_case_id(ss["sid"]) if PUBLIC else None)
        for n, b in uploads.items():
            if sha256(case.ingest(n, b)) != hashlib.sha256(b).hexdigest():
                md(ui.notice(f"The stored copy of {e(n)} does not match the upload. Please upload it again.", "danger"))
                st.stop()
        job = {"case": case.id, "context": context, "fresh": fresh, "ev": [], "t0": time.time()}
        threading.Thread(target=work, args=(job,), daemon=True).start()
        ss["job"] = job
        open_case(case.id)
        st.rerun()
    footer()


def work(job):
    """The analysis, off the page thread, so the page can show live progress and elapsed time."""
    if PUBLIC:
        public.wait_turn(job["case"])
    llm.FRESH = job["fresh"]
    try:
        pipeline.run(Case(job["case"]), job["context"], lambda step, state: job["ev"].append((time.time(), step, state)))
    except Exception:
        job["error"] = traceback.format_exc()
    finally:
        llm.FRESH = False
        job["done"] = time.time()
        if PUBLIC:
            public.done(job["case"])


# ---------- 2. analysing ----------
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
    for step, s in state.items():
        secs = (t_off.get(step, now) - t_on[step]) if step in t_on else None
        note = ""
        if s == "error":
            note = "stopped by the watchdog; the case continues" if "timed out" in errs.get(step, "") else "the case continues without it"
        kind, _, f = step.partition(" · ")
        rows.append(ui.progress_row(s, STEP_NAMES.get(kind, cap(kind)) + (f" · {f}" if f else ""), secs, note))
    for x in case.entries():  # checks that could not run, as soon as they are logged
        if x["action"] == "check_not_run":
            rows.append(ui.progress_row("not_run", cap(check_name(x["check"])), None, short_reason(x["reason"])))
    head = (f'<div class="cardhead">{ui.section("Analysis complete" if job.get("done") else "Analysing")}'
            f'<span class="mono muted">{ui.fdur(now - job["t0"])} elapsed</span></div>')
    return (f'<div class="card" role="status" aria-live="polite">{head}'
            + ("".join(rows) or '<div class="muted pulse">Reading the files…</div>') + "</div>")


def analyzing_page(job):
    case = Case(job["case"])
    page_head("Analysing case", f"Case {case.id}. Checks run per file; findings appear when the case is complete.")
    ev = case.evidence()
    c1, c2 = st.columns([1, 2])
    with c1:
        p = next((p for p in ev if p.suffix.lower() in IMAGE), None)
        kind = "video" if any(q.suffix.lower() in VIDEO for q in ev) else "audio"
        try:
            inner = f'<img class="ev" src="{img_uri(Image.open(p), 600)}" alt="{e(p.name)}">' if p else ui.empty_state(MEDIA_ICON[kind], kind.title())
        except OSError:
            inner = ui.empty_state("image", "Image")
        md(f'<div class="lt"><div class="{"" if job.get("done") else "pulse"}">{inner}</div><div class="cap">'
           + " · ".join(e(q.name) for q in ev) + "</div></div>")
    with c2:
        live_progress()


@st.fragment(run_every=0.5)
def live_progress():
    """Only this box re-runs while the analysis works, so the rest of the page stays put; switches to the case when done."""
    job = ss.get("job")
    if not job:
        return
    case = Case(job["case"])
    if PUBLIC and (q := public.position(job["case"])):
        md(ui.notice(f'Queued: {q} {"analysis" if q == 1 else "analyses"} ahead of yours. Analyses run one at a time.', "info", "schedule"))
    md(progress_html(job, case))
    if not job.get("done"):
        return
    if not job.get("error"):
        ss.pop("job", None)
        st.rerun(scope="app")
    md(ui.notice("The analysis stopped unexpectedly. The case and its evidence are kept, and it is marked failed.", "danger"))
    oops("analysis", job["error"])
    if st.button("Open the case", type="primary"):
        ss.pop("job", None)
        st.rerun(scope="app")


# ---------- 3. case pages ----------
def check_media(c):
    """Media row of a coverage check: "image analysis · x.jpg" -> image, "trained detector (audio)" -> audio, else cross."""
    if " analysis · " in c:
        k = c.split(" analysis · ")[0]
        return k if k in MEDIA_ROWS else "cross"
    if c.startswith("trained detector ("):
        return c[len("trained detector ("):-1]
    return {"transcript": "audio", "visual artifacts (model)": "image", "exif": "image", "lipsync": "video",
            "face_region_flicker": "video"}.get(c, "cross")


def read_only(case):
    return PUBLIC and case.id in public.DEMOS


def verify_now(case):
    """Re-hash the evidence; logged to the custody log (public demo cases: kept in this session, the case stays untouched)."""
    v = case.verify()
    if read_only(case):
        ss[f"verify_{case.id}"] = {"ts": time.time(), "action": "verify", **{k: ("ok" if ok else "MISMATCH") for k, ok in v.items()}}
    else:
        case.log("verify", **{k: ("ok" if ok else "MISMATCH") for k, ok in v.items()})


def verified(case, res, log):
    """True only after an explicit re-hash (Verify) whose newest result is all ok, and the files still match now."""
    v = [x for x in log if x["action"] == "verify"] + [x for x in [ss.get(f"verify_{case.id}")] if x]
    return (bool(v) and all(val == "ok" for k, val in v[-1].items() if k not in ("ts", "action"))
            and all(res["integrity"].values()))


def export_button(case, key, primary=False):
    log_size = case.log_path.stat().st_size if case.log_path.exists() else 0
    if st.download_button("Export report", report_html(case.id, mtime(case.id), log_size), f"{case.id}.html", "text/html",
                          key=key, icon=":material/download:", type="primary" if primary else "secondary",
                          help="Self-contained HTML report; print it to PDF from the browser") and not read_only(case):
        case.log("report_export", format="html")


def new_button(key):
    if st.button("New analysis", type="primary", key=key, icon=":material/add:"):
        open_case("new")
        st.rerun()


def overview(case, res, inds, a, tags, log):
    def actions():
        export_button(case, "exp_top")
        new_button("new_top")
    page_head("Forensic overview", "Your evidence, findings and case integrity in one place.", actions)
    if until := public_expiry(case):
        md(ui.notice(f"This case will be deleted on {e(until)}.", "warning", "schedule"))
    for w in llm.server_warnings(log):
        md(ui.notice(e(public.scrub(w) if PUBLIC else w), "warning"))
    sv = services()
    if sv and not any(sv.values()):
        md(ui.notice("The model server is offline. Results already analysed are shown; new model checks come from the cache "
                     "or are recorded as not run.", "info", "cloud_off"))
    files = [p.name for p in case.evidence()]
    cov = res["coverage"]
    failed = res["headline"].startswith(("Not analysed", "Failed"))
    n_ok = sum(1 for f in cov.get("files", {}).values() if f["analysed"])
    state = "failed" if res.get("failed") else "all analysed" if n_ok >= len(files) else f"{len(files) - n_ok} not analysed"
    counted = [i for i in inds if state_of(i) == "counted"]
    pairs = list(cal.get("detectors", {}).values()) if cal else []
    n_cal = sum(1 for d in pairs if d.get("weight", 0) > 0)
    bad = [f for f, v in res["integrity"].items() if not v]
    with st.container(key="kpis"):
        k1, k2, k3, k4 = st.columns(4)
        with k1.container(key="kpi_media"):
            md(ui.kpi("Media analysed", f"{len(files)}", e(f"{'file' if len(files) == 1 else 'files'} · {state}")))
        with k2.container(key="kpi_findings"):
            md(ui.kpi("Findings", f"{len(inds)}", "", ui.chip(f"{len(counted)} counted in the label", "warning" if counted else "neutral")))
        with k3.container(key="kpi_checks"):
            md(ui.kpi("Calibrated checks", f"{n_cal}", e(f"of {len(pairs)} check and media pairs have a measured weight")))
        with k4.container(key="kpi_integrity"):
            if bad:
                md(ui.kpi("Integrity", "Mismatch", e(", ".join(bad)), ui.chip("Hash changed", "danger", "error")))
            elif verified(case, res, log):
                md(ui.kpi("Integrity", "Verified", "", ui.chip("SHA-256 match", "success", "verified")))
            else:
                md(ui.kpi("Integrity", "Not verified", "Re-hash the evidence against its intake SHA-256."))
                if st.button("Verify", key="verify_kpi", width="stretch", icon=":material/fingerprint:"):
                    verify_now(case)
                    st.rerun()
    creds = report.content_credentials(case, inds)
    ss["_creds"] = bool(creds)
    left, right = st.columns([3, 2])
    with left, st.container(key="card_assess"):
        md(ui.section("Case label"))
        for c in creds:
            md(ui.notice(e(c), "provenance"))
        label_hero(res, inds, a)
        if failed:
            md(ui.notice("Start a <b>New analysis</b> with the same file. Its evidence and custody log stay here.", "info", "arrow_forward",
                         "Next step"))
            if not PUBLIC:
                with st.expander("Details"):
                    st.code(res.get("failed") or cov.get("why") or "")
    with right, st.container(key="card_cov"):
        md(ui.section("Check coverage", "checks run / checks that apply"))
        allc = {c: None for c in cov["ran"]} | cov["not_run"]
        rows = []
        for m, name in MEDIA_ROWS.items():
            mine = [c for c in allc if check_media(c) == m]
            if mine:
                rows.append(ui.coverage_row(name, sum(1 for c in mine if c not in cov["not_run"]), len(mine)))
        why = [f"{cap(check_name(c))}: {short_reason(w)}" for c, w in cov["not_run"].items()]
        md("".join(rows) + (f'<div class="eyebrow" style="margin-top:var(--s-2)">Not run</div><ul class="plain">'
                            + "".join(f"<li>{e(w)}</li>" for w in why) + "</ul>" if why else '<div class="muted">Every check that applies ran.</div>'))
        for h in hints(cov["not_run"].values()):
            md(ui.notice(e(h), "info", "lightbulb"))
    with st.container(key="card_top"):
        with st.container(horizontal=True, vertical_alignment="center", key="tophead"):
            md(ui.section("Top findings", "strongest first"))
            if st.button("View all", key="viewall", icon=":material/arrow_forward:", type="tertiary"):
                go("Analysis")
        errs = [x for x in log if x["action"] == "analyzer_error"]
        rows = tags[:5]
        if not rows and not errs:
            md(ui.empty_state("search_off", "No findings for this case", "No findings does not show the media is unaltered."))
        for t, i in rows:
            cnt = state_of(i) == "counted"
            with st.container(key=f"frow_{t}"):
                r1, r2 = st.columns([6, 1], vertical_alignment="center")
                r1.markdown(finding_html(t, i, short=True), unsafe_allow_html=True)
                if r2.button("Review" if cnt else "Inspect", key=f"{'act_review' if cnt else 'act_inspect'}_{t}", width="stretch",
                             icon=":material/visibility:",
                             help=("Counted in the label" if cnt else cap(fusion.UNCALIBRATED)) + f". Opens {t} in the evidence viewer."):
                    ss["focus"] = {"case": case.id, "tag": t, "file": i.file if i.file != "context" else None}
                    go("Evidence viewer")
        for x in errs[:max(0, 5 - len(rows))]:
            md(ui.finding_row("–", f'{cap(x["analyzer"].rsplit(".", 1)[-1])} checks', x["file"],
                              "The check failed; the case continued without it.", "failed"))
    footer()


def analysis_page(case, res, inds, tags, not_run):
    page_head("Analysis", "Recommended checks, the summary, cross-media links and every finding of this case.",
              lambda: export_button(case, "exp_analysis"))
    c1, c2 = st.columns([3, 2])
    with c1, st.container(key="card_checks"):
        md(ui.section("Recommended human checks"))
        checks = report.human_checks(inds)[:5]
        md('<ol class="plain">' + ("".join(f"<li>{e(c)}</li>" for c in checks) or "<li>Spot-check the original files yourself.</li>")
           + "</ol>")
    with c2, st.container(key="card_summary"):
        md(ui.section("Summary", "written from the findings only") + f'<p>{e(report.summary(res, case))}</p>')
    with st.container(key="card_cross"):
        md(ui.section("Cross-media", "one lane per source on a shared time axis"))
        safe(lambda: cross_tab(inds, tags, not_run), "cross-media view")
    with st.container(key="card_all"):
        md(ui.section("All findings"))
        safe(lambda: findings_tab(case, inds, tags, res["coverage"]), "findings list")
    footer()


def has_picture(log, fname, ext):
    """Does the file have a picture track? From the routing log (an audio-only .mp4 has none); older cases: by extension."""
    r = next((x for x in log if x["action"] == "routed" and x["file"] == fname), None)
    return ext in VIDEO and (r is None or any(a.endswith(".video") for a in r["analyzers"]))


def evidence_tab(case, res, inds, tags, log):
    names = [p.name for p in case.evidence()]
    if not names:
        md(ui.empty_state("folder_off", "This case has no evidence files"))
        return
    focus = ss.get("focus") if (ss.get("focus") or {}).get("case") == case.id else None
    key = f"file_{case.id}"
    if focus and focus.get("file") in names:
        ss[key] = focus["file"]
    elif key not in ss:
        ss[key] = qp.get("file") if qp.get("file") in names else names[0]
    path = case.dir / "evidence" / (ss[key] if len(names) > 1 else names[0])
    fname = path.name
    ext, stem = path.suffix.lower(), path.stem
    mine = [(t, i) for t, i in tags if i.file == fname]
    fi = next((i for t, i in tags if focus and t == focus["tag"]), None)
    digest = next((x["sha256"] for x in log if x["action"] == "ingest" and x["file"] == fname), "?")
    with st.container(key="toolbar_file"):
        c1, c2 = st.columns([2, 3], vertical_alignment="center")
        with c1:
            if len(names) > 1:
                st.selectbox("Evidence file", names, key=key)
            else:
                md(f'<div class="eyebrow">Evidence file</div><div>{e(fname)}</div>')
        c2.markdown(f'{level_html(fusion.headline(res["assessment"], res["coverage"], fname), short=True)} &nbsp; SHA-256 {ui.hash_short(digest)}'
                    + (f' &nbsp; {ui.chip("Focused on " + focus["tag"], "accent", "center_focus_strong")}' if fi is not None and fi.file == fname else ""),
                    unsafe_allow_html=True)
    if conv := next((x for x in log if x["action"] == "converted" and x["file"] == fname), None):
        md(ui.notice(f'Shown and analysed via {e(conv["to"])} (converted with {e(conv["tool"])}): compression-based checks (ELA) see '
                     "the conversion, not the original. The hash above is the original file.", "info"))
    if ext in IMAGE:
        image_view(case, path, stem, mine, fi)
    if has_picture(log, fname, ext):
        video_view(case, path, stem, mine, fi)
    if ext in TEXT_EXTS:
        md('<div class="card mono" style="white-space:pre-wrap;overflow-wrap:anywhere">'
           + e(path.read_text(encoding="utf-8", errors="replace")[:20000]) + "</div>")
    if ext in AUDIO or ext in VIDEO:
        audio_view(case, path, stem, mine, fi, log)
    md(ui.section("Findings in this file", f"{len(mine)} · strongest first"))
    if mine:
        md('<div class="stack">' + "".join(finding_html(t, i, focused=fi is i) for t, i in mine) + "</div>")
    else:
        md(ui.empty_state("search_off", "No findings for this file", "That does not show it is unaltered: review it yourself."))
    context_view(res, log, stem, case)


def brackets(d, box, color, label, width, size):
    """Crosshair marker: thin corner brackets (camera focus box) with the finding ID above them."""
    x, y, w, h = box
    k = max(6, min(w, h) // 4)
    for cx, cy, sx, sy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        d.line([cx, cy, cx + sx * k, cy], fill=color, width=width)
        d.line([cx, cy, cx, cy + sy * k], fill=color, width=width)
    font = ImageFont.load_default(size)
    l, t, r, b = d.textbbox((0, 0), label, font=font)
    top = max(0, y - (b - t) - 8)
    d.rectangle([x, top, x + r - l + 8, top + b - t + 6], fill=ui.rgb("bg") + (230,))
    d.text((x + 4 - l, top + 3 - t), label, fill=color, font=font)


def mark_token(i, focused):
    """Overlay colour of a finding: the same level colour as in the findings list; uncalibrated ones in info, focused in accent."""
    if focused:
        return "accent"
    return LVL_TOKEN[finding_level(i)] if finding_level(i) else "info"


def image_view(case, path, stem, mine, fi):
    avail = [k for k, (suf, _) in LAYER.items() if case.artifact_path(stem + suf).exists()]
    cm = case.artifact_path(stem + "_copymove.png")
    with st.container(key="toolbar"):
        k1, k2, k3 = st.columns([3, 2, 2], vertical_alignment="center")
        layers = k1.multiselect("Overlay layers", avail, default=avail[1:2] or avail[:1], key=f"lay_{case.id}_{stem}")
        opacity = k2.slider("Overlay opacity", 0.0, 1.0, 0.6, key=f"op_{case.id}_{stem}")
        boxes = k3.toggle("Finding regions", True, key=f"bx_{case.id}_{stem}")
        show_cm = cm.exists() and k3.toggle("Copy-move pairs", True, key=f"cm_{case.id}_{stem}")
    try:
        base = Image.open(readable_image(path, case) or path).convert("RGBA")
    except (OSError, ValueError):
        md(ui.notice("This image can't be displayed here (unreadable format). Its findings are listed below.", "info"))
        return
    out = base.copy()
    for k in layers:
        suf, tok = LAYER[k]
        h = np.asarray(Image.open(case.artifact_path(stem + suf)).convert("L").resize(base.size), np.float32)
        h = np.clip(h / max(np.percentile(h, 99.5), 1), 0, 1)
        lay = Image.new("RGBA", base.size, ui.rgb(tok) + (0,))
        lay.putalpha(Image.fromarray((h * 255 * opacity).astype(np.uint8)))
        out = Image.alpha_composite(out, lay)
    if boxes:
        d = ImageDraw.Draw(out)
        lw = max(2, base.width // 400)
        for t, i in mine:
            for kb in ("bbox", "bbox2"):
                if kb in i.location:
                    brackets(d, i.location[kb], ui.rgb(mark_token(i, fi is i)), t, lw * (2 if fi is i else 1), max(13, base.width // 45))
    legend = [(LAYER[k][1], k) for k in layers] + ([("lvl-moderate", "Finding region (level colour)"), ("info", "Uncalibrated finding")]
                                                   if boxes else [])
    v1, v2 = st.columns(2)
    with v1:
        light_table(f'<img class="ev" src="{img_uri(out)}" alt="Analysis view of {e(path.name)} with finding regions">',
                    "Analysis view: corner brackets mark finding regions, labelled with the finding ID", legend)
    with v2:
        other = Image.open(cm) if show_cm else base
        light_table(f'<img class="ev" src="{img_uri(other)}" alt="{"Copy-move pairs" if show_cm else "Original"} of {e(path.name)}">',
                    "Copy-move pairs" if show_cm else "Original, as ingested")
    if fi is not None and fi.file == path.name and "bbox" in fi.location:
        x, y, w, h = fi.location["bbox"]
        pad = max(w, h) // 2
        crop = base.crop((max(0, x - pad), max(0, y - pad), min(base.width, x + w + pad), min(base.height, y + h + pad)))
        light_table(f'<img class="ev" style="max-height:var(--h-scroll)" src="{img_uri(crop, 800)}" alt="Close-up of the focused region">',
                    f"Close-up of {e(ss['focus']['tag'])} · region x{x} y{y}, {w}×{h} px")
    spec = case.artifact_path(f"{stem}_spectrum.png")
    if spec.exists():
        with st.expander("Noise spectrum (upsampling trace)"):
            light_table(f'<img class="ev" src="{img_uri(Image.open(spec), 700)}" alt="Noise spectrum">',
                        "Log power of the high-pass residual, centre = low frequencies. Bright isolated dots off the centre are "
                        "periodic patterns; a dark outer ring is missing fine detail.")


def svg_ticks(dur, X, y):
    return "".join(f'<text x="{X(s):.0f}" y="{y}" class="t-mono">{tick_label(s)}</text>' for s in ticks(dur))


def video_view(case, path, stem, mine, fi):
    timed = sorted((i for _, i in mine if tt(i) is not None), key=tt)
    start = int(max(0.0, tt(fi) - 0.5)) if fi is not None and fi.file == path.name and tt(fi) is not None else 0
    vc, tc = st.columns([3, 2])
    vc.video(str(path), start_time=start)
    with tc, st.container(key="card_timed"):
        timed_rows = [(t, i) for t, i in mine if tt(i) is not None and fusion.contribution(i) >= fusion.LEVELS[2][0]]
        md(ui.section("Timed findings", "Weak level or above"))
        md(ui.table(["ID", "Time", "Finding"], [[f'<span class="mono">{t}</span>', ui.fdur(tt(i)), e(cap(term(i.type)[0]))]
                                               for t, i in sorted(timed_rows, key=lambda ti: tt(ti[1]))[:10]], numeric=(1,),
                    empty="No timed finding reaches the Weak level."))
        if start:
            md(f'<div class="muted">Player starts at {start} s (0.5 s before {e(ss["focus"]["tag"])}).</div>')
    frames = sorted(case.artifact_path("").glob(f"{stem}_f[0-9][0-9][0-9][0-9].jpg"))
    if not frames:
        md(ui.notice("No frames could be read from this video, so the per-frame checks could not run.", "info"))
    else:
        hits = {}
        for i in timed:
            if i.modality == "video" and fusion.contribution(i) >= fusion.LEVELS[2][0]:  # Weak or above
                hits.setdefault(round(tt(i)), LVL_TOKEN[WORD[fusion.label(fusion.contribution(i))]])
        md('<div class="cap">Keyframes, 1 per second · outlined: a counted finding of Weak level or above</div>'
           + film_strip(frames, hits, tt(fi) if fi is not None and fi.file == path.name else None))
    dur = max([len(frames), 1] + [i.location.get("end", tt(i)) + 1 for i in timed])
    X = lambda t: 40 + 940 * t / dur
    marks = "".join(
        f'<g><title>{e(term(i.type)[0])} {e(where(i.location))}</title><rect x="{X(tt(i)):.1f}" y="{14 if i.modality == "video" else 40}" '
        f'width="{max(3, X(i.location["end"]) - X(tt(i))) if "end" in i.location else 3:.1f}" height="18" rx="2" '
        f'style="fill:var(--{"mark-video" if i.modality == "video" else "mark-audio"})"/></g>' for i in timed if i.modality in ("video", "audio"))
    fj, lj = case.artifact_path(f"{stem}_faces.json"), case.artifact_path(f"{stem}_lipsync.json")
    track = json.loads(fj.read_text(encoding="utf-8")) if fj.exists() else {"rows": [], "face_ratio": 0}
    lip = json.loads(lj.read_text(encoding="utf-8")) if lj.exists() else None
    has = np.array(["box" in r for r in track["rows"]])
    ts = [r["t"] for r in track["rows"]]
    faces = "".join(f'<rect x="{X(ts[a]):.1f}" y="66" width="{max(2, X(ts[b - 1]) - X(ts[a])):.1f}" height="14" rx="2" '
                    f'style="fill:var(--mark-face)"/>' for a, b in _runs(has)) if len(has) else ""

    def curve(vals, tok):
        v = np.nan_to_num(np.asarray(vals, float))
        v = (v - v.min()) / max(np.ptp(v), 1e-9)
        return (f'<polyline fill="none" style="stroke:var(--{tok})" stroke-width="1.5" points="'
                + " ".join(f"{X(t):.1f},{126 - 34 * y:.1f}" for t, y in zip(lip["t"], v)) + '"/>')
    lips = (curve(lip["mouth"], "mark-video") + curve(lip["audio_db"], "mark-audio")
            + f'<text x="985" y="100" class="t-muted" text-anchor="end">best lag {lip["lag_ms"]:+d} ms · r {lip["r"]:.2f}</text>') if lip else ""
    track_ = "".join(f'<rect x="40" y="{y}" width="940" height="{h}" style="fill:var(--bg)"/>' for y, h in ((14, 18), (40, 18), (66, 14), (90, 40)))
    light_table(f'<svg viewBox="0 0 1000 154" width="100%" role="img" aria-label="Timeline of findings">'
                f'<text x="0" y="27" class="t-muted">Video</text><text x="0" y="53" class="t-muted">Audio</text>'
                f'<text x="0" y="78" class="t-muted">Face</text><text x="0" y="115" class="t-muted">Lips</text>'
                f'{track_}{marks}{faces}{lips}{svg_ticks(dur, X, 148)}</svg>',
                f'Face visible in {track["face_ratio"] * 100:.1f}% of frames' + ("" if lip else " · lip-sync not run (no face or no speech)"),
                [("mark-video", "Video finding"), ("mark-audio", "Audio finding"), ("mark-face", "Face visible")]
                + ([("mark-video", "Mouth motion"), ("mark-audio", "Speech loudness")] if lip else []))
    for _, i in mine:
        if i.type == "learned_synthetic_video" and i.location.get("segments"):
            score_strip(i, i.location["segments"], "two-second segment")
            crops = [c for c in i.location.get("crops", []) if case.artifact_path(c["file"]).exists()]
            if crops:
                md('<div class="film" role="list" aria-label="Frames the trained detector scored highest">' + "".join(
                    f'<figure class="hit" style="--fc:var(--mark-detector)"><img src="{img_uri(Image.open(case.artifact_path(c["file"])), 240, 80)}" '
                    f'alt="{"Face" if c["face"] else "Frame"} at {c["t"]:.1f} s"><figcaption>{c["t"]:.1f} s · '
                    f'{"face" if c["face"] else "frame"} · {c["score"]:.2f}</figcaption></figure>' for c in crops) + "</div>")


def score_strip(i, cells, what):
    """Trained-detector scores per time window as bars (threshold dashed), under the spectrogram or video timeline."""
    dur = max(c[1] for c in cells)
    X = lambda t: 40 + 940 * t / dur
    thr = i.location.get("threshold", 0.5)
    bars = "".join(f'<rect x="{X(c[0]):.1f}" y="{70 - 56 * c[2]:.1f}" width="{max(2.0, X(c[1]) - X(c[0]) - 1):.1f}" '
                   f'height="{56 * c[2]:.1f}" style="fill:var(--{"mark-detector" if c[2] >= thr else "mark-off"})" fill-opacity="0.75">'
                   f'<title>{c[0]:.1f}–{c[1]:.1f} s · {c[2]:.2f}</title></rect>' for c in cells)
    light_table(f'<svg viewBox="0 0 1000 92" width="100%" role="img" aria-label="Trained detector score per {what}">'
                f'<rect x="40" y="14" width="940" height="56" style="fill:var(--bg)"/>{bars}'
                f'<line x1="40" x2="980" y1="{70 - 56 * thr:.1f}" y2="{70 - 56 * thr:.1f}" style="stroke:var(--text)" stroke-dasharray="4 3"/>'
                f'<text x="0" y="46" class="t-muted">Score</text>{svg_ticks(dur, X, 88)}</svg>',
                f"Trained detector ({e(i.location.get('model_version', '?'))}) score per {what}; threshold {thr:.2f} (dashed). "
                "Indicative: the threshold was set on whole files.",
                [("mark-detector", "At or above threshold"), ("mark-off", "Below threshold")])


def audio_view(case, path, stem, mine, fi, log):
    spec = case.artifact_path(f"{stem}_spec.png")
    if spec.exists():
        light_table(f'<img class="ev" style="width:100%;object-fit:fill" src="{img_uri(Image.open(spec), 1600)}" '
                    f'alt="Spectrogram of {e(path.name)}">', "Spectrogram: low frequencies at the bottom",
                    [("danger", "Splice or click"), ("info", "Too-clean pause"), ("warning", "Flat prosody"), ("accent", "Band-limit cutoff")])
    elif any(x["action"] == "no_audio" and x["file"] == path.name for x in log):
        md(ui.notice("No audio track was found in this file, so the audio checks couldn't run.", "info"))
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
        md(f'<div class="muted">Player starts at {start} s (0.5 s before {e(ss["focus"]["tag"])}).</div>')


def context_view(res, log, stem, case):
    claims = next((x for x in reversed(log) if x["action"] == "text_claims"), {})
    if res.get("context"):
        marked = e(res["context"])
        toks = {"when": "accent", "where": "lvl-weak", "who": "mark-detector", "quotes": "warning"}
        for k, tok in toks.items():
            for v in claims.get(k, []):
                if v and e(str(v)) in marked:
                    marked = marked.replace(e(str(v)), f'<mark style="background:var(--{tok})" title="{k}">{e(str(v))}</mark>', 1)
        md(ui.card(ui.section("Caption or claims") + f"<p>{marked}</p>" + ui.legend([(tok, cap(k)) for k, tok in toks.items()])))
    tp = case.artifact_path(f"{stem}_transcript.json")
    if tp.exists():
        said = " ".join(s["text"] for s in json.loads(tp.read_text(encoding="utf-8")))
        found = [(q, fuzz.partial_ratio(str(q).lower(), said.lower())) for q in claims.get("quotes", [])]
        md(ui.card(ui.section("Transcript") + f'<p class="mono">{e(said) or "<span class=muted>No speech recognised.</span>"}</p>'
                   + "".join(f'<div>{ui.chip("Heard", "success", "check") if r >= 70 else ui.chip("Not found", "danger", "close")} '
                             f'“{e(str(q))}” <span class="mono muted">match {r:.0f}/100</span></div>' for q, r in found)))


def cross_tab(inds, tags, not_run):
    lanes = {"Video": 30, "Audio": 90, "Text / claims": 150, "Metadata": 210}
    timed = [i for i in inds if tt(i) is not None and i.modality in ("video", "audio")]
    dur = max([1.0] + [tt(i) + 1 for i in timed])
    X = lambda t: 200 + 780 * t / dur
    els = [f'<text x="0" y="{y + 4}" class="t-main">{n}</text><line x1="200" y1="{y}" x2="980" y2="{y}" '
           f'style="stroke:var(--mark-track)" stroke-width="18" stroke-linecap="round"/>' for n, y in lanes.items()]
    els.append('<text x="125" y="252" class="t-muted" text-anchor="middle">Untimed</text>')
    els.append(svg_ticks(dur, X, 252))
    for i in timed:
        y, tok = (lanes["Video"], "mark-video") if i.modality == "video" else (lanes["Audio"], "mark-audio")
        els.append(f'<g><title>{e(i.file)}: {e(term(i.type)[0])} {e(where(i.location))}</title><circle cx="{X(tt(i)):.1f}" cy="{y}" r="6" '
                   f'style="fill:var(--{tok})"/></g>')
    for n, i in enumerate(i for i in inds if "meta" in i.location):
        els.append(f'<g><title>{e(i.file)}: {e(term(i.type)[0])}</title><rect x="{205 + 12 * n}" y="{lanes["Metadata"] - 5}" width="8" '
                   f'height="10" style="fill:var(--text-muted)"/></g>')
    cross = [(t, i) for t, i in tags if i.modality == "cross"]
    for n, (t, i) in enumerate(cross):
        if i.type == "av_aligned_flags":
            x = X(i.location["t"])
            x2 = X(min((tt(j) for j in timed if j.modality == "audio" and j.file in i.related),
                       key=lambda v: abs(v - i.location["t"]), default=i.location["t"]))
            els.append(f'<g><title>{t}: {e(i.explanation)}</title><path d="M{x:.1f},{lanes["Video"]} C{x:.1f},60 {x2:.1f},60 {x2:.1f},{lanes["Audio"]}" '
                       f'style="stroke:var(--mark-link)" stroke-width="2.5" fill="none"/></g>')
        else:
            y1, y2 = lanes["Text / claims"], lanes[{"claim_date_mismatch": "Metadata", "quote_not_in_transcript": "Audio"}.get(i.type, "Video")]
            x = 150 + 14 * (n % 4)
            els.append(f'<g><title>{t}: {e(i.explanation)}</title><path d="M{x},{y1} C{x - 50},{y1} {x - 50},{y2} {x},{y2}" '
                       f'style="stroke:var(--mark-link)" stroke-width="2.5" fill="none"/><circle cx="{x}" cy="{y1}" r="5" style="fill:var(--mark-link)"/>'
                       f'<circle cx="{x}" cy="{y2}" r="5" style="fill:var(--mark-link)"/></g>')
    light_table(f'<svg viewBox="0 0 1000 262" width="100%" role="img" aria-label="Cross-media lanes">{"".join(els)}</svg>',
                "Curves join findings that a cross-media check found to line up (video and audio within 0.5 s) or to contradict "
                "each other (claim vs metadata or transcript). Hover for details.",
                [("mark-video", "Video finding"), ("mark-audio", "Audio finding"), ("text-muted", "Metadata finding"),
                 ("mark-link", "Cross-media check")])
    if cross:
        md('<div class="stack">' + "".join(finding_html(t, i) for t, i in cross) + "</div>")
    else:
        md(ui.empty_state("link_off", "No cross-media links or contradictions were found",
                          " ".join(hints(not_run.values())) or "Add a caption to check claims against the media."))


def findings_tab(case, inds, tags, cov):
    with st.container(key="toolbar"):
        c1, c2 = st.columns(2)
        c3 = st.container()
        groups = c1.pills("Type", list(GROUPS), selection_mode="multi", default=list(GROUPS), key=f"fg_{case.id}")
        medias = sorted({i.modality for i in inds}) or ["image"]
        media = c2.pills("Media", medias, selection_mode="multi", default=medias, key=f"fm_{case.id}", format_func=cap)
        status = c3.pills("State", ["Counted", "Uncalibrated", "Not counted", "Not run"], selection_mode="multi",
                          default=["Counted", "Uncalibrated", "Not counted", "Not run"], key=f"fs2_{case.id}")
    want = {"Counted": "counted", "Uncalibrated": "uncalibrated", "Not counted": "zero"}
    keep = [(t, i) for t, i in tags if fusion.group(i) in groups and i.modality in media and
            any(state_of(i) == want[s] for s in status if s in want)]
    nr = cov["not_run"] if "Not run" in status else {}
    md(f'<div class="muted">{len(keep)} findings · {len(nr)} checks not run · sorted by contribution to the label</div>')
    if keep or nr:
        md('<div class="stack">' + "".join(finding_html(t, i) for t, i in keep) + "".join(not_run_html(c, w) for c, w in nr.items()) + "</div>")
    elif inds:
        md(ui.empty_state("filter_alt_off", "Nothing matches these filters"))
    else:
        md(ui.empty_state("search_off", "No findings for this case", "No findings does not show the media is unaltered."))
    with st.expander("Table view (sortable, all columns)"):
        rows = [{"ID": t, "Type": fusion.group(i), "Finding": cap(term(i.type)[0]), "Media": i.modality, "File": i.file,
                 "Location": where(i.location), "Score": round(i.score, 2), "Weight": round(fusion.weight(i), 2),
                 "Contribution": round(fusion.contribution(i), 3), "State": ui.STATE[state_of(i)][0],
                 "Explanation": i.explanation, "Could also be": "; ".join(i.alt_explanations)} for t, i in tags]
        st.dataframe(rows, width="stretch", hide_index=True) if rows else md(ui.empty_state("table_rows", "No findings"))
    md(f'<div class="muted" style="font-size:var(--fs-xs)">{e(fusion.ALIGN_RULE)}</div>')


def custody_tab(case, res, log):
    def actions():
        if st.button("Verify integrity", type="primary", icon=":material/fingerprint:", key="verify_custody"):
            verify_now(case)
            st.rerun()
        export_button(case, "exp_custody")
    page_head("Chain of custody", "Every step logged, every file hashed on arrival.", actions)
    c1, c2 = st.columns([2, 3])
    with c1:
        custody_stamp(case, log, res)
        with st.expander("Full SHA-256 hashes (copy)"):
            for x in log:
                if x["action"] == "ingest":
                    st.code(f'{x["sha256"]}  {x["file"]}', language=None)
        md('<div class="muted" style="font-size:var(--fs-xs)">For PDF: open the exported HTML report and print it to PDF.</div>')
    with c2, st.container(key="card_log"):
        md(ui.section("Custody log", f"{len(log)} entries, oldest first"))
        rows = []
        for x in log:
            txt = ", ".join(f"{k}={v}" for k, v in x.items() if k not in ("ts", "action"))
            rows.append([f'<span class="mono">{e(ui.fdt(x["ts"]))}</span>', e(x["action"].replace("_", " ")),
                         f'<span class="mono trunc" title="{e(txt[:2000])}">{e(txt[:160])}</span>'])
        md(ui.table(["Time", "Action", "Details (hover for all)"], rows))


PRETTY_MEDIA = {"image": "image", "video": "video", "audio": "audio", "cross": "cross-media"}


def pretty_key(k):
    """'learned_synthetic_image/image' -> 'Trained detector (image) · image'."""
    ty, _, med = k.partition("/")
    return f"{cap(term(ty)[0])} · {PRETTY_MEDIA.get(med, med)}"


def calibration_tab():
    page_head("Calibration", "How each check was measured, and what is not counted yet.")
    if not cal:
        md(ui.notice("No calibration loaded. Run <code>python calibrate.py</code> on the labelled test set; until then every weight is 0 "
                     "and labels are not meaningful.", "danger"))
        return
    md(f'{ui.chip(f"{cal["dataset"]} test set · n={cal["n"]}", "success" if cal["dataset"] == "REAL" else "warning", "dataset")} '
       f'{ui.chip("Calibrated " + cal["date"], "neutral", "event")} {ui.chip("git " + cal["git"], "neutral", "commit")} '
       f'{ui.chip(f"fitted on n={cal["n_calibration"]} · held-out n={cal["n_held_out"]}", "neutral", "call_split")}')
    if cal["dataset"] != "REAL":
        md(ui.notice("These numbers come from SYNTHETIC placeholder media. They show the system runs end to end; they are not detector "
                     "accuracy.", "warning"))
    det = cal["detectors"]
    rows = []
    for t, d in det.items():
        p, lo, hi = d["precision"]
        bar = (f'<div class="ci" role="img" aria-label="precision {ui.fpct(p)}, interval {ui.fpct(lo)} to {ui.fpct(hi)}">'
               f'<i style="left:{100 * lo:.1f}%;width:{100 * (hi - lo):.1f}%"></i><b style="left:{100 * p:.1f}%"></b></div>')
        rows.append([e(pretty_key(t)), bar, f'{ui.fpct(p)}', f'{d["hits"]} / {d["n_manipulated"]}', f'{d["false_alarms"]} / {d["n_untouched"]}',
                     f'{d["weight"]:.2f}'])
    with st.container(key="card_prec"):
        md(ui.section("Precision per check", "bar: 95% Wilson interval · dot: measured precision"))
        md(ui.table(["Check · media", "Precision (0–100%)", "Precision", "Fired on manipulated", "Fired on untouched", "Weight"], rows,
                    numeric=(2, 3, 4, 5)))
    h = cal["held_out"]
    c1, c2 = st.columns(2)
    with c1, st.container(key="card_cm"):
        md(ui.section("Held-out confusion matrix", "Moderate or Strong = flagged"))
        md(ui.table(["", "Flagged", "Not flagged"], [["Manipulated", str(h["tp"]), str(h["fn"])], ["Untouched", str(h["fp"]), str(h["tn"])]],
                    numeric=(1, 2)))
        md(f'<div class="muted">Hit rate {ui.fpct(h["hit_rate"][0])} ({ui.fpct(h["hit_rate"][1])}–{ui.fpct(h["hit_rate"][2])}) · '
           f'false-alarm rate {ui.fpct(h["false_alarm_rate"][0])} ({ui.fpct(h["false_alarm_rate"][1])}–{ui.fpct(h["false_alarm_rate"][2])})</div>')
    with c2, st.container(key="card_lim"):
        md(ui.section("Limitations"))
        md(f'''<ul class="plain"><li>The test set is small and team-made, so every rate has a wide interval: read the bars, not the dots.</li>
<li>Platform recompression and resizing can both create and erase these indicators.</li>
<li>{e(fusion.ALIGN_RULE)}</li>
<li>Trained detectors: {e(learned.WEAK)} {e(learned.FIRST_RESULT)} {e(learned.V2)}</li>
<li>Fully generated media without provenance metadata or visible artifacts may show no indicators at all.</li>
<li>No indicators does not mean unaltered. The language model reads text, compares captions with pictures and lists visible
artifacts with their location; it is never asked whether media is synthetic.</li></ul>''')
    if lw := cal.get("learned"):
        with st.container(key="card_learned"):
            md(ui.section("Trained detectors on our own media",
                          f"weight 0 (uncalibrated, not counted in the label) until a media type has {MIN_OWN} untouched and {MIN_OWN} AI files"))
            md(ui.table(["Detector", "Untouched (phone hold-out)", "False alarms", "AI files", "Weight", "Still needed"],
                        [[e(pretty_key(t)), f'{v["n_untouched"]} ({v["phone_holdout"]})', f'{v["false_alarms"]} / {v["n_untouched"]}',
                          str(v["n_manipulated"]), "uncalibrated" if v["weight"] is None else f'{v["weight"]:.2f}',
                          f'{v["need"]["untouched"]} untouched · {v["need"]["ai"]} AI'] for t, v in lw.items()], numeric=(1, 2, 3, 4)))
    card = server_model_card()
    ver = re.search(r"\((v\d+)[,)]", card.splitlines()[0]) if card else None
    with st.expander("Model card: trained detectors (from the detector server)" + (f" · {ver[1]}" if ver else "")):
        with st.container(key="modelcard"):
            if card:
                st.markdown(card)
            else:
                md(ui.empty_state("description", "Model card: not yet available from the detector server"))
    with st.expander("Glossary: what each finding means"):
        md(ui.table(["Finding", "What it means"], [[e(cap(n)), e(d)] for n, d in sorted(TERMS.values())]))


@st.cache_data(ttl=300, show_spinner=False)
def server_model_card():
    return llm.model_card()


def go(page):
    ss["page"] = page
    qp["page"] = page
    ss["_scroll"] = True
    st.rerun()


def results_page(case, page):
    res = load(case.id, mtime(case.id))
    inds, a, log = res["indicators"], res["assessment"], case.entries()
    tags = report.tagged(inds)
    not_run = {x["check"]: x["reason"] for x in log if x["action"] == "check_not_run"}
    if page == "Analysis":
        analysis_page(case, res, inds, tags, not_run)
    elif page == "Evidence viewer":
        page_head("Evidence viewer", "The evidence with every finding at its location: regions, frames, time spans and text.")
        safe(lambda: evidence_tab(case, res, inds, tags, log), "evidence view")
        footer()
    elif page == "Chain of custody":
        safe(lambda: custody_tab(case, res, log), "custody view")
        footer()
    else:
        safe(lambda: overview(case, res, inds, a, tags, log), "overview")


def public_expiry(case):
    t = public.expires(case.id) if PUBLIC else None
    return ui.fdt(t) if t else None


page = "Calibration" if nav_page() == "Calibration" else "New analysis" if pick == "new" else nav_page()
job = ss.get("job")
topbar("Analysing case" if page != "Calibration" and job and job["case"] == pick else page)
if ss.pop("_scroll", False):
    st.iframe("<script>window.parent.document.querySelector('[data-testid=stMain]').scrollTo(0, 0)</script>", height=1)


def route():
    if page == "Calibration":
        safe(calibration_tab, "calibration view")
        footer()
    elif pick == "new":
        upload_page()
    elif job and job["case"] == pick:
        analyzing_page(job)
    else:
        results_page(Case(pick), page)


try:
    route()
except Exception:
    if not PUBLIC:
        raise
    oops("page", traceback.format_exc())
