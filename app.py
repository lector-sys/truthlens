"""streamlit run app.py"""
import streamlit as st
from truthlens.vault import Case
from truthlens import pipeline, report

IMAGE = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
VIDEO = (".mp4", ".mov", ".mkv", ".webm", ".avi")
AUDIO = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

st.set_page_config("TruthLens", layout="wide")
st.title("TruthLens: multimodal forensic analysis")

with st.form("case"):
    files = st.file_uploader("Evidence (image / video / audio)", accept_multiple_files=True)
    context = st.text_area("Context (caption, claimed date/place, quotes)")
    go = st.form_submit_button("Analyze")

if go and files:
    case = Case()
    for f in files:
        case.ingest(f.name, f.getvalue())
    with st.spinner("Analyzing..."):
        st.session_state.result = pipeline.run(case, context)
    st.session_state.case = case


def show(i):
    st.markdown(f"**{i.modality} · {i.type}** · strength {i.score:.2f} ({i.reliability} reliability) · `{i.location}`  \n"
                f"{i.explanation}  \n_Could also be:_ {'; '.join(i.alt_explanations) or '-'}")


if "result" in st.session_state:
    case, res = st.session_state.case, st.session_state.result
    a, inds = res["assessment"], sorted(res["indicators"], key=lambda i: -i.score)

    st.subheader(f"{a['case']['label']} · strength {a['case']['p']:.2f}")
    st.caption("Indicator, not verdict. Every finding needs human review.")
    st.write(report.summary(res))

    bad = [f for f, ok in res["integrity"].items() if not ok]
    (st.error(f"Integrity check FAILED: {', '.join(bad)}") if bad else
     st.success(f"Integrity: all {len(res['integrity'])} evidence file(s) match their SHA-256 at ingest."))

    for path in case.evidence():
        mine = [i for i in inds if i.file == path.name and i.modality != "cross"]
        ext = path.suffix.lower()
        with st.expander(f"{path.name}: {a['files'].get(path.name, {}).get('label', 'No significant indicators')}", expanded=True):
            if ext in IMAGE:
                arts = [i for i in mine if i.artifact]
                cols = st.columns(1 + len(arts[:3]))
                cols[0].image(str(path), caption="original")
                for col, i in zip(cols[1:], arts):
                    col.image(i.artifact, caption=f"{i.type} {i.location.get('bbox', '')}")
            elif ext in VIDEO:
                st.video(str(path))
                timed = sorted((i for i in mine if "t" in i.location), key=lambda i: i.location["t"])
                if timed:
                    st.caption("Flagged moments")
                    cols = st.columns(min(len(timed), 5))
                    for col, i in zip(cols, timed[:5]):
                        if i.artifact:
                            col.image(i.artifact, caption=f"{i.location['t']:.1f}s · {i.type}")
                        else:
                            col.caption(f"{i.location['t']:.1f}s · {i.type}")
            if ext in AUDIO or ext in VIDEO:
                if ext in AUDIO:
                    st.audio(str(path))
                spec = next((i.artifact for i in mine if i.modality == "audio" and i.artifact), None)
                if spec:
                    st.image(spec, caption="Spectrogram (red = flagged segments)")
            for i in mine:
                show(i)
            if not mine:
                st.caption("No indicators for this file.")

    cross = [i for i in inds if i.modality in ("cross", "text")]
    st.subheader("Cross-modal & context findings")
    if cross:
        st.dataframe([{"type": i.type, "file": i.file, "related": ", ".join(i.related), "strength": round(i.score, 2),
                       "reliability": i.reliability, "why": i.explanation, "could also be": "; ".join(i.alt_explanations)}
                      for i in cross], use_container_width=True)
    else:
        st.caption("No cross-modal or context indicators.")

    checks = report.human_checks(inds)
    if checks:
        st.subheader("Recommended human checks")
        st.markdown("\n".join(f"{n}. {c}" for n, c in enumerate(checks, 1)))

    st.download_button("Download report (HTML)", report.render(case, res), f"{case.id}.html", "text/html")
