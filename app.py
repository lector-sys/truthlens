"""streamlit run app.py"""
import streamlit as st
from truthlens.vault import Case
from truthlens import pipeline, report

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

if "result" in st.session_state:
    case, res = st.session_state.case, st.session_state.result
    a = res["assessment"]
    st.subheader(f"{a['case']['label']}  ·  strength {a['case']['p']:.2f}")
    st.caption("Indicator, not verdict. Review every finding.")

    for path in case.evidence():
        inds = [i for i in res["indicators"] if i.file == path.name or path.name in i.related]
        with st.expander(f"{path.name}: {a['files'].get(path.name, {}).get('label', 'No significant indicators')}", expanded=True):
            # TODO D: per-modality views (heatmap overlay, video timeline, waveform/spectrogram segments)
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                st.image(str(path), width=400)
            for i in inds:
                st.markdown(f"**{i.type}** · {i.score:.2f} ({i.reliability}) · `{i.location}`  \n{i.explanation}  \n"
                            f"_Could also be:_ {'; '.join(i.alt_explanations) or '-'}")
                if i.artifact:
                    st.image(i.artifact, width=400)

    st.download_button("Download report (HTML)", report.render(case, res), f"{case.id}.html", "text/html")
