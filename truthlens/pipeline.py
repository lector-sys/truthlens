import json
from . import analyzers, crossmodal, fusion, report
from .analyzers import text as text_analyzer
from .indicator import Indicator


def run(case, context_text="", progress=lambda step, state: None):
    """Analyze every evidence file, then context text, cross-modal checks and fusion. `progress(step, state)` is called
    with state "running" / "done" / "error" for each step so a UI can show live status. Result saved to result.json."""
    indicators = []
    for path in case.evidence():
        mods = analyzers.BY_EXT.get(path.suffix.lower())
        if not mods:
            case.log("skipped", file=path.name, reason="unsupported type")
            continue
        for mod in mods:
            step = f"{mod.__name__.rsplit('.', 1)[1]} · {path.name}"
            progress(step, "running")
            try:
                found = mod.analyze(path, case)
            except Exception as e:  # one broken analyzer must not kill the case
                case.log("analyzer_error", file=path.name, analyzer=mod.__name__, error=repr(e))
                progress(step, "error")
                continue
            case.log("analyzed", file=path.name, analyzer=mod.__name__, n=len(found))
            progress(step, "done")
            indicators += found
    if context_text:
        progress("text · context", "running")
        indicators += text_analyzer.analyze(context_text, case)
        progress("text · context", "done")
    progress("cross-media checks", "running")
    indicators += crossmodal.analyze(indicators, context_text, case)
    progress("cross-media checks", "done")
    result = {"indicators": indicators, "assessment": fusion.assess(indicators), "integrity": case.verify(),
              "context": context_text}
    case.log("assessed", **result["assessment"]["case"])
    report.summary(result, case)  # once per run, saved below: viewing the case must not call (or log) the model again
    (case.dir / "result.json").write_text(json.dumps({**result, "indicators": [i.to_dict() for i in indicators]}),
                                          encoding="utf-8")
    return result


def load(case):
    """The saved result of a previous run (None if never analyzed). Fusion and integrity are recomputed now, so new
    calibration weights and any tampering since the run show up."""
    p = case.dir / "result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    inds = [Indicator(**d) for d in r["indicators"]]
    return {**r, "indicators": inds, "assessment": fusion.assess(inds), "integrity": case.verify()}
