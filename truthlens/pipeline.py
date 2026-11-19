from . import analyzers, crossmodal, fusion
from .analyzers import text as text_analyzer


def run(case, context_text=""):
    indicators = []
    for path in case.evidence():
        mod = analyzers.BY_EXT.get(path.suffix.lower())
        if not mod:
            case.log("skipped", file=path.name, reason="unsupported type")
            continue
        try:
            found = mod.analyze(path, case)
        except Exception as e:  # one broken analyzer must not kill the case
            case.log("analyzer_error", file=path.name, analyzer=mod.__name__, error=repr(e))
            continue
        case.log("analyzed", file=path.name, analyzer=mod.__name__, n=len(found))
        indicators += found
    if context_text:
        indicators += text_analyzer.analyze(context_text, case)
    indicators += crossmodal.analyze(indicators, context_text, case)
    result = {"indicators": indicators, "assessment": fusion.assess(indicators), "integrity": case.verify()}
    case.log("assessed", **result["assessment"]["case"])
    return result
