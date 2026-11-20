"""TruthLens UI components: small functions that return consistent HTML for the dashboard (styles in ui/theme.css,
values in ui/tokens.css). Display only. Icons: Material Symbols Rounded (bundled with Streamlit), always with a text label.
See docs/UI_GUIDE.md."""
import html
import re
import time
from pathlib import Path

e = html.escape
TOKENS_CSS = Path(__file__).with_name("tokens.css")
T = dict(re.findall(r"--([\w-]+):\s*([^;]+);", TOKENS_CSS.read_text(encoding="utf-8")))  # name -> value


def rgb(name):
    """Token colour as an (r, g, b) tuple, for PIL overlays."""
    h = T[name].lstrip("#")[:6]
    return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))


def contrast(a, b):
    """WCAG contrast ratio of two #RRGGBB colours."""
    def lum(h):
        c = [int(h.lstrip("#")[k:k + 2], 16) / 255 for k in (0, 2, 4)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# ---- formats: one way to write each kind of value ----
def fdt(ts):
    """20 Nov 2026, 14:05"""
    lt = time.localtime(ts)
    return f"{lt.tm_mday} {time.strftime('%b %Y, %H:%M', lt)}"


def fdate(ts):
    lt = time.localtime(ts)
    return f"{lt.tm_mday} {time.strftime('%b %Y', lt)}"


def fshort(ts):
    """Case list: "Today 14:05" or "19 Nov"."""
    lt = time.localtime(ts)
    return f"Today {time.strftime('%H:%M', lt)}" if lt[:3] == time.localtime()[:3] else f"{lt.tm_mday} {time.strftime('%b', lt)}"


def fdur(s):
    return f"{s:.1f} s"


def fscore(x):
    return f"{x:.2f}"


def fpct(x):
    return f"{100 * x:.1f}%"


# ---- atoms ----
def icon(name, cls=""):
    return f'<span class="ico {cls}" aria-hidden="true">{name}</span>'


KINDS = ("success", "warning", "danger", "info", "neutral", "accent")


def chip(text, kind="neutral", ico=None, title=None):
    """Small status chip: text always present, colour and icon only reinforce it."""
    t = f' title="{e(title)}"' if title else ""
    return f'<span class="chip chip-{kind}"{t}>{icon(ico) if ico else ""}{e(text)}</span>'


def pill(text, kind="neutral", ico=None, title=None):
    """Rounded status pill (top bar, case list)."""
    t = f' title="{e(title)}"' if title else ""
    return f'<span class="pill pill-{kind}"{t}>{icon(ico) if ico else ""}{e(text)}</span>'


LEVEL_ICON = {"No": "radio_button_unchecked", "Weak": "clock_loader_20", "Moderate": "clock_loader_60", "Strong": "clock_loader_90",
              None: "remove"}
LEVEL_CLASS = {"No": "lvl-no", "Weak": "lvl-weak", "Moderate": "lvl-moderate", "Strong": "lvl-strong", None: "lvl-none"}


def level_label(word, text, size="sm"):
    """The four indicator levels (or None for not analysed / failed) with icon and words, small or large."""
    return (f'<span class="level level-{size} {LEVEL_CLASS[word]}">{icon(LEVEL_ICON[word])}'
            f'<span>{e(text)}</span></span>')


def hash_short(h, n=12):
    """Monospace hash, shortened; the full value on hover."""
    return f'<span class="mono hash" title="{e(h)}">{e(h[:n])}…{e(h[-4:])}</span>' if len(h) > n + 4 else f'<span class="mono hash">{e(h)}</span>'


def legend(items):
    """[(token, label)] -> swatch legend, the same under every heatmap, timeline and chart."""
    return ('<div class="legend">' + "".join(f'<span><i style="background:var(--{tok})"></i>{e(lab)}</span>' for tok, lab in items)
            + "</div>")


# ---- blocks ----
def page_header(title, desc=""):
    return f'<header class="ph"><h1>{e(title)}</h1>' + (f'<p>{e(desc)}</p>' if desc else "") + "</header>"


def section(title, desc=""):
    return (f'<div class="sec"><div class="sh" role="heading" aria-level="2">{e(title)}</div>'
            + (f'<span>{e(desc)}</span>' if desc else "") + "</div>")


def kpi(label, value, sub="", chip_html=""):
    return (f'<div class="kpi"><div class="eyebrow">{e(label)}</div><div class="kv">{value}</div>'
            + (f'<div class="kc">{chip_html}</div>' if chip_html else "") + (f'<div class="ks">{sub}</div>' if sub else "") + "</div>")


def card(body, cls=""):
    return f'<div class="card {cls}">{body}</div>'


def notice(text, kind="info", ico=None, title=None):
    ico = ico or {"info": "info", "warning": "warning", "danger": "error", "provenance": "verified", "success": "check_circle"}[kind]
    head = f"<b>{e(title)}</b> " if title else ""
    return f'<div class="notice notice-{kind}" role="note">{icon(ico)}<div>{head}{text}</div></div>'


def empty_state(ico, msg, hint=""):
    return (f'<div class="empty">{icon(ico, "big")}<div class="em">{e(msg)}</div>'
            + (f'<div class="eh">{e(hint)}</div>' if hint else "") + "</div>")


def coverage_row(label, ran, total):
    pct = 100 * ran / total if total else 0
    return (f'<div class="covrow"><span>{e(label)}</span><span class="mono">{ran} / {total}</span>'
            f'<div class="bar" role="progressbar" aria-valuenow="{ran}" aria-valuemin="0" aria-valuemax="{total}" '
            f'aria-label="{e(label)}: {ran} of {total} checks run"><i style="width:{pct:.0f}%"></i></div></div>')


def table(cols, rows, numeric=(), empty="Nothing to show."):
    """cols: [label]; rows: [[cell html]] (escape text before); numeric: column indexes, right-aligned with tabular figures."""
    if not rows:
        return empty_state("table_rows", empty)
    head = "".join(f'<th class="{"num" if k in numeric else ""}">{e(c)}</th>' for k, c in enumerate(cols))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if k in numeric else ""}">{c}</td>' for k, c in enumerate(r)) + "</tr>"
                   for r in rows)
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


STATE = {"counted": ("Counted", "warning", "check"), "uncalibrated": ("Uncalibrated", "info", "help"),
         "zero": ("Not counted", "neutral", "do_not_disturb_on"), "not_run": ("Not run", "neutral", "block"),
         "failed": ("Failed", "danger", "error")}


def finding_row(tag, title, media, desc, state, level_word=None, meta="", details="", focused=False, title_tip=""):
    """One finding: E-xx tag, title, media · description, state chip; optional details (location, score, could also be)."""
    label, kind, ico = STATE[state]
    tip = f' title="{e(title_tip)}"' if title_tip else ""
    det = f'<details><summary>Details</summary>{details}</details>' if details else ""
    return (f'<div class="finding {LEVEL_CLASS[level_word]}{" focused" if focused else ""}">'
            f'<span class="mono ftag">{e(tag)}</span><div class="fbody"><div class="ftitle"><span{tip}>{e(title)}</span>'
            f'{chip(label, kind, ico)}</div><div class="fdesc">{e(media)} · {e(desc)}</div>'
            + (f'<div class="fmeta mono">{meta}</div>' if meta else "") + f"{det}</div></div>")


def progress_row(state, name, secs=None, note=""):
    look = {"queued": ("Queued", "neutral", "schedule"), "running": ("Running", "info", "progress_activity"),
            "done": ("Done", "success", "check"), "error": ("Failed", "danger", "error"), "not_run": ("Not run", "neutral", "block")}
    label, kind, ico = look.get(state, (state.title(), "neutral", "radio_button_unchecked"))
    return (f'<div class="prow">{chip(label, kind, ico)}<span class="pn">{e(name)}'
            + (f' <span class="muted">· {e(note)}</span>' if note else "") + f'</span><span class="mono pt">{"" if secs is None else fdur(secs)}</span></div>')
