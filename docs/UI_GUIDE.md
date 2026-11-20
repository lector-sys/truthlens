# TruthLens UI guide

One design system for the dashboard and the HTML report. Values live in **`ui/tokens.css`**, styles in **`ui/theme.css`**,
HTML components in **`ui/components.py`**. No colour, size or spacing value is written anywhere else; `test_smoke.py`
fails on a hard-coded colour in `app.py`, `ui/components.py` or `ui/theme.css`, and on any text/background pairing below
WCAG AA (4.5:1). The one exception: CSS media queries can't read variables, so the phone breakpoint is written as `640px`.

## Tokens
| Group | Tokens |
|---|---|
| Spacing | `--s-1`…`--s-8` = 4 · 8 · 12 · 16 · 24 · 32 · 48 · 64 px |
| Type | `--fs-xs` 12 · `--fs-sm` 14 · `--fs-md` 16 · `--fs-lg` 20 · `--fs-xl` 28 · `--fs-2xl` 40 (page titles); weights 400 / 600 |
| Fonts | Space Grotesk (headings) · Inter (text) · JetBrains Mono (IDs, hashes, times, coordinates, scores; tabular figures) |
| Surfaces | `--bg`, `--surface-1` (cards), `--surface-2` (inputs, table heads), `--border` |
| Text | `--text`, `--text-secondary`, `--text-muted`; `--accent-ink` on accent fills |
| Status | `--success`, `--warning`, `--danger`, `--info`, `--neutral` |
| Levels | `--lvl-no`, `--lvl-weak`, `--lvl-moderate`, `--lvl-strong` (only for the four indicator levels) |
| Evidence marks | `--mark-video`, `--mark-audio`, `--mark-face`, `--mark-detector`, `--mark-link`, `--mark-track`, `--mark-off` |
| Shape | radii 6 (inputs, chips) · 10 (buttons) · 14 (cards) · pill; one hairline border; no shadows |
| Motion | 180 ms ease-out for hover and expand; only the analysis progress pulses; `prefers-reduced-motion` respected |
| Print | `--print-*` for the report's light theme |

## Components (`ui/components.py`)
| Component | Function | Notes |
|---|---|---|
| Button | Streamlit `st.button` / `st.download_button` | primary (accent fill, dark text) · secondary · tertiary (ghost) · danger (`key="danger_…"`); 44 px touch height; Material icon via `icon=":material/…:"` |
| Status chip / pill | `chip()`, `pill()` | success · warning · danger · info · neutral · accent; always text + icon, never colour alone |
| Indicator label | `level_label(word, text, "sm" or "lg")` | No · Weak · Moderate · Strong (or none for Limited / Failed / Not analysed) |
| KPI tile | `kpi()` inside `st.container(key="kpi_…")` | eyebrow label, value, sub-text, optional chip; equal height per row |
| Card | `st.container(key="card_…")` + `section(title, desc)` | header title left, action right; cards in one row stretch to equal height |
| Page header | `page_head()` in app.py → `page_header()` | title, one-line description, actions on the right; breadcrumb in the top bar |
| Finding row | `finding_row()` | `E-xx`, title (Sentence case), media · description, state chip (Counted · Uncalibrated · Not counted · Not run · Failed), Details |
| Table | `table(cols, rows, numeric)` | hairline rows, numeric columns right-aligned mono; scrolls inside its card; empty state built in |
| Coverage row | `coverage_row()` | label, `n / m`, bar |
| Empty state | `empty_state(icon, message, hint)` | |
| Error state | `oops()` in app.py | plain message + Details expander (public mode: reference id only) |
| Notice | `notice(text, kind)` | info · warning · danger · success · provenance |
| Hash | `hash_short()` | shortened mono, full value on hover; full copyable hashes on Chain of custody |
| Legend | `legend([(token, label)])` | under every heatmap, timeline, spectrogram and score strip |
| Progress row | `progress_row()` | Queued · Running · Done · Failed · Not run, with duration |

## Words
| Concept | Use | Not |
|---|---|---|
| One result of a check | **Finding** (ID `E-01`) | indicator, flag, evidence tag |
| The case outcome | **Label** | verdict, assessment |
| An analyzer | **Check** | detector (except "trained detector", a named check) |
| An analysis job | **Case** | analysis |
| Spelling | British: analyse, analysing, colour | analyze |

Formats: dates `20 Nov 2026, 14:05`; case list `Today 14:05` / `19 Nov`; durations `2.4 s`; scores two decimals; percentages
one decimal. Sentence case everywhere except the small uppercase eyebrow labels.

## Layout
Sidebar (brand, Workspace nav, Current case, New analysis, Cases) · top bar (breadcrumb, system status, demo badge) ·
page header · cards on Streamlit's column grid with fixed gutters. At 390 px the sidebar folds into the menu button,
KPI tiles go two by two and tables scroll inside their card. Streamlit's toolbar, status widget, spinners and footer are
hidden; the page doesn't dim or jump while it reruns.
