# Design system: finance-portfolio

## Product context
- **What this is:** five working finance tools (comps, variance, LBO, Form D sourcing, portfolio KPIs) on one verified Python library, shown as a Streamlit app.
- **Who it's for:** hiring managers and analysts in banking, PE, FP&A and venture, who read sell-side research and live in Excel and FactSet.
- **Project type:** data-dense web app with a short editorial front page.

## Aesthetic direction
- **Direction:** *research note*. It should read like a sell-side initiation report or an FT data page, not a SaaS dashboard. Paper, ink, rules, numbered sections.
- **Decoration:** minimal. Typography and hairline rules do the work. No gradients, shadows, emoji, or icon-in-a-circle grids.
- **Mood:** calm, exact, a little old-school. A reader should trust the numbers before they read them.
- **Reference points:** sell-side research PDFs (numbered exhibits, source lines under every chart), FT and Economist data pages, Koyfin's density.

## Typography
| Role | Font | Why |
| --- | --- | --- |
| Display / headings | **Newsreader** (500–600) | An editorial serif with optical sizes. It reads like a research masthead, and nothing else in Streamlit-land looks like it. |
| Body / UI | **IBM Plex Sans** (400/500/600) | Plain-spoken and engineered, with IBM's finance heritage. Not one of the overused defaults. |
| Figures / code | **IBM Plex Mono** (400/500) | Tabular by construction, so columns of numbers align. Used for exhibit numbers, tickers and code. |

Scale (base 15px): h1 40, h2 26, h3 20, h4 16, small caps eyebrow 11px with 0.08em tracking.
Sentence case everywhere. Title Case reads like a template.

## Color
Restrained: paper, ink, one blue, and status colors that mean gain and loss only.

| Token | Hex | Use |
| --- | --- | --- |
| paper | `#FBFAF7` | page background |
| paper-2 | `#F3F1EB` | sidebar, widget fills, table headers |
| rule | `#E2DED4` | hairlines, borders |
| ink | `#16181D` | body text |
| ink-2 | `#4B4F58` | secondary text |
| muted | `#8A8D94` | axis labels, captions |
| navy (primary) | `#1C3F73` | buttons, links, active nav; white text clears AA |
| series 1 / blue | `#2A5EA8` | primary chart series |
| series 2 / copper | `#D4692C` | second series only |
| series 3 / verdigris | `#1B9A78` | third series only |
| gain | `#1A7F4F` | favorable, profitable, good |
| caution | `#B54708` | warning (text); fill `#F5B544` |
| loss | `#B42318` | unfavorable, critical |

The three series colors pass the dataviz validator on the paper surface: lightness band, chroma floor, adjacent CVD ΔE ≥ 10.3, normal-vision ΔE ≥ 23.8, and contrast ≥ 3:1. Status colors never mark a series.

## Spacing and layout
- 4px base and comfortable density. Tight inside tables, generous between sections.
- Top navigation, so the sidebar holds only a page's inputs.
- Every page follows the same anatomy, like an exhibit in a research note:
  1. Eyebrow (`02 · VARIANCE`), serif title, one-line dek.
  2. Provenance line: a gray badge (`Sample data · synthetic` or `Live · SEC EDGAR`) plus the source in a caption. Never a blue info box.
  3. KPI strip: bordered metrics in a horizontal container.
  4. Exhibits. Each chart gets `Exhibit N` in mono, a sentence-case title, and a `Source:` caption underneath.
- Radius 4px (`small`), 1px borders, no shadows.

## Charts
- Plotly with one y-axis per chart. Hairline gridlines in `rule`, axis text in `muted`, and Plex Sans throughout.
- The title goes in the page as an exhibit label, not inside the plot, so charts align with the text grid.
- Direct labels on the few marks that matter, and hover on everything.

## Motion
Minimal-functional. Streamlit's own transitions only.

## Decisions log
| Date | Decision | Rationale |
| --- | --- | --- |
| 2026-10-11 | Research-note direction: Newsreader, IBM Plex Sans and Mono, paper and ink with a navy accent | The owner asked for a less generic look. A finance audience trusts the visual language of research reports, and stock Streamlit reads as a template. |
| 2026-10-11 | Theme via `.streamlit/config.toml`; CSS only for exhibit labels and the masthead | Streamlit's bundled guidance: native theming survives upgrades, while CSS on internal classes does not. |
| 2026-10-11 | Parts of a whole across more than three components are a one-hue heatmap with values printed, not a stacked bar | Only three series colors are validated; a fourth or fifth hue would be unvalidated or collide with chrome gray. |
| 2026-10-11 | Every provenance badge is gray, including Live and uploaded data | Provenance is not a status; green and blue would borrow meanings the palette reserves. |
