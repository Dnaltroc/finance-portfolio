"""Filing Tone: a sentiment model fine-tuned on Financial PhraseBank, run over real 10-K text.

A view only. Every number here is read from projects/filing_tone/tone_snapshot.json (scored
10-Ks) and results.json (model comparison), both written by the CLI. No torch import: the
app image and Streamlit Cloud never need the ml extra."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fp_ui as ui
from projects.filing_tone.cli import load_results, load_snapshot, peer_median

ui.page_setup("What the 10-K sounds like, sentence by sentence",
              "A transformer fine-tuned on 4,846 expert-labeled finance sentences reads the Risk Factors and "
              "MD&A of each company's latest 10-K and calls every sentence positive, negative or neutral. "
              "Net tone is the share positive minus the share negative.",
              number="06", section="Filing tone")

snap = load_snapshot()
results = load_results()
companies = snap["companies"]
model_run = snap["run_id"]
st.markdown(f":gray-badge[:material/description: Snapshot · SEC EDGAR 10-Ks]&nbsp; <small>Scored {snap['date']} "
            f"by `{snap['model']}` (run {model_run}), trained on {snap['training_data']}. "
            f"Re-run with `python -m projects.filing_tone snapshot`.</small>", unsafe_allow_html=True)

# ---------------------------------------------------------------- KPI strip ----------
aehr = companies["AEHR"]["sections"]
peer_mdna = peer_median(snap, "mdna")
best = results.get("distilbert") or max(results.values(), key=lambda r: r["macro_f1"]) if results else None
with st.container(horizontal=True):
    st.metric("AEHR net tone, MD&A", f"{aehr['mdna']['net_tone']:+.2f}",
              f"{aehr['mdna']['n']} sentences, FY ended {companies['AEHR']['report_date']}",
              delta_color="off", delta_arrow="off", border=True, height="stretch")
    st.metric("AEHR net tone, risk factors", f"{aehr['risk_factors']['net_tone']:+.2f}",
              f"{aehr['risk_factors']['n']} sentences", delta_color="off", delta_arrow="off", border=True,
              height="stretch")
    st.metric("Peer median, MD&A", "NM" if peer_mdna is None else f"{peer_mdna:+.2f}",
              f"{len(companies) - 1} peers with 10-Ks", delta_color="off", delta_arrow="off", border=True,
              height="stretch")
    st.metric("Model macro-F1, test split", "NM" if best is None else f"{best['macro_f1']:.3f}",
              "NM" if best is None else f"{best['accuracy']:.1%} accuracy, {best['test_size']} held-out sentences",
              delta_color="off", delta_arrow="off", border=True, height="stretch")

st.markdown(
    "**How to read this.** Risk-factor sections are written to be negative, so every company's risk tone sits "
    "below zero; compare companies against each other, not against zero. MD&A is management's own narrative, "
    "and a negative MD&A tone while peers are positive is the signal worth reading. The model learned on news "
    "sentences, not filings, so treat the numbers as a first read, then open the driver sentences below."
)

# ---------------------------------------------------------------- exhibits -----------
order = sorted(companies, key=lambda t: companies[t]["sections"]["mdna"]["net_tone"])


def tone_bar(section: str, n: int, title: str) -> None:
    ui.exhibit(n, title)
    tickers = [t for t in order if companies[t]["sections"][section]["n"] > 0]
    vals = [companies[t]["sections"][section]["net_tone"] for t in tickers]
    fig = go.Figure(go.Bar(
        x=vals, y=tickers, orientation="h",
        marker_color=[ui.NAVY if t == "AEHR" else ui.BLUE for t in tickers],
        text=[f"{v:+.2f}" for v in vals], textposition="outside", cliponaxis=False,
        customdata=[[companies[t]["company"], companies[t]["sections"][section]["n"],
                     companies[t]["report_date"]] for t in tickers],
        hovertemplate="%{customdata[0]}<br>net tone %{x:+.2f} over %{customdata[1]} sentences"
                      "<br>fiscal year ended %{customdata[2]}<extra></extra>",
    ))
    fig = ui.style(fig, height=60 + 34 * len(tickers), y_title=None)
    fig.update_xaxes(title="net tone (share positive − share negative)", tickformat="+.2f", zeroline=True,
                     zerolinecolor=ui.AXIS)
    ui.show(fig, key=f"tone_{section}")
    ui.source(f"latest 10-K per company, {'Item 7 (MD&A)' if section == 'mdna' else 'Item 1A (Risk Factors)'}, "
              f"SEC EDGAR; sentences scored by {snap['model']}. AEHR in navy.")


left, right = st.columns(2, gap="large")
with left:
    tone_bar("mdna", 1, "Net tone of management's discussion (MD&A), by company")
with right:
    tone_bar("risk_factors", 2, "Net tone of the risk factors, by company")

ui.exhibit(3, "Three model tiers on the same held-out test split")
if results:
    rows = []
    for name in ("baseline", "cnn", "distilbert"):
        r = results.get(name)
        if not r:
            continue
        ab = r.get("ablation") or {}
        rows.append({"Model": {"baseline": "TF-IDF + logistic regression", "cnn": "TextCNN (from scratch, PyTorch)",
                               "distilbert": "DistilBERT fine-tuned (PyTorch)"}[name],
                     "Accuracy": f"{r['accuracy']:.1%}", "Macro-F1": f"{r['macro_f1']:.3f}",
                     "F1 negative": f"{r['per_class']['negative']['f1']:.3f}",
                     "F1 neutral": f"{r['per_class']['neutral']['f1']:.3f}",
                     "F1 positive": f"{r['per_class']['positive']['f1']:.3f}",
                     "All-agree accuracy": f"{ab['accuracy']:.1%}" if ab else "NM",
                     "Train time": f"{r['wall_seconds']:.0f}s on {r['device']}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ui.source(f"projects/filing_tone/results.json, written by `python -m projects.filing_tone train`. Test split: "
              f"{rows and results[next(iter(results))]['test_size']} sentences, 10% of Financial PhraseBank "
              "(50% agreement), stratified by label, seed 42. All-agree: the test sentences every annotator agreed on.")
else:
    st.caption("No results.json yet. Train with `python -m projects.filing_tone train --model distilbert`.")

# ---------------------------------------------------------------- drivers ------------
st.space("small")
st.subheader("The sentences behind the numbers")
st.caption("For each filing, the five sentences the model was most confident were positive and negative, with "
           "its probability. This is the check on the model: if these read wrong, so does the score.")
for t in sorted(companies):
    c = companies[t]
    with st.expander(f"{t} · {c['company']} · 10-K for fiscal year ended {c['report_date']}"):
        st.caption(f"[{c['accession']}]({c['document_url']}), filed {c['filed']}")
        for key, label in (("mdna", "MD&A"), ("risk_factors", "Risk factors")):
            s = c["sections"][key]
            st.markdown(f"**{label}** · {s['n']} sentences · {s['share_positive']:.0%} positive, "
                        f"{s['share_negative']:.0%} negative, {s['share_neutral']:.0%} neutral · net "
                        f"**{s['net_tone']:+.2f}**")
            a, b = st.columns(2, gap="medium")
            with a:
                st.markdown("Most positive")
                for d in s["top_positive"]:
                    st.markdown(f"- `{d['p']:.2f}` {ui.escape_text(d['sentence'])}")
            with b:
                st.markdown("Most negative")
                for d in s["top_negative"]:
                    st.markdown(f"- `{d['p']:.2f}` {ui.escape_text(d['sentence'])}")
        for w in c["warnings"]:
            st.warning(w)
if snap["skipped"]:
    ui.rule()
    for t, why in snap["skipped"].items():
        st.caption(f"{t} not scored: {why}")
ui.source("SEC EDGAR 10-K primary documents; sentence scores from the fine-tuned model. Training data: "
          "Financial PhraseBank (Malo et al., 2014), CC BY-NC-SA 3.0. Student project, not investment advice.")
