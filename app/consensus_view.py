"""
consensus_view.py - the "Consensus" section shown under the Assess Applicant results.
All scoring lives in src/consensus.py; this file is layout only.
"""
import html as _html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import consensus as cs
import ui

TONE = {"Low": "low", "Medium": "med", "High": "high"}


@st.cache_data(show_spinner=False)
def _cached(items):
    return cs.run_all(dict(items))


def _chart(table, majority):
    t = table.sort_values("risk_score", ascending=True)
    labels = [m + ("  (differs)" if l != majority else "") for m, l in zip(t["model"], t["label"])]
    fig = go.Figure()
    for k in cs.ORDER:
        fig.add_bar(name=k, y=labels, x=(t[k] * 100).round(1), orientation="h",
                    marker_color=charts.RISK_COLOR[k], hovertemplate="%{y}<br>" + k + ": %{x:.1f}%<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.3, xaxis=dict(range=[0, 100], title="Probability (%)", gridcolor="#e8edf7"),
                      yaxis=dict(automargin=True))
    return charts._base(fig, max(300, 30 * len(t) + 90), legend=True).update_layout(
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center"))


def render(inputs, chosen_model=None):
    ui.section("Consensus across all models", "layers", "the same applicant scored by every saved model")
    try:
        with st.spinner("Running all models..."):
            table, failed = _cached(tuple(sorted(inputs.items())))
    except Exception as e:
        ui.note("Consensus could not be computed: %s" % _html.escape(str(e)), "bad")
        return
    if failed:
        ui.note("Skipped %d model(s) that could not run: %s" % (len(failed), _html.escape(", ".join(failed))), "warn")
    if len(table) < 3:
        ui.note("Fewer than 3 models ran, so a consensus is not meaningful.", "bad")
        return

    s = cs.summarize(table)
    maj = s["majority"]
    c = st.columns(4, gap="medium")
    with c[0]:
        st.markdown(ui.stat_card(maj + " risk", "Majority vote", "check_circle", TONE[maj],
                                 delta="%d of %d models" % (s["votes"][maj], s["n"])), unsafe_allow_html=True)
    with c[1]:
        st.markdown(ui.stat_card("%.0f%%" % (s["agreement"] * 100), "Agreement", "percent",
                                 "low" if s["agreement"] >= cs.AGREEMENT_THRESHOLD else "high",
                                 bar=s["agreement"]), unsafe_allow_html=True)
    with c[2]:
        v = s["votes"]
        st.markdown(ui.stat_card("%d / %d / %d" % (v["Low"], v["Medium"], v["High"]), "Votes: Low / Medium / High",
                                 "pie", "violet"), unsafe_allow_html=True)
    with c[3]:
        st.markdown(ui.stat_card("%.1f" % s["mean_score"], "Average risk score", "activity", "brand"), unsafe_allow_html=True)

    st.write("")
    if s["review"]:
        ui.note("<b>Models disagree, manual review recommended.</b> %s." % _html.escape("; ".join(s["reasons"]).capitalize()), "warn")
    else:
        ui.note("The models agree. %d of %d predict <b>%s</b> risk." % (s["votes"][maj], s["n"], maj), "good")
    if chosen_model and chosen_model in set(table["model"]):
        row = table[table["model"] == chosen_model].iloc[0]
        if row["label"] != maj:
            ui.note("The model used above (<b>%s</b>) predicts <b>%s</b>, which differs from the majority." % (
                _html.escape(chosen_model), row["label"]), "info")

    a, b = st.columns([3, 2], gap="large")
    with a:
        ui.section("Probabilities per model", "pie", "sorted by risk score")
        charts.show(_chart(table, maj))
    with b:
        ui.section("Model by model", "cpu", "models that differ from the majority are marked")
        t = table.sort_values("risk_score", ascending=False)
        df = pd.DataFrame({
            "Model": [("<b>%s</b>" % _html.escape(m)) if l != maj else _html.escape(m) for m, l in zip(t["model"], t["label"])],
            "Predicted": [ui.risk_pill(l) for l in t["label"]],
            "Confidence": ["{:.0%}".format(x) for x in t["confidence"]],
            "Risk score": ["%.1f" % x for x in t["risk_score"]],
            "Vs majority": [ui.pill("Agrees", "low") if l == maj else ui.pill("Disagrees", "high") for l in t["label"]],
        })
        st.markdown(ui.table(df), unsafe_allow_html=True)
    ui.note("Majority vote uses each model's top class. A tie is broken by average probability. Review is flagged when agreement "
            "is under %d%% or the top two levels are within %d vote of each other." % (cs.AGREEMENT_THRESHOLD * 100, cs.SPLIT_MARGIN), "info")
