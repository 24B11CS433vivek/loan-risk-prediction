"""
whatif_page.py - the "What-If Simulator" page of the LoanIQ app.
All scoring and search logic lives in src/whatif.py; this file is layout only.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import ui
import whatif as wi
from preprocessing import EDUCATION_DISPLAY

KEYS = {"credit_score": "wi_credit", "loan_amount": "wi_loan", "loan_term": "wi_term",
        "income": "wi_income", "existing_debt": "wi_debt", "previous_defaults": "wi_defaults"}
SIG_KEY = "wi_baseline_sig"


def inr(v):
    return "Rs {:,.0f}".format(v)


def _fmt(lever, v):
    if lever == "credit_score":
        return "%d" % v
    if lever == "loan_term":
        return "%d months" % v
    return inr(v)


def _load_baseline_into_state(base):
    for field, key in KEYS.items():
        st.session_state[key] = base[field]


def _reset(base):
    _load_baseline_into_state(base)


def _compare_chart(base_p, new_p):
    labels = list(charts.RISK_ORDER)
    fig = go.Figure()
    fig.add_bar(name="Baseline", x=labels, y=[base_p.get(k, 0) * 100 for k in labels],
                marker_color="#cbd5e8", hovertemplate="%{x}: %{y:.1f}%<extra>Baseline</extra>")
    fig.add_bar(name="Scenario", x=labels, y=[new_p.get(k, 0) * 100 for k in labels],
                marker_color=[charts.RISK_COLOR[k] for k in labels],
                hovertemplate="%{x}: %{y:.1f}%<extra>Scenario</extra>")
    fig.update_layout(barmode="group", bargap=0.35, yaxis=dict(title="Probability (%)", range=[0, 100],
                      gridcolor="#e8edf7", zeroline=False), xaxis=dict(showgrid=False))
    return charts._base(fig, 280, legend=True).update_layout(
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))


@st.cache_data(show_spinner=False)
def _cached_search(applicant_items, model):
    return wi.find_low_risk(dict(applicant_items), model)


def render(artifact, models, best_name):
    ui.page_header("What-If Simulator", "Move the sliders and watch the risk score, probabilities and EMI respond",
                   "sliders", "Scenario analysis")

    last = st.session_state.get("last")
    base = wi.normalise_baseline(last["inputs"] if last else None)
    from_last = bool(last)
    names = [m["name"] for m in models]
    default_model = last["result"]["model_used"] if last and last["result"]["model_used"] in names else best_name

    sig = tuple(sorted((k, str(v)) for k, v in base.items()))
    if st.session_state.get(SIG_KEY) != sig or any(k not in st.session_state for k in KEYS.values()):
        _load_baseline_into_state(base)
        st.session_state[SIG_KEY] = sig

    ui.note("Baseline: <b>%s</b>. Age %d, %s, %s, %d dependents are held fixed. Debt-to-income is recalculated from income, debt and loan."
            % ("your last assessment" if from_last else "default applicant", base["age"], base["employment_status"],
               EDUCATION_DISPLAY.get(base["education"], base["education"]),
               base["dependents"]), "brand")

    top = st.columns([2, 1], gap="large")
    with top[0]:
        model = st.selectbox("Prediction model", names, index=names.index(default_model), key="wi_model",
                             format_func=lambda n: n + ("  (recommended)" if n == best_name else ""))
    with top[1]:
        st.write("")
        st.write("")
        st.button("Reset to baseline", on_click=_reset, args=(base,), key="wi_reset")

    ui.section("Adjust the applicant", "sliders")
    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.slider("Credit score", 300, 900, step=1, key=KEYS["credit_score"])
        st.slider("Loan amount (Rs)", 1000, int(max(200000, base["loan_amount"] * 1.5)), step=500, key=KEYS["loan_amount"])
        st.select_slider("Loan term (months)", options=wi.TERMS, key=KEYS["loan_term"])
    with s2:
        st.slider("Annual income (Rs)", 0, int(max(500000, base["income"] * 1.5)), step=1000, key=KEYS["income"])
        st.slider("Existing debt (Rs)", 0, int(max(200000, base["existing_debt"] * 1.5)), step=500, key=KEYS["existing_debt"])
        st.slider("Previous defaults", 0, 20, step=1, key=KEYS["previous_defaults"])

    scen = dict(base)
    for field, key in KEYS.items():
        scen[field] = st.session_state[key]

    try:
        b = wi.score_one(base, model)
        s = wi.score_one(scen, model)
    except Exception as e:
        ui.note("Scoring failed: %s. Check that the model files exist and try again." % e, "bad")
        return

    d_score = s["risk_score"] - b["risk_score"]
    d_emi = s["emi"] - b["emi"]
    st.write("")
    c = st.columns(4, gap="medium")
    tone = {"Low": "low", "Medium": "med", "High": "high"}
    with c[0]:
        same = s["label"] == b["label"]
        st.markdown(ui.stat_card(s["label"] + " risk", "Predicted class", "assess", tone[s["label"]],
                                 delta="Unchanged from baseline" if same else "Baseline: %s" % b["label"],
                                 delta_dir="up" if (same or s["risk_score"] <= b["risk_score"]) else "down"),
                    unsafe_allow_html=True)
    with c[1]:
        st.markdown(ui.stat_card("%.1f" % s["risk_score"], "Risk score", "activity", "brand",
                                 delta="%+.1f vs baseline (%.1f)" % (d_score, b["risk_score"]),
                                 delta_dir="up" if d_score <= 0 else "down"), unsafe_allow_html=True)
    with c[2]:
        st.markdown(ui.stat_card(inr(s["emi"]), "Monthly EMI", "wallet", "violet",
                                 delta="%sRs {:,.0f} vs baseline".format(abs(d_emi)) % ("+" if d_emi >= 0 else "-"),
                                 delta_dir="up" if d_emi <= 0 else "down"), unsafe_allow_html=True)
    with c[3]:
        st.markdown(ui.stat_card("%.2f" % s["dti"], "Debt-to-income", "percent", "cyan",
                                 delta="%+.2f vs baseline" % (s["dti"] - b["dti"]),
                                 delta_dir="up" if s["dti"] <= b["dti"] else "down"), unsafe_allow_html=True)

    a, bcol = st.columns(2, gap="large")
    with a:
        ui.section("Risk score gauge", sub="0 = safest, 100 = riskiest")
        charts.show(charts.gauge(s["risk_score"], s["label"]))
    with bcol:
        ui.section("Class probabilities", "pie", "baseline vs scenario")
        charts.show(_compare_chart(b["probabilities"], s["probabilities"]))

    # ------------------------------------------------------------ what would make it Low
    ui.section("What would make this applicant Low risk?", "target", "smallest realistic changes from the current slider values")
    with st.spinner("Searching changes..."):
        try:
            rows, status = _cached_search(tuple(sorted(scen.items())), model)
        except Exception as e:
            rows, status = [], "error"
            ui.note("The search failed: %s" % e, "bad")

    if status == "already_low":
        ui.note("The current scenario is already predicted <b>Low risk</b> by this model. Move the sliders the other way to see where it flips.", "good")
    elif status == "none":
        ui.note("No combination of a higher credit score (up to +150), a smaller loan, a longer term and lower debt makes this model predict Low. "
                "Income and previous defaults are not searched. Try a different model or adjust them above.", "warn")
    elif status == "found":
        table_rows = []
        for i, r in enumerate(rows, start=1):
            parts = ["%s: %s to %s" % (wi.LEVER_LABEL[k], _fmt(k, o), _fmt(k, n)) for k, o, n in r["changes"]]
            table_rows.append((i, "<br>".join(parts), "%.1f" % r["risk_score"],
                               "%+.1f" % (r["risk_score"] - s["risk_score"]), inr(r["emi"])))
        df = pd.DataFrame(table_rows, columns=["#", "Change", "New risk score", "vs now", "New EMI"])
        st.markdown(ui.table(df), unsafe_allow_html=True)
        ui.note("Ranked by overall size of change (credit points, share of loan or debt removed, extra months). "
                "Each option makes the model predict Low risk. These are model outputs, not lending advice.", "info")
