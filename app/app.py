import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import joblib
import pandas as pd
import streamlit as st

import charts
import batch_page
import report_view
import consensus_view
import whatif_page
import ui
from icons import icon
from predict import (predict_risk, list_models, estimate_income, max_affordable_loan)
from preprocessing import (EDUCATION_CATEGORIES, EDUCATION_DISPLAY, EMPLOYMENT_CATEGORIES,
                           estimate_dti, monthly_emi)

st.set_page_config(page_title="LoanIQ - Risk Intelligence", page_icon=os.path.join(HERE, "assets", "favicon.svg"),
                   layout="wide", initial_sidebar_state="expanded")
ui.inject_theme()

MODEL_PATH = os.path.join(HERE, "..", "models", "loan_risk_pipeline.pkl")
DATA_PATH = os.path.join(HERE, "..", "data", "raw", "loan_data.csv")
FIG_DIR = os.path.join(HERE, "..", "reports", "figures")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None


def edu_label(v):
    return EDUCATION_DISPLAY.get(v, v)


def inr(v):
    return "Rs {:,.0f}".format(v)


def go_to(page_name):
    st.session_state["page"] = page_name


artifact = load_artifact()
data = load_data()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    ui.sidebar_brand()
    page = st.radio("Navigation", [n for n, _ in ui.NAV], key="page", label_visibility="collapsed")
    ui.sidebar_footer(["<b>B.Tech CSE</b>", "Aditya University", "ML Classification Project"])

if artifact is None:
    ui.page_header("Model not found", "Train the models first", "warning")
    ui.note("Run <b>python src/train.py</b> from the project folder, then refresh this page.", "bad")
    st.stop()

models = list_models()
metrics = artifact["test_metrics"]
best_name = artifact["best_model_name"]

# ================================================================== DASHBOARD
if page == "Dashboard":
    ui.page_header("Risk Intelligence Dashboard", "Applicant loan risk classification: model performance and portfolio overview", "dashboard", "Overview")
    ui.hero("Credit risk screening", "Classify applicant risk with a model you can audit",
            "Compare %d trained models, auto-estimate income and debt ratio, and see which factors move an applicant "
            "between Low, Medium and High risk." % len(models),
            stats=[(best_name, "Recommended model"), ("{:.1%}".format(metrics["accuracy"]), "Test accuracy"),
                   ("{:.3f}".format(metrics["roc_auc"]), "ROC-AUC"), ("{:,}".format(artifact["dataset_info"]["rows"]), "Records")])
    st.write("")
    st.button("New assessment", on_click=go_to, args=("Assess Applicant",), key="cta")

    ui.section("Portfolio overview", sub="dataset and training summary")
    cc = artifact["dataset_info"]["class_counts"]
    c = st.columns(4, gap="medium")
    items = [(str(len(models)), "Models trained", "cpu", "brand"),
             (str(len(artifact["model_features"])), "Model features", "layers", "violet"),
             ("{:,}".format(artifact["dataset_info"]["rows"]), "Training records", "database", "info"),
             ("{:.0%}".format(cc.get("High", 0) / max(1, sum(cc.values()))), "High-risk share", "warning", "high")]
    for col, (v, l, i, t) in zip(c, items):
        with col:
            st.markdown(ui.stat_card(v, l, i, t), unsafe_allow_html=True)

    # ---- model picker + selected model performance
    ui.section("Selected model performance", sub="held-out test set, macro averaged")
    by_name = {m["name"]: m for m in models}
    ref = by_name[best_name]
    names = [m["name"] for m in models]
    pk, meta = st.columns([2, 3], gap="large")
    with pk:
        chosen = st.selectbox("Select model", names, index=names.index(best_name), key="dash_model",
                              format_func=lambda n: n + ("  (recommended)" if n == best_name else ""))
    sel = by_name[chosen]
    rank = names.index(chosen) + 1
    with meta:
        tags = ui.pill("Test F1 rank %d of %d" % (rank, len(models)), "gray")
        tags += ui.pill("Recommended", "brand") if sel["selected"] else ""
        tags += ui.pill("Tuned", "violet") if sel["tuned"] else ""
        tags += ui.pill("ROC-AUC {:.3f}".format(sel["roc_auc"]), "info")
        st.markdown('<div class="model-meta">' + tags + "</div>", unsafe_allow_html=True)

    def delta_for(key):
        if sel["name"] == ref["name"]:
            return "Recommended model", "up"
        d = (sel[key] - ref[key]) * 100
        return "%+.1f pts vs recommended" % d, ("up" if d >= 0 else "down")

    perf = [("accuracy", "Accuracy", "#2563eb"), ("precision", "Precision", "#06b6d4"),
            ("recall", "Recall", "#f59e0b"), ("f1_macro", "F1 score", "#7c3aed")]
    c = st.columns(4, gap="medium")
    for col, (k, label, color) in zip(c, perf):
        dtxt, ddir = delta_for(k)
        with col:
            charts.show(charts.ring(sel[k], color, label))
            st.markdown('<div class="kpi-d %s" style="text-align:center;margin-top:-.4rem;">%s</div>' % (ddir, dtxt),
                        unsafe_allow_html=True)

    a, b = st.columns(2, gap="large")
    with a:
        ui.section("Metric radar", sub=chosen + " vs recommended")
        charts.show(charts.radar(sel, ref))
    with b:
        ui.section("Per-class performance", sub="precision, recall and F1 for each risk level")
        cm = artifact["confusion_matrices"].get(chosen)
        if cm is not None:
            charts.show(charts.per_class(cm, artifact["class_names"]))

    a, b = st.columns(2, gap="large")
    with a:
        ui.section("Confusion matrix", sub=chosen + ", share of each actual class")
        if cm is not None:
            charts.show(charts.confusion(cm, artifact["class_names"]))
    with b:
        ui.section("Accuracy vs ROC-AUC", sub="bubble size = F1, purple = selected")
        charts.show(charts.bubble(models, chosen))

    ui.section("All models at a glance", sub="every metric, every model")
    charts.show(charts.model_heatmap(models, chosen))

    left, right = st.columns([2, 3], gap="large")
    with left:
        ui.section("Applicant risk mix", sub="training data")
        charts.show(charts.class_donut(cc))
    with right:
        ui.section("Model leaderboard", sub="test F1 score, top 8")
        charts.show(charts.leaderboard(models, chosen))

    ui.section("How it works")
    c = st.columns(4, gap="medium")
    steps = [("1. Enter details", "Personal and financial information. Leave income or debt ratio to the auto-estimator.", "user", "brand"),
             ("2. Pipeline runs", "Same preprocessing and derived ratios used during training.", "cpu", "cyan"),
             ("3. Model scores", "Choose any of the %d models to predict class probabilities." % len(models), "sliders", "violet"),
             ("4. Explained result", "Risk level, gauge, drivers and practical tips.", "insights", "low")]
    for col, (t, d, i, tone) in zip(c, steps):
        with col:
            st.markdown(ui.feature_card(t, d, i, tone), unsafe_allow_html=True)

    ui.section("Risk classes")
    c = st.columns(3, gap="medium")
    guide = [("Low risk", "Strong credit, stable income, manageable debt and no prior defaults.", "check_circle", "low"),
             ("Medium risk", "Mixed signals: moderate credit or elevated debt. Needs closer review.", "warning", "med"),
             ("High risk", "Weak credit, heavy debt, no stable income or past defaults.", "x_circle", "high")]
    for col, (t, d, i, tone) in zip(c, guide):
        with col:
            st.markdown(ui.feature_card(t, d, i, tone), unsafe_allow_html=True)

# ================================================================== ASSESS
elif page == "Assess Applicant":
    ui.page_header("Applicant Risk Assessment", "Fill in the details, or let the app estimate income and debt ratio", "assess")

    ui.section("Personal profile", "user")
    p = st.columns(4, gap="medium")
    age = p[0].number_input("Age", 18, 90, 35, 1)
    employment = p[1].selectbox("Employment status", EMPLOYMENT_CATEGORIES, index=EMPLOYMENT_CATEGORIES.index("Employed"))
    education = p[2].selectbox("Education level", EDUCATION_CATEGORIES, index=EDUCATION_CATEGORIES.index("Bachelor"),
                               format_func=edu_label)
    dependents = p[3].number_input("Dependents", 0, 10, 1, 1)

    ui.section("Income", "wallet")
    i1, i2 = st.columns([1, 2], gap="large")
    auto_income = i1.toggle("Estimate income automatically", value=True,
                            help="Uses age, education, employment and dependents.")
    if auto_income:
        est = estimate_income(age, education, employment, dependents)
        if est is None:
            i2.markdown('<div class="note warn">Income estimator not found. Switch the toggle off and type the income.</div>',
                        unsafe_allow_html=True)
            income = 0
        else:
            income = est["estimate"]
            i2.markdown(ui._flat(
                '<div class="est-box"><div class="est-v">%s per year</div>'
                '<div class="est-s">Likely range %s to %s</div></div>' % (inr(income), inr(est["low"]), inr(est["high"]))),
                unsafe_allow_html=True)
    else:
        income = i2.number_input("Annual income (Rs)", 0, 100000000, 60000, 1000)

    ui.section("Loan and credit details", "card")
    f = st.columns(3, gap="medium")
    loan_amount = f[0].number_input("Loan amount (Rs)", 1000, 100000000, 30000, 500)
    loan_term = f[1].selectbox("Loan term (months)", [12, 24, 36, 48, 60, 72, 84], index=2)
    credit_score = f[2].number_input("Credit score", 300, 900, 700, 1)
    g = st.columns(3, gap="medium")
    existing_debt = g[0].number_input("Existing debt (Rs)", 0, 100000000, 5000, 500)
    previous_defaults = g[1].number_input("Previous defaults", 0, 20, 0, 1)
    auto_dti = g[2].toggle("Calculate debt-to-income automatically", value=True)

    auto_value = float(estimate_dti(income, existing_debt, loan_amount, loan_term))
    if auto_dti:
        dti = round(auto_value, 2)
        ui.note("Debt-to-income ratio is calculated as <b>%.2f</b> (monthly EMI on existing debt and this loan, divided by monthly income)." % dti, "brand")
    else:
        dti = st.slider("Debt-to-income ratio", 0.0, 1.0, round(min(auto_value, 1.0), 2), 0.01)

    cap = max_affordable_loan(income, existing_debt, loan_term)
    emi = float(monthly_emi(loan_amount, loan_term))
    ui.note("Monthly EMI for this loan is about <b>%s</b>. The largest loan that keeps debt-to-income under 40%% is about <b>%s</b>." % (inr(emi), inr(cap)), "info")

    ui.section("Model", "cpu")
    names = [m["name"] for m in models]
    chosen = st.selectbox("Prediction model", names, index=names.index(best_name),
                          format_func=lambda n: n + ("  (recommended)" if n == best_name else ""))
    st.write("")
    run = st.button("Run assessment", key="run")

    if run:
        errors = []
        if loan_amount < 1000:
            errors.append("Loan amount must be at least Rs 1,000.")
        if not auto_income and income <= 0:
            errors.append("Income must be greater than zero, or switch on automatic estimation.")
        for e in errors:
            ui.note(e, "bad")
        if not errors:
            payload = {"age": age, "income": income, "dependents": dependents, "employment_status": employment,
                       "education": education, "loan_amount": loan_amount, "loan_term": loan_term,
                       "credit_score": credit_score, "existing_debt": existing_debt,
                       "debt_to_income_ratio": dti, "previous_defaults": previous_defaults}
            with st.spinner("Scoring applicant..."):
                st.session_state["last"] = {"result": predict_risk(payload, model_name=chosen),
                                            "inputs": payload, "emi": emi}

    last = st.session_state.get("last")
    if last:
        res, inp = last["result"], last["inputs"]
        risk = res["risk_label"]
        tone = {"Low": "v-low", "Medium": "v-med", "High": "v-high"}[risk]
        st.write("")
        ui.html("""
        <div class="verdict %s">
          <div><div class="vk">RISK ASSESSMENT</div><div class="vt">%s Risk</div><div class="vs">%s</div></div>
          <div class="vmeta">
            <div class="vm"><b>%.0f%%</b><span>Confidence</span></div>
            <div class="vm"><b>%.1f</b><span>Risk score</span></div>
            <div class="vm"><b>%s</b><span>Model</span></div>
          </div>
        </div>""" % (tone, risk, res["message"], res["confidence"] * 100, res["risk_score"], res["model_used"]))
        report_view.render(inp, res, last["emi"], models)

        a, b = st.columns(2, gap="large")
        with a:
            ui.section("Risk score gauge", sub="0 = safest, 100 = riskiest")
            charts.show(charts.gauge(res["risk_score"], risk))
        with b:
            ui.section("Probability by class", "pie")
            charts.show(charts.donut(res["probabilities"]))

        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            ui.section("What drives this result", "trending_up", "impact vs a typical applicant")
            charts.show(charts.contributions(res["contributions"]))
        with c2:
            ui.section("Suggestions", "insights")
            for t in res["tips"]:
                ui.note(t, "good" if t.startswith("No major") else "warn")
            ui.section("Applicant summary", "file")
            rows = [("Age", inp["age"], "calendar"), ("Employment", inp["employment_status"], "briefcase"),
                    ("Education", edu_label(inp["education"]), "education"),
                    ("Annual income", inr(inp["income"]), "rupee"), ("Loan amount", inr(inp["loan_amount"]), "wallet"),
                    ("Loan term", "%d months" % inp["loan_term"], "clock"), ("Credit score", inp["credit_score"], "card"),
                    ("Debt-to-income", "%.2f" % inp["debt_to_income_ratio"], "percent"),
                    ("Previous defaults", inp["previous_defaults"], "flag")]
            st.markdown('<div class="card">' + "".join(ui.kv_row(k, v, i) for k, v, i in rows) + "</div>",
                        unsafe_allow_html=True)
        consensus_view.render(inp, res["model_used"])

        ui.note("This is decision support only. It is not a loan approval or rejection, and final decisions need human review.", "warn")

# ================================================================== COMPARE MODELS
elif page == "Compare Models":
    ui.page_header("Compare Models", "How all %d trained models perform on the held-out test set" % len(models), "compare")
    ui.section("Leaderboard", "award", "sorted by F1 score")
    df = pd.DataFrame([{
        "Model": m["name"], "Accuracy": "{:.3f}".format(m["accuracy"]), "F1 (macro)": "{:.3f}".format(m["f1_macro"]),
        "ROC-AUC": "{:.3f}".format(m["roc_auc"]), "Tuned": "Yes" if m["tuned"] else "No",
        "Status": '<span class="pill brand">Selected</span>' if m["selected"] else ""} for m in models])
    st.markdown(ui.table(df), unsafe_allow_html=True)

    ui.section("Heat map", sub="all five test metrics")
    charts.show(charts.model_heatmap(models, best_name))
    ui.section("Metric comparison", "compare")
    charts.show(charts.metric_bars(models))
    ui.section("Cross-validation vs test", sub="a large gap suggests overfitting")
    charts.show(charts.cv_vs_test(artifact["comparison_df"], models))

    ui.section("Look inside one model", "search")
    pick = st.selectbox("Model", [m["name"] for m in models], index=0, key="cmp_pick")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Confusion matrix** (test set)")
        cm = artifact["confusion_matrices"].get(pick)
        if cm is not None:
            charts.show(charts.confusion(cm, artifact["class_names"]))
    with b:
        st.markdown("**ROC curves** (selected model: %s)" % best_name)
        charts.show(charts.roc(artifact["roc_data"]))
    with st.expander("Cross-validation details"):
        st.markdown(ui.table(artifact["comparison_df"]), unsafe_allow_html=True)

# ================================================================== INSIGHTS
elif page == "Insights":
    ui.page_header("Insights", "What the data and the model say about loan risk", "insights")
    t1, t2 = st.tabs(["Model drivers", "Data explorer"])
    with t1:
        ui.section("Feature importance", "target", "permutation importance on the test set")
        charts.show(charts.importance(artifact["permutation_importance"]))
        ui.note("Importance is how much the model score drops when a feature is shuffled. Higher means the model relies on it more.", "info")
        shown = [f for f in ["confusion_matrix.png", "roc_curves.png", "feature_importance.png"] if os.path.exists(os.path.join(FIG_DIR, f))]
        if shown:
            with st.expander("Static evaluation figures"):
                cols = st.columns(len(shown))
                for col, fn in zip(cols, shown):
                    col.image(os.path.join(FIG_DIR, fn), use_column_width=True)
    with t2:
        if data is None:
            ui.note("Dataset not found at data/raw/loan_data.csv.", "bad")
        else:
            ui.section("Numeric features by risk class", "activity")
            labels = {"credit_score": "Credit score", "income": "Annual income", "loan_amount": "Loan amount",
                      "debt_to_income_ratio": "Debt-to-income ratio", "age": "Age", "existing_debt": "Existing debt"}
            col = st.selectbox("Feature", list(labels), format_func=lambda k: labels[k])
            charts.show(charts.box_by_risk(data, col, labels[col]))
            x, y = st.columns(2, gap="large")
            with x:
                ui.section("Risk by education", "education")
                d = data.copy()
                d["education"] = d["education"].map(edu_label)
                charts.show(charts.risk_mix(d, "education", [edu_label(e) for e in EDUCATION_CATEGORIES]))
            with y:
                ui.section("Risk by employment", "briefcase")
                charts.show(charts.risk_mix(data, "employment_status", EMPLOYMENT_CATEGORIES))

# ================================================================== BATCH SCORING
elif page == "Batch Scoring":
    batch_page.render(artifact, models, best_name)

# ================================================================== WHAT-IF SIMULATOR
elif page == "What-If Simulator":
    whatif_page.render(artifact, models, best_name)

# ================================================================== ABOUT
elif page == "About":
    ui.page_header("About This Project", "B.Tech CSE Machine Learning Project, Aditya University", "info")
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.section("Problem statement", "flag")
        st.markdown('<div class="card"><div class="card-d" style="font-size:.92rem;">Financial institutions lose money to loan defaults, '
                    'and manual credit assessment is slow and inconsistent. This system classifies applicants as Low, Medium or High risk '
                    'from financial and personal indicators, and explains which factors drive each decision.</div></div>',
                    unsafe_allow_html=True)
        ui.section("ML workflow", "layers")
        wf = [("Data generation", "5,098 records: 98 original plus 5,000 realistic synthetic applicants", "database"),
              ("Preprocessing", "Median imputation, scaling, ordinal encoding and 4 derived ratio features", "sliders"),
              ("Stratified split", "80% train, 20% test with class balance preserved", "pie"),
              ("Cross-validation", "5-fold stratified CV on training data only", "refresh"),
              ("Model selection", "12 models compared; best chosen by CV F1 macro, then tuned", "cpu"),
              ("Final evaluation", "Single pass on the held-out test set", "check_circle"),
              ("Explainability", "Per-applicant impact against a typical applicant", "insights")]
        st.markdown('<div class="card">' + "".join(ui.kv_row(t, d, i) for t, d, i in wf) + "</div>", unsafe_allow_html=True)
        ui.note("<b>Ethical note.</b> This tool supports decisions only. Final lending decisions must involve human review and comply with financial regulations.", "brand")
    with right:
        ui.section("Models compared", "cpu")
        st.markdown('<div class="card">' + "".join(
            ui.kv_row(m["name"], "Selected" if m["selected"] else "{:.3f}".format(m["f1_macro"])) for m in models) + "</div>",
            unsafe_allow_html=True)
        ui.section("Tech stack", "zap")
        pills = "".join('<span class="pill brand" style="margin:.2rem;">%s</span>' % s for s in
                        ["Python", "scikit-learn", "XGBoost", "Pandas", "NumPy", "Plotly", "Streamlit", "Joblib"])
        st.markdown('<div class="card">' + pills + "</div>", unsafe_allow_html=True)
        ui.section("Limitations", "warning")
        st.markdown('<div class="card">' + "".join(ui.kv_row(x, "", "minus") for x in [
            "Mostly synthetic data", "Local deployment only", "No fairness or bias audit yet",
            "Income estimate is approximate"]) + "</div>", unsafe_allow_html=True)
        ui.section("Built by", "user")
        st.markdown('<div class="card"><div class="stat-v" style="font-size:1.15rem;">Talagaturi Vivek</div>'
                    '<div class="card-d">B.Tech CSE, Aditya University<br>Roll No: 24B11CS433</div></div>', unsafe_allow_html=True)
