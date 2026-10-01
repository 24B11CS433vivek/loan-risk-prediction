"""
batch_page.py - the "Batch Scoring" page of the LoanIQ app.
All scoring logic lives in src/batch_scoring.py; this file is layout only.
"""
import hashlib
import html as _html
import math
import re
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import batch_scoring as bs
import charts
import ui

STATE_KEY = "batch_result"
PARSED_KEY = "batch_parsed"
PAGE_SIZE = 50
RISK_TONE = {"Low": "low", "Medium": "med", "High": "high"}


def _esc(v):
    return _html.escape(str(v), quote=True)


def _donut(counts, total):
    labels = list(charts.RISK_ORDER)
    fig = go.Figure(go.Pie(
        labels=labels, values=[counts.get(k, 0) for k in labels], hole=0.66, sort=False,
        direction="clockwise",
        marker=dict(colors=[charts.RISK_COLOR[k] for k in labels], line=dict(color="#ffffff", width=3)),
        textinfo="percent", textfont=dict(color="#ffffff", size=13),
        hovertemplate="%{label}: %{value:,} applicants (%{percent})<extra></extra>"))
    fig.add_annotation(text="<b>{:,}</b><br><span style='font-size:12px;color:#6b7a99'>applicants</span>".format(total),
                       showarrow=False, font=dict(size=22, color="#0b1437", family=charts.FONT))
    return charts._base(fig, 320, legend=True).update_layout(
        legend=dict(orientation="h", y=-0.04, x=0.5, xanchor="center"))


def _parse_upload(data, sig):
    """Parse + validate once per file (kept in session state so reruns are instant)."""
    cached = st.session_state.get(PARSED_KEY)
    if cached and cached[0] == sig:
        return cached[1]
    validated = bs.validate(bs.read_csv_bytes(data))
    st.session_state[PARSED_KEY] = (sig, validated)
    return validated


def _issue_tables(v):
    summary = bs.summarize_issues(v.issues)
    st.markdown(ui.table(summary.assign(**{c: summary[c].map(_esc) for c in ["Column", "Problem", "Example rows"]})),
                unsafe_allow_html=True)
    with st.expander("Show the first 50 problem cells"):
        detail = v.issues.head(50).rename(columns={"row": "File row", "column": "Column", "problem": "Problem",
                                                    "value": "Value found"})
        detail = detail.assign(**{c: detail[c].map(_esc) for c in ["Column", "Problem", "Value found"]})
        st.markdown(ui.table(detail), unsafe_allow_html=True)
        st.caption("File row 1 is the header, so the first applicant is row 2.")


def _run(v, model, sig):
    t0 = time.perf_counter()
    prog = st.progress(0.0, text="Preparing applicants...")
    try:
        feat = v.features.loc[v.valid]
        feat, inc_m, dti_m = bs.fill_missing(feat)
        prog.progress(0.05, text="Scoring applicants...")
        res = bs.score(feat, model, progress=lambda f: prog.progress(0.05 + 0.95 * f, text="Scoring applicants..."))
        out = bs.build_output(v, feat, inc_m, dti_m, res, model)
    except bs.BatchError as e:
        prog.empty()
        ui.note(_esc(e), "bad")
        return
    except Exception as e:                                   # keep the app alive on anything unexpected
        prog.empty()
        ui.note("Scoring failed: %s. Check the file and try again." % _esc(e), "bad")
        return
    prog.empty()
    st.session_state[STATE_KEY] = {
        "sig": sig, "id": st.session_state.get(STATE_KEY, {}).get("id", 0) + 1, "model": model, "out": out,
        "csv": out.to_csv(index=False).encode("utf-8"), "seconds": time.perf_counter() - t0,
        "skipped": int((~v.valid).sum()), "n_income": int(inc_m.sum()), "n_dti": int(dti_m.sum()),
    }


def _results(r):
    out = r["out"]
    total = len(out)
    counts = out["risk_label"].value_counts().to_dict()
    avg = float(out["risk_score"].mean())

    ui.section("Batch summary", sub="model: " + r["model"])
    cards = [
        ("{:,}".format(total), "Applicants scored", "users", "brand", "", None),
        ("{:.1%}".format(counts.get("Low", 0) / total), "Low risk", "check_circle", "low",
         "{:,} applicants".format(counts.get("Low", 0)), counts.get("Low", 0) / total),
        ("{:.1%}".format(counts.get("Medium", 0) / total), "Medium risk", "warning", "med",
         "{:,} applicants".format(counts.get("Medium", 0)), counts.get("Medium", 0) / total),
        ("{:.1%}".format(counts.get("High", 0) / total), "High risk", "x_circle", "high",
         "{:,} applicants".format(counts.get("High", 0)), counts.get("High", 0) / total),
        ("{:.1f}".format(avg), "Average risk score", "gauge", "violet", "Scale 0 to 100", avg / 100.0),
    ]
    for col, (v, l, i, t, d, b) in zip(st.columns(5, gap="medium"), cards):
        with col:
            st.markdown(ui.stat_card(v, l, i, t, delta=d, delta_dir="", bar=b), unsafe_allow_html=True)

    a, b = st.columns([3, 2], gap="large")
    with a:
        ui.section("Risk mix", sub="share of scored applicants")
        charts.show(_donut(counts, total))
    with b:
        ui.section("Run details", sub="what was done to your file")
        rows = [("Model", r["model"], "cpu"), ("Rows scored", "{:,}".format(total), "check"),
                ("Rows skipped (invalid)", "{:,}".format(r["skipped"]), "minus"),
                ("Income estimated", "{:,} rows".format(r["n_income"]), "wallet"),
                ("Debt-to-income estimated", "{:,} rows".format(r["n_dti"]), "percent"),
                ("Scoring time", "%.1f s" % r["seconds"], "clock")]
        st.markdown('<div class="card">' + "".join(ui.kv_row(k, _esc(v), i) for k, v, i in rows) + "</div>",
                    unsafe_allow_html=True)
        st.write("")
        slug = re.sub(r"[^a-z0-9]+", "_", r["model"].lower()).strip("_")
        st.download_button("Download scored CSV", r["csv"], file_name="loaniq_scored_%s.csv" % slug,
                           mime="text/csv", key="batch_dl")

    ui.section("Scored applicants", sub="risk label, confidence and risk score for each row")
    f1, f2, f3 = st.columns([1, 1, 1], gap="medium")
    flt = f1.selectbox("Risk level", ["All", "High", "Medium", "Low"], key="batch_filter")
    srt = f2.selectbox("Sort by", ["Highest risk score", "Lowest risk score", "Lowest confidence", "File order"],
                       key="batch_sort")
    view = out if flt == "All" else out[out["risk_label"] == flt]
    if srt == "Highest risk score":
        view = view.sort_values("risk_score", ascending=False, kind="stable")
    elif srt == "Lowest risk score":
        view = view.sort_values("risk_score", ascending=True, kind="stable")
    elif srt == "Lowest confidence":
        view = view.sort_values("confidence", ascending=True, kind="stable")
    pages = max(1, math.ceil(len(view) / PAGE_SIZE))
    page_no = f3.number_input("Page (1 to %d)" % pages, 1, pages, 1, 1,
                              key="batch_page_%s_%s_%s" % (r["id"], flt, srt))
    chunk = view.iloc[(page_no - 1) * PAGE_SIZE: page_no * PAGE_SIZE]
    if chunk.empty:
        ui.note("No applicants match this filter.", "info")
    else:
        disp = pd.DataFrame({
            "Row": chunk["row"].astype(int).to_numpy(),
            "Age": chunk["age"].to_numpy(),
            "Employment": chunk["employment_status"].map(_esc).to_numpy(),
            "Income (Rs)": ["{:,.0f}{}".format(v, " *" if e == "Yes" else "")
                            for v, e in zip(chunk["income"], chunk["income_estimated"])],
            "Loan (Rs)": ["{:,.0f}".format(v) for v in chunk["loan_amount"]],
            "Credit score": chunk["credit_score"].to_numpy(),
            "Debt-to-income": ["{:.2f}{}".format(v, " *" if e == "Yes" else "")
                               for v, e in zip(chunk["debt_to_income_ratio"], chunk["dti_estimated"])],
            "Risk": [ui.pill(k, RISK_TONE[k]) for k in chunk["risk_label"]],
            "Confidence": ["{:.0f}%".format(v * 100) for v in chunk["confidence"]],
            "Risk score": ["{:.1f}".format(v) for v in chunk["risk_score"]],
        })
        st.markdown(ui.table(disp), unsafe_allow_html=True)
        first = (page_no - 1) * PAGE_SIZE + 1
        st.caption("Showing %s to %s of %s applicants. * marks a value estimated by the app. "
                   "The downloaded CSV has every row and column." % (format(first, ","),
                                                                      format(first + len(chunk) - 1, ","),
                                                                      format(len(view), ",")))
    ui.note("This is decision support only. It is not a loan approval or rejection, and final decisions need human review.", "warn")


def render(artifact, models, best_name):
    ui.page_header("Batch Scoring", "Upload a CSV of applicants and score every row with one model", "layers", "Portfolio")

    ui.section("Prepare your file", "file", "download the template, fill in your applicants")
    t1, t2 = st.columns([2, 3], gap="large")
    with t1:
        st.download_button("Download CSV template", bs.template_csv(), file_name="loaniq_batch_template.csv",
                           mime="text/csv", key="batch_tpl")
    with t2:
        ui.note("Leave <b>income</b> or <b>debt_to_income_ratio</b> blank on any row and the app estimates it, "
                "the same way as the Assess Applicant page. Extra columns such as an applicant ID are kept in the output.",
                "brand")
    with st.expander("Column guide"):
        st.markdown(ui.table(bs.column_guide().assign(**{"Accepted values": lambda d: d["Accepted values"].map(_esc)})),
                    unsafe_allow_html=True)

    ui.section("Upload and score", "layers")
    names = [m["name"] for m in models]
    u1, u2 = st.columns([3, 2], gap="large")
    with u1:
        up = st.file_uploader("Applicants CSV file", type=["csv"], key="batch_upload")
    with u2:
        chosen = st.selectbox("Scoring model", names, index=names.index(best_name), key="batch_model",
                              format_func=lambda n: n + ("  (recommended)" if n == best_name else ""))
    if up is None:
        ui.note("Upload a CSV to begin. Use the template above for the exact column names.", "info")
        return

    data = up.getvalue()
    sig = hashlib.md5(data).hexdigest()
    try:
        v = _parse_upload(data, sig)
    except bs.BatchError as e:
        ui.note(_esc(e), "bad")
        return

    n_bad = v.n_rows - v.n_valid
    inc_blank = int(v.features.loc[v.valid, "income"].isna().sum())
    dti_blank = int(v.features.loc[v.valid, "debt_to_income_ratio"].isna().sum())
    msg = "Read <b>{:,}</b> rows. <b>{:,}</b> are ready to score".format(v.n_rows, v.n_valid)
    if inc_blank or dti_blank:
        msg += "; income will be estimated for <b>{:,}</b> and debt-to-income for <b>{:,}</b>".format(inc_blank, dti_blank)
    ui.note(msg + ".", "good" if not n_bad else "info")
    if v.extras:
        ui.note("Columns not used by the model are kept in the output: %s." % _esc(", ".join(v.extras)), "info")

    skip = False
    if n_bad:
        ui.note("<b>{:,} of {:,} rows have problems</b> and cannot be scored as they are.".format(n_bad, v.n_rows), "bad")
        _issue_tables(v)
        if v.n_valid:
            skip = st.checkbox("Skip the {:,} invalid rows and score the other {:,}".format(n_bad, v.n_valid),
                               key="batch_skip_%s" % sig[:8])
        else:
            ui.note("No row is valid. Fix the file and upload it again.", "bad")
    blocked = v.n_valid == 0 or (n_bad > 0 and not skip)

    st.write("")
    if st.button("Score applicants", key="batch_run", disabled=blocked):
        _run(v, chosen, sig)

    r = st.session_state.get(STATE_KEY)
    if r and r["sig"] == sig:
        _results(r)
