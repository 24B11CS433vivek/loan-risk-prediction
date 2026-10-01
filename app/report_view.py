"""
report_view.py - the "Download PDF report" button on the Assess Applicant results.
All PDF logic lives in src/pdf_report.py; this file is layout only.
"""
import html as _html
from datetime import datetime

import streamlit as st

import ui
from preprocessing import EDUCATION_DISPLAY

STATE_KEY = "pdf_report"


def _summary_rows(inp, emi):
    inr = lambda v: "Rs {:,.0f}".format(v)
    return [("Age", inp["age"]), ("Employment", inp["employment_status"]),
            ("Education", EDUCATION_DISPLAY.get(inp["education"], inp["education"])), ("Dependents", inp["dependents"]),
            ("Annual income", inr(inp["income"])), ("Loan amount", inr(inp["loan_amount"])),
            ("Loan term", "%d months" % inp["loan_term"]), ("Monthly EMI", inr(emi)),
            ("Credit score", inp["credit_score"]), ("Existing debt", inr(inp["existing_debt"])),
            ("Debt-to-income", "%.2f" % inp["debt_to_income_ratio"]), ("Previous defaults", inp["previous_defaults"])]


def render(inp, res, emi, models):
    try:
        import pdf_report
    except ImportError as e:
        ui.note("PDF export needs reportlab. Run <b>pip install -r requirements.txt</b> and restart the app. (%s)" % _html.escape(str(e)), "warn")
        return
    sig = repr((sorted(inp.items()), res["model_used"], round(float(emi), 2)))
    cached = st.session_state.get(STATE_KEY)
    if not cached or cached["sig"] != sig:
        try:
            with st.spinner("Preparing PDF report..."):
                pdf = pdf_report.build_report(_summary_rows(inp, emi), res, models)
        except Exception as e:                                  # never let a report failure break the results page
            ui.note("The PDF report could not be created: %s" % _html.escape(str(e)), "bad")
            return
        cached = {"sig": sig, "pdf": pdf, "name": "loaniq_assessment_%s.pdf" % datetime.now().strftime("%Y%m%d_%H%M%S")}
        st.session_state[STATE_KEY] = cached
    st.download_button("Download PDF report", cached["pdf"], file_name=cached["name"], mime="application/pdf", key="pdf_dl")
