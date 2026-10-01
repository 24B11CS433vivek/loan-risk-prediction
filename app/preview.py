"""Design preview: run with  streamlit run app/preview.py  to check theme + icons."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import ui
from icons import icon

st.set_page_config(page_title="LoanIQ Design Preview", layout="wide")
ui.inject_theme()

with st.sidebar:
    ui.sidebar_brand()
    st.radio("Navigation", [n for n, _ in ui.NAV], label_visibility="collapsed")
    ui.sidebar_footer(["<b>B.Tech CSE</b>", "Aditya University", "ML Classification Project"])

ui.page_header("Design Preview", "Theme, icons and components used across LoanIQ", "sparkles")
ui.hero("DESIGN SYSTEM", "Smarter lending decisions",
        "A clean indigo interface with SVG icons, readable contrast and responsive cards.")

ui.section("Stat cards", "activity")
c = st.columns(4, gap="medium")
cards = [("12", "Models", "cpu", "brand", "", "up"), ("15", "Features", "layers", "info", "", "up"),
         ("97.2%", "ROC-AUC", "target", "low", "+1.4%", "up"), ("5,098", "Records", "database", "med", "", "up")]
for col, (v, l, i, t, d, dd) in zip(c, cards):
    with col:
        st.markdown(ui.stat_card(v, l, i, t, d, dd), unsafe_allow_html=True)

ui.section("Risk pills and callouts", "shield")
st.markdown(ui.risk_pill("Low") + "&nbsp;&nbsp;" + ui.risk_pill("Medium") + "&nbsp;&nbsp;" + ui.risk_pill("High"),
            unsafe_allow_html=True)
ui.note("<b>Info.</b> Income was estimated from age, education and employment.", "info")
ui.note("<b>Warning.</b> Debt-to-income is above the recommended 40%.", "warn")
ui.note("<b>Good.</b> No previous defaults on record.", "good")
ui.note("<b>High risk.</b> Credit score is below 550.", "bad")

ui.section("Probability bars and key/value rows", "pie")
a, b = st.columns(2, gap="large")
with a:
    st.markdown('<div class="card">' + ui.prob_bars({"Low": .72, "Medium": .2, "High": .08}) + "</div>",
                unsafe_allow_html=True)
with b:
    st.markdown('<div class="card">' + ui.kv_row("Credit Score", 712, "card")
                + ui.kv_row("Education", "None", "education")
                + ui.kv_row("Annual Income", "Rs 60,000", "rupee") + "</div>", unsafe_allow_html=True)

ui.section("Form widgets (check every text is readable)", "sliders")
with st.form("demo"):
    f = st.columns(4, gap="medium")
    f[0].text_input("Age", "35")
    f[1].number_input("Dependents", 0, 10, 1)
    f[2].selectbox("Education", ["None", "Primary School", "High School", "Diploma", "Bachelor", "Master", "PhD"])
    f[3].slider("Debt-to-Income", 0.0, 1.0, 0.28, 0.01)
    st.form_submit_button("Run Assessment")

