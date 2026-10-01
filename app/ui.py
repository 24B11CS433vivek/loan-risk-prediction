"""
ui.py - reusable UI building blocks for the LoanIQ app.
All HTML is emitted on a single line (no leading spaces) so Streamlit's
markdown parser never turns it into a code block.
"""
import os
import streamlit as st
from icons import icon, icon_uri

_HERE = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(_HERE, "assets", "theme.css")

# (page title, icon name)
NAV = [
    ("Dashboard",        "dashboard"),
    ("Assess Applicant", "assess"),
    ("What-If Simulator", "sliders"),
    ("Batch Scoring",    "layers"),
    ("Compare Models",   "compare"),
    ("Insights",         "insights"),
    ("About",            "info"),
]

RISK_COLOR = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#f43f5e"}
RISK_TONE  = {"Low": "low", "Medium": "med", "High": "high"}
TONE_HEX   = {"brand": "#2563eb", "low": "#059669", "med": "#d97706",
              "high": "#e11d48", "info": "#0284c7", "violet": "#7c3aed", "cyan": "#0891b2"}


def _flat(html):
    """Collapse a multi-line HTML string into one line."""
    return "".join(line.strip() for line in html.strip().splitlines())


def html(markup):
    st.markdown(_flat(markup), unsafe_allow_html=True)


def inject_theme():
    """Load theme.css and add one SVG icon per sidebar navigation item."""
    with open(CSS_PATH, encoding="utf-8") as f:
        css = f.read()
    nav_css = ""
    for i, (_, ic) in enumerate(NAV, start=1):
        idle = icon_uri(ic, "#7c8bab")
        nav_css += (
            'section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(%d)::before'
            '{background-image:url("%s");}' % (i, idle)
        )
        active = icon_uri(ic, "#ffffff")
        nav_css += (
            'section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type(%d):has(input:checked)::before'
            '{background-image:url("%s");}' % (i, active)
        )
    st.markdown("<style>" + css + nav_css + "</style>", unsafe_allow_html=True)


def sidebar_brand():
    html(f"""
    <div class="brand">
      <div class="brand-logo">{icon("bank", 18, "#ffffff")}</div>
      <div><div class="brand-name">LoanIQ</div><div class="brand-tag">Risk Intelligence</div></div>
    </div>
    <div class="side-label">Menu</div>
    """)


def sidebar_footer(lines):
    body = "<br>".join(lines)
    html(f'<div class="side-foot">{body}</div>')


def page_header(title, subtitle, icon_name="dashboard", eyebrow="LoanIQ"):
    html(f"""
    <div class="ph">
      <div class="ph-eyebrow">{eyebrow}</div>
      <div class="ph-title">{title}</div><div class="ph-sub">{subtitle}</div>
    </div>
    <div class="ph-line"></div>
    """)


def section(title, icon_name="layers", sub=""):
    s = f'<span class="sec-s">{sub}</span>' if sub else ""
    html(f"""
    <div class="sec"><div class="sec-t">{title}</div>{s}</div>
    """)


def stat_card(value, label, icon_name="activity", tone="brand", delta="", delta_dir="up", bar=None):
    """KPI card. `bar` (0..1) draws a thin progress bar under the value."""
    d = f'<div class="kpi-d {delta_dir}">{delta}</div>' if delta else ""
    b = f'<div class="kpi-bar"><span style="width:{max(0, min(1, bar)) * 100:.1f}%;"></span></div>' if bar is not None else ""
    return _flat(f"""
    <div class="card accent-{tone}"><div class="kpi">
      <div><div class="kpi-l">{label}</div><div class="kpi-v">{value}</div>{d}</div>
      <div class="tile tone-{tone}">{icon(icon_name, 18, TONE_HEX[tone])}</div>
    </div>{b}</div>
    """)


def feature_card(title, desc, icon_name="sparkles", tone="brand"):
    return _flat(f"""
    <div class="card card-hover">
      <div class="tile tone-{tone}">{icon(icon_name, 18, TONE_HEX[tone])}</div>
      <div class="card-t">{title}</div><div class="card-d">{desc}</div>
    </div>
    """)


def hero(chip, title, text, stats=None):
    st_html = ""
    if stats:
        st_html = '<div class="hero-stats">' + "".join(
            f'<div class="hero-stat"><b>{v}</b><span>{l}</span></div>' for v, l in stats) + "</div>"
    html(f"""
    <div class="hero">
      <div class="hero-chip">{chip}</div>
      <h2>{title}</h2><p>{text}</p>{st_html}
    </div>
    """)


def note(text, kind="info"):
    ic = {"info": "info", "warn": "warning", "good": "check_circle",
          "bad": "x_circle", "brand": "info"}[kind]
    col = {"info": "#0284c7", "warn": "#d97706", "good": "#059669",
           "bad": "#e11d48", "brand": "#2563eb"}[kind]
    html(f'<div class="note {kind}">{icon(ic, 18, col)}<div>{text}</div></div>')


def risk_pill(level):
    t = RISK_TONE.get(level, "info")
    ic = {"Low": "check_circle", "Medium": "warning", "High": "x_circle"}.get(level, "info")
    col = {"low": "#059669", "med": "#d97706", "high": "#e11d48", "info": "#0284c7"}[t]
    return f'<span class="pill {t}">{icon(ic, 14, col)}{level} Risk</span>'


def pill(text, tone="gray"):
    return f'<span class="pill {tone}">{text}</span>'


def kv_row(label, value, icon_name=None):
    ic = icon(icon_name, 16, "#64748b") if icon_name else ""
    return _flat(f'<div class="kv"><span class="kv-k">{ic}{label}</span><span class="kv-v">{value}</span></div>')


def prob_bars(probs):
    rows = ""
    for label, p in probs.items():
        pct = round(p * 100)
        c = RISK_COLOR.get(label, "#2563eb")
        rows += (f'<div class="bar-row"><span class="bar-l">{label}</span>'
                 f'<span class="bar-t"><span class="bar-f" style="display:block;width:{pct}%;background:{c};"></span></span>'
                 f'<span class="bar-v">{pct}%</span></div>')
    return rows


def table(df):
    """Render a pandas DataFrame as a styled HTML table."""
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = ""
    for _, row in df.iterrows():
        body += "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
    return f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
