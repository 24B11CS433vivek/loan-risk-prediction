"""
pdf_report.py - builds the downloadable PDF assessment report (reportlab + matplotlib, no browser needed).
No Streamlit imports. Call build_report(...) to get the PDF as bytes.
"""
import io
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

BLUE, VIOLET, CYAN = "#2563eb", "#7c3aed", "#06b6d4"
RISK_HEX = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#f43f5e"}
RISK_ORDER = ["Low", "Medium", "High"]
INK, MUTED, LINE, SOFT = "#0b1437", "#64748b", "#e3e9f6", "#f4f7fd"
PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

DISCLAIMER = ("This report is decision support only. It is an estimate produced by a machine learning model "
              "trained mostly on synthetic data, and it is not a loan approval or rejection. Final lending decisions "
              "need human review and must comply with applicable financial regulations.")


def _style(name, **kw):
    base = dict(fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor(INK), alignment=TA_LEFT)
    base.update(kw)
    return ParagraphStyle(name, **base)


S_BODY = _style("body")
S_SMALL = _style("small", fontSize=8, leading=11, textColor=colors.HexColor(MUTED))
S_H = _style("h", fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=10, spaceAfter=5)
S_CELL = _style("cell", fontSize=9, leading=12)
S_CELL_B = _style("cellb", fontName="Helvetica-Bold", fontSize=9, leading=12)
S_HEAD = _style("head", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.white)
S_BULLET = _style("bullet", leftIndent=12, bulletIndent=0, spaceAfter=3)


def _p(text, style=S_BODY):
    return Paragraph(escape(str(text)), style)


# ------------------------------------------------------------------ charts
def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cbd5e8")
    ax.tick_params(colors="#3b4a6b", labelsize=8)


def probability_chart(probs):
    labels = [k for k in RISK_ORDER if k in probs]
    vals = [probs[k] * 100 for k in labels]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    bars = ax.barh(labels[::-1], vals[::-1], color=[RISK_HEX[k] for k in labels[::-1]], height=0.55)
    for b, v in zip(bars, vals[::-1]):
        ax.text(min(v + 2, 88), b.get_y() + b.get_height() / 2, "%.1f%%" % v, va="center", fontsize=8.5,
                color=INK, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", fontsize=8, color="#3b4a6b")
    _style_axes(ax)
    ax.xaxis.grid(True, color="#e8edf7")
    ax.set_axisbelow(True)
    return _png(fig)


def drivers_chart(contribs, top=7):
    items = [c for c in contribs if abs(float(c["impact"])) >= 0.05][:top] or list(contribs[:top])
    items = items[::-1]
    names = [c["feature"] for c in items]
    vals = [float(c["impact"]) for c in items]
    fig, ax = plt.subplots(figsize=(3.9, 2.5))
    ax.barh(names, vals, color=["#f43f5e" if v > 0 else "#10b981" for v in vals], height=0.55)
    span = max([abs(v) for v in vals] + [1.0])
    for i, v in enumerate(vals):
        ax.text(v + (span * 0.03 if v >= 0 else -span * 0.03), i, "%+.1f" % v, va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_xlabel("Impact on risk score (points)", fontsize=8, color="#3b4a6b")
    _style_axes(ax)
    return _png(fig)


# ------------------------------------------------------------------ page furniture
def _header_footer(generated):
    def draw(c, doc):
        c.saveState()
        c.setFillColor(colors.HexColor(BLUE))
        c.rect(0, PAGE_H - 2.1 * cm, PAGE_W, 2.1 * cm, stroke=0, fill=1)
        c.setFillColor(colors.HexColor(VIOLET))
        c.rect(PAGE_W * 0.72, PAGE_H - 2.1 * cm, PAGE_W * 0.28, 2.1 * cm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.circle(MARGIN + 0.35 * cm, PAGE_H - 1.05 * cm, 0.35 * cm, stroke=0, fill=1)
        c.setFillColor(colors.HexColor(BLUE))
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(MARGIN + 0.35 * cm, PAGE_H - 1.16 * cm, "L")
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 17)
        c.drawString(MARGIN + 1.0 * cm, PAGE_H - 1.03 * cm, "LoanIQ")
        c.setFont("Helvetica", 8.5)
        c.drawString(MARGIN + 1.0 * cm, PAGE_H - 1.55 * cm, "Risk Intelligence")
        c.setFont("Helvetica-Bold", 10)
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.0 * cm, "Applicant Risk Assessment")
        c.setFont("Helvetica", 8.5)
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.5 * cm, generated)
        # footer
        c.setStrokeColor(colors.HexColor(LINE))
        c.line(MARGIN, 1.5 * cm, PAGE_W - MARGIN, 1.5 * cm)
        c.setFillColor(colors.HexColor(MUTED))
        c.setFont("Helvetica", 7.5)
        c.drawString(MARGIN, 1.05 * cm, "LoanIQ  |  Decision support only, not a loan decision")
        c.drawRightString(PAGE_W - MARGIN, 1.05 * cm, "Page %d" % doc.page)
        c.restoreState()
    return draw


def _section(title):
    return Paragraph(escape(title), S_H)


# ------------------------------------------------------------------ blocks
def _verdict(result):
    risk = result["risk_label"]
    col = colors.HexColor(RISK_HEX[risk])
    left = [Paragraph('<font size="8" color="#ffffff">RISK ASSESSMENT</font>', _style("v1", textColor=colors.white)),
            Paragraph('<font size="19" color="#ffffff"><b>%s Risk</b></font>' % escape(risk), _style("v2", leading=24))]
    stats = [("%.0f%%" % (result["confidence"] * 100), "Confidence"), ("%.1f" % result["risk_score"], "Risk score (0-100)"),
             (result["model_used"], "Model")]
    cells = []
    for v, l in stats:
        cells.append([Paragraph('<b>%s</b>' % escape(str(v)), _style("vv", fontSize=11, leading=14)),
                      Paragraph(escape(l), S_SMALL)])
    stat_tbl = Table([[c[0] for c in cells], [c[1] for c in cells]], colWidths=[3.4 * cm, 3.4 * cm, 4.3 * cm])
    stat_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4),
                                  ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    t = Table([[left, stat_tbl]], colWidths=[4.6 * cm, CONTENT_W - 4.6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), col), ("BACKGROUND", (1, 0), (1, 0), colors.HexColor(SOFT)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(LINE))]))
    return [t, Spacer(1, 5), _p(result.get("message", ""), S_BODY)]


def _summary(rows):
    data, half = [], (len(rows) + 1) // 2
    for i in range(half):
        a = rows[i]
        b = rows[i + half] if i + half < len(rows) else ("", "")
        data.append([_p(a[0], S_SMALL), _p(a[1], S_CELL_B), _p(b[0], S_SMALL), _p(b[1], S_CELL_B)])
    t = Table(data, colWidths=[3.3 * cm, CONTENT_W / 2 - 3.3 * cm, 3.3 * cm, CONTENT_W / 2 - 3.3 * cm])
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor(SOFT)]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(LINE)),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def _charts(result):
    w = CONTENT_W / 2 - 0.2 * cm
    p_img = Image(probability_chart(result["probabilities"]), width=w, height=w * 0.69)
    d_img = Image(drivers_chart(result["contributions"]), width=w, height=w * 0.64)
    cap = [Paragraph("<b>Class probabilities</b>", S_CELL), Paragraph("<b>Top risk drivers</b> (vs a typical applicant)", S_CELL)]
    t = Table([cap, [p_img, d_img]], colWidths=[CONTENT_W / 2, CONTENT_W / 2])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    return t


def _metrics_table(models, used):
    head = ["Model", "Accuracy", "Precision", "Recall", "F1 (macro)", "ROC-AUC", "Tuned"]
    rows = [[Paragraph(h, S_HEAD) for h in head]]
    used_row = None
    for i, m in enumerate(models, start=1):
        is_used = m["name"] == used
        if is_used:
            used_row = i
        name = m["name"] + ("  (recommended)" if m.get("selected") else "")
        rows.append([_p(name, S_CELL_B if is_used else S_CELL), "%.3f" % m["accuracy"], "%.3f" % m["precision"],
                     "%.3f" % m["recall"], "%.3f" % m["f1_macro"], "%.3f" % m["roc_auc"], "Yes" if m.get("tuned") else "No"])
    t = Table(rows, colWidths=[5.2 * cm] + [(CONTENT_W - 5.2 * cm) / 6] * 6, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE)), ("FONTSIZE", (1, 1), (-1, -1), 8.5),
          ("FONTNAME", (1, 1), (-1, -1), "Helvetica"), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(SOFT)]),
          ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(LINE)), ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor(LINE)),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if used_row:
        st += [("BACKGROUND", (0, used_row), (-1, used_row), colors.HexColor("#e0e9ff")),
               ("FONTNAME", (1, used_row), (-1, used_row), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    return t


# ------------------------------------------------------------------ public API
def build_report(summary_rows, result, models, generated=None):
    """
    summary_rows : list of (label, value) pairs for the applicant table
    result       : dict returned by predict.predict_risk (risk_label, confidence, risk_score, probabilities,
                   contributions, tips, message, model_used)
    models       : list of dicts from predict.list_models() (test metrics per model)
    Returns the PDF as bytes.
    """
    generated = generated or datetime.now().strftime("%d %b %Y, %H:%M")
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2.9 * cm,
                            bottomMargin=2.0 * cm, title="LoanIQ Applicant Risk Assessment", author="LoanIQ")
    used = next((m for m in models if m["name"] == result["model_used"]), None)
    story = [Paragraph('<font size="16"><b>Applicant Risk Assessment Report</b></font>', _style("t", leading=20)),
             Paragraph("Generated on %s" % escape(generated), S_SMALL), Spacer(1, 8)]
    story += _verdict(result)
    story += [_section("Applicant summary"), _summary(summary_rows)]
    story += [_section("Probabilities and risk drivers"), _charts(result)]
    tips = result.get("tips") or []
    tip_block = [_section("Suggestions")] + [Paragraph(escape(t), S_BULLET, bulletText="-") for t in tips]
    story.append(KeepTogether(tip_block))
    mblock = [_section("Model used and test performance"),
              _p("Scored with %s. Metrics are from the held-out test set (macro averaged); the highlighted row is the model used for this assessment."
                 % result["model_used"], S_SMALL), Spacer(1, 4), _metrics_table(models, result["model_used"])]
    story.append(KeepTogether(mblock))
    disc = Table([[Paragraph("<b>Disclaimer.</b> " + escape(DISCLAIMER), _style("d", fontSize=8.5, leading=12))]], colWidths=[CONTENT_W])
    disc.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff7e6")),
                              ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#f5d28a")),
                              ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                              ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [Spacer(1, 12), disc]
    hf = _header_footer(generated)
    doc.build(story, onFirstPage=hf, onLaterPages=hf)
    return buf.getvalue()
