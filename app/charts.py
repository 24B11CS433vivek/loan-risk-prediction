"""
charts.py - Plotly figures used across the LoanIQ app.
Every function returns a plotly Figure; pages draw it with show(fig).
"""
import plotly.graph_objects as go
import streamlit as st

RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLOR = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#f43f5e"}
BRAND = "#2563eb"
VIOLET = "#7c3aed"
CYAN = "#06b6d4"
GRID = "#edf1fa"
FONT = "Inter, system-ui, sans-serif"
METRIC_COLOR = {"accuracy": "#2563eb", "precision": "#06b6d4", "recall": "#f59e0b", "f1_macro": "#7c3aed"}


def _base(fig, height=320, legend=False):
    fig.update_layout(
        height=height, margin=dict(l=14, r=14, t=24, b=14),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color="#3b4a6b", size=13), showlegend=legend,
        hoverlabel=dict(bgcolor="#0b1437", font=dict(color="#ffffff", family=FONT)),
    )
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ------------------------------------------------------------------ assessment charts
def gauge(score, label):
    color = RISK_COLOR[label]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number=dict(font=dict(size=48, color="#0b1437", family=FONT)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#9aa7c4", tickfont=dict(size=11, color="#6b7a99")),
            bar=dict(color=color, thickness=0.3), bgcolor="#ffffff", borderwidth=0,
            steps=[dict(range=[0, 33], color="#d1fae5"), dict(range=[33, 66], color="#fef3c7"),
                   dict(range=[66, 100], color="#ffe0e8")],
            threshold=dict(line=dict(color="#0b1437", width=3), thickness=0.8, value=score),
        )))
    return _base(fig, 280).update_layout(margin=dict(l=34, r=34, t=24, b=10))


def donut(probs):
    labels = [k for k in RISK_ORDER if k in probs]
    values = [probs[k] for k in labels]
    top = labels[values.index(max(values))]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.68, sort=False, direction="clockwise",
        marker=dict(colors=[RISK_COLOR[k] for k in labels], line=dict(color="#ffffff", width=3)),
        textinfo="percent", textfont=dict(color="#ffffff", size=13), hovertemplate="%{label}: %{percent}<extra></extra>"))
    fig.add_annotation(text=f"<b>{top}</b><br><span style='font-size:12px;color:#6b7a99'>most likely</span>",
                       showarrow=False, font=dict(size=22, color="#0b1437", family=FONT))
    return _base(fig, 280, legend=True).update_layout(legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))


def contributions(contribs, top=8):
    rows = sorted(contribs[:top], key=lambda c: c["impact"])
    fig = go.Figure(go.Bar(
        x=[c["impact"] for c in rows], y=[c["feature"] for c in rows], orientation="h",
        marker=dict(color=["#f43f5e" if c["impact"] > 0 else "#10b981" for c in rows], cornerradius=6),
        text=[f"{c['impact']:+.1f}" for c in rows], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.1f} risk points<extra></extra>"))
    fig.update_xaxes(zeroline=True, zerolinecolor="#9aa7c4", gridcolor=GRID, title="Risk points  (red raises risk, green lowers it)")
    fig.update_yaxes(automargin=True)
    return _base(fig, 60 + 38 * len(rows))


# ------------------------------------------------------------------ dashboard / model charts
def ring(value, color, label=""):
    """One KPI ring: a donut that fills to `value` (0..1)."""
    fig = go.Figure(go.Pie(
        values=[value, max(0.0, 1 - value)], hole=0.78, sort=False, direction="clockwise", rotation=0,
        marker=dict(colors=[color, "#edf1fa"], line=dict(width=0)), textinfo="none", hoverinfo="skip", showlegend=False))
    fig.add_annotation(text=f"<b>{value:.1%}</b>", showarrow=False, font=dict(size=24, color="#0b1437", family=FONT), y=0.56)
    fig.add_annotation(text=label, showarrow=False, font=dict(size=12, color="#6b7a99", family=FONT), y=0.38)
    return _base(fig, 200).update_layout(margin=dict(l=8, r=8, t=8, b=8))


def radar(sel, ref):
    keys = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
            ("f1_macro", "F1 score"), ("roc_auc", "ROC-AUC")]
    labels = [k[1] for k in keys] + [keys[0][1]]
    fig = go.Figure()
    if sel["name"] != ref["name"]:
        r = [ref[k] for k, _ in keys]; r.append(r[0])
        fig.add_trace(go.Scatterpolar(r=r, theta=labels, name=ref["name"] + " (recommended)", fill="toself",
                                      line=dict(color="#94a3b8", width=2, dash="dot"), fillcolor="rgba(148,163,184,.12)"))
    r = [sel[k] for k, _ in keys]; r.append(r[0])
    fig.add_trace(go.Scatterpolar(r=r, theta=labels, name=sel["name"], fill="toself",
                                  line=dict(color=BRAND, width=3), fillcolor="rgba(37,99,235,.22)",
                                  hovertemplate="%{theta}: %{r:.3f}<extra></extra>"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0.6, 1.0], tickvals=[0.7, 0.8, 0.9, 1.0], angle=90, tickangle=90, gridcolor="#dde4f3", tickfont=dict(size=10, color="#9aa7c4"), tickformat=".0%"),
                                 angularaxis=dict(gridcolor="#dde4f3", tickfont=dict(size=12, color="#3b4a6b"))))
    return _base(fig, 360, legend=True).update_layout(legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
                                                      margin=dict(l=40, r=40, t=24, b=30))


def per_class(cm, classes):
    """Precision / recall / F1 for each risk class, derived from the confusion matrix."""
    n = len(classes)
    rows = {}
    for i, c in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        rows[c] = (p, r_, f)
    order = [c for c in RISK_ORDER if c in rows]
    fig = go.Figure()
    for j, (name, col) in enumerate([("Precision", CYAN), ("Recall", "#f59e0b"), ("F1 score", VIOLET)]):
        fig.add_bar(name=name, x=order, y=[rows[c][j] for c in order], marker_color=col, marker_cornerradius=5,
                    text=["{:.0%}".format(rows[c][j]) for c in order], textposition="outside", cliponaxis=False,
                    hovertemplate="%{x} " + name + ": %{y:.3f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_yaxes(range=[0, 1.1], gridcolor=GRID, tickformat=".0%")
    return _base(fig, 360, legend=True).update_layout(legend=dict(orientation="h", y=1.1, x=0))


def confusion(matrix, classes):
    order = [classes.index(c) for c in RISK_ORDER if c in classes]
    matrix = [[matrix[i][j] for j in order] for i in order]
    classes = [classes[i] for i in order]
    z = [list(r) for r in matrix]
    pct = [[(v / sum(r) if sum(r) else 0) for v in r] for r in z]
    text = [["%d<br><span style='font-size:11px'>%.0f%%</span>" % (z[i][j], pct[i][j] * 100) for j in range(len(z))] for i in range(len(z))]
    fig = go.Figure(go.Heatmap(
        z=pct, x=classes, y=classes, colorscale=[[0, "#f0f6ff"], [0.5, "#60a5fa"], [1, "#1d4ed8"]], showscale=False,
        zmin=0, zmax=1, text=text, texttemplate="%{text}", textfont=dict(size=16, color="#0b1437"),
        xgap=4, ygap=4, customdata=z,
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{customdata}<extra></extra>"))
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return _base(fig, 360)


def model_heatmap(rows, chosen):
    """All models x all metrics on one bright heat map."""
    keys = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
            ("f1_macro", "F1 score"), ("roc_auc", "ROC-AUC")]
    names = [(r["name"] + "  \u25CF") if r["name"] == chosen else r["name"] for r in rows]
    z = [[r[k] for k, _ in keys] for r in rows]
    fig = go.Figure(go.Heatmap(
        z=z, x=[k[1] for k in keys], y=names, colorscale=[[0, "#fff1f4"], [0.45, "#e0ecff"], [1, "#2563eb"]],
        zmin=0.75, zmax=0.97, text=[["%.3f" % v for v in row] for row in z], texttemplate="%{text}",
        textfont=dict(size=12, color="#0b1437"), xgap=3, ygap=3, showscale=False,
        hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(side="top")
    return _base(fig, 60 + 36 * len(rows)).update_layout(margin=dict(l=14, r=14, t=40, b=10))


def bubble(rows, chosen):
    """Accuracy vs ROC-AUC, bubble size = F1."""
    fig = go.Figure()
    other = [r for r in rows if r["name"] != chosen]
    sel = [r for r in rows if r["name"] == chosen]
    for grp, col, op in [(other, "#93b4f5", .75), (sel, "#7c3aed", 1.0)]:
        if not grp:
            continue
        fig.add_scatter(x=[r["accuracy"] for r in grp], y=[r["roc_auc"] for r in grp], mode="markers+text",
                        text=[r["name"] if r["name"] == chosen else "" for r in grp], textposition="top center",
                        textfont=dict(size=12, color="#6d28d9"),
                        marker=dict(size=[(r["f1_macro"] - 0.6) * 120 + 8 for r in grp], color=col, opacity=op,
                                    line=dict(color="#ffffff", width=2)),
                        customdata=[[r["name"], r["f1_macro"]] for r in grp],
                        hovertemplate="%{customdata[0]}<br>Accuracy %{x:.3f}<br>ROC-AUC %{y:.3f}<br>F1 %{customdata[1]:.3f}<extra></extra>")
    fig.update_xaxes(title="Accuracy", gridcolor=GRID, tickformat=".0%")
    fig.update_yaxes(title="ROC-AUC", gridcolor=GRID, tickformat=".2f")
    return _base(fig, 420)


def leaderboard(rows, chosen, top=8):
    top_rows = rows[:top][::-1]
    fig = go.Figure(go.Bar(
        x=[m["f1_macro"] for m in top_rows], y=[m["name"] for m in top_rows], orientation="h",
        marker=dict(color=[VIOLET if m["name"] == chosen else "#bcd3fb" for m in top_rows], cornerradius=6),
        text=["{:.3f}".format(m["f1_macro"]) for m in top_rows], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0.5, 1.0], gridcolor=GRID)
    fig.update_yaxes(automargin=True)
    return _base(fig, 320)


def metric_bars(rows):
    names = [r["name"] for r in rows]
    fig = go.Figure()
    for key, label, col in [("accuracy", "Accuracy", "#2563eb"), ("f1_macro", "F1 (macro)", "#7c3aed"),
                            ("roc_auc", "ROC-AUC", "#06b6d4")]:
        fig.add_bar(name=label, x=names, y=[r[key] for r in rows], marker_color=col, marker_cornerradius=4,
                    hovertemplate="%{x}<br>" + label + ": %{y:.3f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(range=[0.6, 1.0], gridcolor=GRID)
    fig.update_xaxes(tickangle=-30)
    return _base(fig, 420, legend=True).update_layout(legend=dict(orientation="h", y=1.1, x=0))


def metric_profile(sel, ref):
    keys = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1_macro", "F1 score")]
    labels = [k[1] for k in keys]
    fig = go.Figure()
    if sel["name"] != ref["name"]:
        fig.add_bar(name=ref["name"] + " (recommended)", x=labels, y=[ref[k] for k, _ in keys],
                    marker_color="#cbd5e1", marker_cornerradius=4)
    fig.add_bar(name=sel["name"], x=labels, y=[sel[k] for k, _ in keys], marker_color=BRAND, marker_cornerradius=4,
                text=["{:.1%}".format(sel[k]) for k, _ in keys], textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group", bargap=0.35)
    fig.update_yaxes(range=[0.5, 1.0], gridcolor=GRID, tickformat=".0%")
    return _base(fig, 340, legend=True).update_layout(legend=dict(orientation="h", y=1.14, x=0))


def cv_vs_test(df_cv, rows):
    """Cross-validation F1 vs test F1 per model: a quick overfitting check."""
    cv = {r["Model"]: r["CV F1 (macro)"] for _, r in df_cv.iterrows()}
    names = [r["name"] for r in rows if r["name"] in cv]
    fig = go.Figure()
    fig.add_bar(name="Cross-validation F1", x=names, y=[cv[n] for n in names], marker_color="#bcd3fb", marker_cornerradius=4)
    fig.add_bar(name="Test F1", x=names, y=[next(r["f1_macro"] for r in rows if r["name"] == n) for n in names],
                marker_color=VIOLET, marker_cornerradius=4)
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(range=[0.6, 1.0], gridcolor=GRID)
    fig.update_xaxes(tickangle=-30)
    return _base(fig, 380, legend=True).update_layout(legend=dict(orientation="h", y=1.12, x=0))


# ------------------------------------------------------------------ insights charts
def roc(roc_data):
    fig = go.Figure()
    for cls in RISK_ORDER:
        if cls in roc_data:
            d = roc_data[cls]
            fig.add_scatter(x=d["fpr"], y=d["tpr"], mode="lines", name=f"{cls} (AUC {d['auc']:.3f})",
                            line=dict(color=RISK_COLOR[cls], width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#9aa7c4", dash="dash"), showlegend=False)
    fig.update_xaxes(title="False positive rate", gridcolor=GRID)
    fig.update_yaxes(title="True positive rate", gridcolor=GRID)
    return _base(fig, 360, legend=True).update_layout(legend=dict(orientation="h", y=-0.25, x=0))


def importance(df, top=10):
    d = df.sort_values("importance", ascending=False).head(top).iloc[::-1]
    names = [n.replace("_", " ").title() for n in d["feature"]]
    fig = go.Figure(go.Bar(
        x=d["importance"], y=names, orientation="h", error_x=dict(array=d["std"], color="#9aa7c4"),
        marker=dict(color=d["importance"], colorscale=[[0, "#bcd3fb"], [0.5, "#2563eb"], [1, "#7c3aed"]], cornerradius=6),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Drop in F1 when the feature is shuffled", gridcolor=GRID)
    return _base(fig, 60 + 36 * len(d))


def class_donut(counts):
    labels = [k for k in RISK_ORDER if k in counts]
    fig = go.Figure(go.Pie(
        labels=labels, values=[counts[k] for k in labels], hole=0.64, sort=False,
        marker=dict(colors=[RISK_COLOR[k] for k in labels], line=dict(color="#ffffff", width=3)),
        textinfo="label+percent", textfont=dict(color="#ffffff", size=13)))
    total = sum(counts.values())
    fig.add_annotation(text=f"<b>{total:,}</b><br><span style='font-size:12px;color:#6b7a99'>records</span>",
                       showarrow=False, font=dict(size=22, color="#0b1437", family=FONT))
    return _base(fig, 320)


def box_by_risk(df, col, title):
    fig = go.Figure()
    for cls in RISK_ORDER:
        fig.add_box(y=df.loc[df["risk_label"] == cls, col], name=cls, marker_color=RISK_COLOR[cls], boxmean=True, line=dict(width=2))
    fig.update_yaxes(title=title, gridcolor=GRID)
    return _base(fig, 340)


def risk_mix(df, col, order=None):
    ct = df.groupby([col, "risk_label"]).size().unstack(fill_value=0)
    ct = ct.div(ct.sum(axis=1), axis=0) * 100
    if order:
        ct = ct.reindex([o for o in order if o in ct.index])
    fig = go.Figure()
    for cls in RISK_ORDER:
        if cls in ct.columns:
            fig.add_bar(name=cls, x=ct.index, y=ct[cls], marker_color=RISK_COLOR[cls],
                        hovertemplate="%{x}<br>" + cls + ": %{y:.0f}%<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Share of applicants (%)", gridcolor=GRID)
    return _base(fig, 340, legend=True).update_layout(legend=dict(orientation="h", y=1.12, x=0))
