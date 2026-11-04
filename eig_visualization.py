"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import eig_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


PALETTE = {6: "#4c78a8", 8: "#c0504d", 12: "#2F6B65", 16: "#b08a2e", 20: "#8e6bb0"}


def build_board(inst, p, title: str) -> go.Figure:
    """Plätze als graue Quadrate, Bauteile als Punkte mit Namen auf ihrem Platz; Verbindungen mit Fluss als Linien (Dicke nach dem größten Fluss der Instanz skaliert)."""
    xs_, ys_ = inst.xy[:, 0].astype(float), -inst.xy[:, 1].astype(float)
    pos = [(xs_[int(p[i])], ys_[int(p[i])]) for i in range(inst.n)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_, y=ys_, mode="markers", marker=dict(symbol="square", size=34 if inst.n > 12 else 40, color="rgba(150,150,150,0.18)", line=dict(width=1, color="rgba(150,150,150,0.5)")), hoverinfo="skip", showlegend=False))
    fmax = max(1, int(inst.F.max()))
    level = lambda a, b: min(4, 1 + int(3 * int(inst.F[a, b]) / fmax))
    levels = sorted({level(a, b) for a in range(inst.n) for b in range(a + 1, inst.n) if inst.F[a, b]})
    for lv in levels:
        xs, ys = [], []
        for a in range(inst.n):
            for b in range(a + 1, inst.n):
                if inst.F[a, b] and level(a, b) == lv:
                    xs += [pos[a][0], pos[b][0], None]
                    ys += [pos[a][1], pos[b][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=0.6 + 1.2 * lv, color="rgba(76,120,168,0.35)"), name=("schwacher Fluss", "mittlerer Fluss", "starker Fluss", "stärkster Fluss")[lv - 1], hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[q[0] for q in pos], y=[q[1] for q in pos], mode="markers+text", text=list(inst.names), textposition="middle center", showlegend=False,
                             marker=dict(size=24 if inst.n > 12 else 28, color=C.COLORS["best"]), textfont=dict(color="white", size=10), hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title=title)
    fig.update_xaxes(visible=False, range=[xs_.min() - 0.7, xs_.max() + 0.7])
    fig.update_yaxes(visible=False, range=[ys_.min() - 0.7, ys_.max() + 0.7])
    return _base(fig, height=300, legend_below=True)


# ----------------------------------------------------------------------------- Experiment 1 und 2: eine Instanz

def build_cost_bar(v: dict) -> go.Figure:
    """Die Schranken und der Vergleichswert einer Instanz auf einer Achse (die Eigenwertschranke ist meist negativ)."""
    sp = v["spectral"]
    rows = [("Eigenwertschranke (EB)", sp.eb, C.COLORS["other"]), ("Projizierte Schranke (PB)", sp.pb, C.COLORS["line"]), ("Gilmore–Lawler-Schranke", v["glb"], C.COLORS["warn"]),
            ("Optimum" if v["exact"] else "beste gefundene Lösung", v["ref"], C.COLORS["best"])]
    fig = go.Figure(go.Bar(x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h", marker_color=[r[2] for r in rows], hovertemplate="%{y}: %{x:.1f}<extra></extra>", showlegend=False,
                           text=[f"{r[1]:.1f}".replace(".0", "") for r in rows], textposition="auto"))
    fig.update_layout(title="Schranken und Vergleichswert (die Null ist die triviale Schranke)")
    fig.update_xaxes(title_text="Kosten", zeroline=True)
    fig.update_yaxes(autorange="reversed")
    return _base(fig, height=300, legend_below=False)


def build_spectra(sp) -> go.Figure:
    """Links die gepaarten Eigenwerte (F aufsteigend, D absteigend), rechts die Summanden der Eigenwertschranke."""
    n = len(sp.eig_F)
    idx = list(range(1, n + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Eigenwerte, gepaart", "Summanden der Eigenwertschranke"))
    fig.add_trace(go.Bar(x=idx, y=sp.eig_F, name="Fluss F (aufsteigend)", marker_color=C.COLORS["line"]), row=1, col=1)
    fig.add_trace(go.Bar(x=idx, y=sp.eig_D[::-1], name="Abstand D (absteigend)", marker_color=C.COLORS["warn"]), row=1, col=1)
    fig.add_trace(go.Bar(x=idx, y=sp.products, name="Beitrag", marker_color=[C.COLORS["local"] if p < 0 else C.COLORS["best"] for p in sp.products], showlegend=False,
                         hovertemplate="Paar %{x}: %{y:.2f}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="Rang i", dtick=1 if n <= 12 else None)
    fig.update_yaxes(title_text="Eigenwert", row=1, col=1)
    fig.update_yaxes(title_text="Beitrag zur Schranke", row=1, col=2)
    fig.update_layout(barmode="group")
    return _base(fig, height=360, legend_below=True)


def build_terms(sp) -> go.Figure:
    """Die drei Terme der projizierten Schranke und ihre Summe."""
    labels = ["konstant", "linear", "quadratisch", "PB (Summe)"]
    vals = [sp.const, sp.linear, sp.quad, sp.pb]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[C.COLORS["other"], C.COLORS["other"], C.COLORS["other"], C.COLORS["line"]], hovertemplate="%{x}: %{y:.2f}<extra></extra>", showlegend=False,
                           text=[f"{x:.1f}" for x in vals], textposition="auto"))
    fig.update_layout(title="Die drei Terme der projizierten Schranke")
    fig.update_yaxes(title_text="Beitrag (in Kosten)", zeroline=True)
    return _base(fig, height=340, legend_below=False)


def build_projected_spectra(sp) -> go.Figure:
    """Die Eigenwerte der projizierten Matrizen (F̂ aufsteigend, D̂ absteigend gepaart)."""
    m = len(sp.eig_Fp)
    idx = list(range(1, m + 1))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=idx, y=sp.eig_Fp, name="F̂ (aufsteigend)", marker_color=C.COLORS["line"]))
    fig.add_trace(go.Bar(x=idx, y=sp.eig_Dp[::-1], name="D̂ (absteigend)", marker_color=C.COLORS["warn"]))
    fig.update_layout(title="Eigenwerte der projizierten Matrizen", barmode="group")
    fig.update_xaxes(title_text="Rang i", dtick=1 if m <= 12 else None)
    fig.update_yaxes(title_text="Eigenwert")
    return _base(fig, height=340, legend_below=True)


# ----------------------------------------------------------------------------- Experiment 3 bis 5: Vergleiche

def build_gap_bars(n: int, by_family: dict, median_of) -> go.Figure:
    """Median der Lücke je Familie: projizierte Schranke gegen Gilmore–Lawler (die Eigenwertschranke steht nur in der Tabelle, sie liegt unter 0)."""
    fams = [f for f in C.FAMILIES if by_family.get(f)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[C.FAMILY_NAMES[f] for f in fams], y=[100 * median_of(by_family[f], "gap_glb") for f in fams], name="Gilmore–Lawler", marker_color=C.COLORS["warn"]))
    fig.add_trace(go.Bar(x=[C.FAMILY_NAMES[f] for f in fams], y=[100 * median_of(by_family[f], "gap_pb") for f in fams], name="projizierte Schranke (PB)", marker_color=C.COLORS["line"]))
    fig.update_layout(title=f"Lücke zum Optimum je Familie (Median, n = {n})", barmode="group")
    fig.update_yaxes(title_text="Lücke (%)", rangemode="tozero")
    return _base(fig)


def build_winner(by_n: dict) -> go.Figure:
    """Anteil der Instanzen, in denen die projizierte Schranke größer ist als die Gilmore–Lawler-Schranke, je Familie und Größe."""
    fig = go.Figure()
    for n in sorted(by_n):
        fams = [f for f in C.FAMILIES if by_n[n].get(f)]
        fig.add_trace(go.Bar(x=[C.FAMILY_NAMES[f] for f in fams], y=[100 * sum(1 for r in by_n[n][f] if r["pb_wins"]) / len(by_n[n][f]) for f in fams], name=f"n = {n}", marker_color=PALETTE.get(n, "#888")))
    fig.update_layout(title="Anteil der Instanzen, in denen die projizierte Schranke die bessere ist", barmode="group")
    fig.update_yaxes(title_text="PB größer als Gilmore–Lawler (%)", range=[0, 100])
    return _base(fig)


def build_order(corr_by_n: dict) -> go.Figure:
    """Rangkorrelation der Lücken der Schranken untereinander, je Größe."""
    fig = go.Figure()
    keys = ["EB~GLB", "PB~GLB", "EB~PB"]
    for n in sorted(corr_by_n):
        fig.add_trace(go.Bar(x=keys, y=[corr_by_n[n][k] for k in keys], name=f"n = {n}", marker_color=PALETTE.get(n, "#888"), hovertemplate="%{x}: %{y:+.2f}<extra>n = " + str(n) + "</extra>"))
    fig.update_layout(title="Rangkorrelation der Lücken untereinander", barmode="group")
    fig.update_yaxes(title_text="Spearman", range=[-1, 1], zeroline=True)
    return _base(fig)


def build_large(by_n: dict, median_of) -> go.Figure:
    """Median der Lücke gegen die beste gefundene Lösung: links projizierte Schranke, rechts Gilmore–Lawler, je Familie und Größe."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("projizierte Schranke (PB)", "Gilmore–Lawler"), shared_yaxes=True)
    for col, key in ((1, "gap_pb"), (2, "gap_glb")):
        for n in sorted(by_n):
            fams = [f for f in C.FAMILIES if by_n[n].get(f)]
            fig.add_trace(go.Bar(x=[C.FAMILY_NAMES[f] for f in fams], y=[100 * median_of(by_n[n][f], key) for f in fams], name=f"n = {n}", marker_color=PALETTE.get(n, "#888"), showlegend=col == 1), row=1, col=col)
    fig.update_layout(barmode="group", title="Lücke gegen die beste gefundene Lösung (Median, obere Grenze)")
    fig.update_yaxes(title_text="Lücke (höchstens, %)", rangemode="tozero", row=1, col=1)
    return _base(fig, legend_below=True)
