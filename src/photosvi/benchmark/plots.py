"""Plotly views of a `BenchmarkResult` -- responsive, full-width, hover-rich.

Every figure is built `autosize=True` with no fixed width, so it fills the notebook cell
and resizes with the window; render with `show(fig)` (or `fig.show(config=CONFIG)`) to get
the responsive behaviour plus scroll-zoom and a clean modebar. Colors are the CVD-validated
categorical pair (blue = matched filter / our model, orange = GuPPy / the baseline); a third
detector (fitted-q) gets aqua, further ones fall back to the reserve ramp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# cvd-validated categorical slots (dataviz palette): blue, orange, aqua + reserve
PALETTE = {"matched-filter": "#2a78d6", "guppy": "#eb6834", "fitted-q": "#1baf7a"}
_FALLBACK = ["#4a3aa7", "#e34948", "#eda100"]
_INK = "#52514e"
_GRID = "#c3c2b7"
_REF = "#9a9992"
TEMPLATE = "plotly_white"

# responsive: fills + resizes with the window; scrollZoom + hi-res export, no plotly logo
CONFIG = {
    "responsive": True,
    "scrollZoom": True,
    "displaylogo": False,
    "toImageButtonOptions": {"scale": 2},
}


def _color(name: str, i: int) -> str:
    return PALETTE.get(name, _FALLBACK[i % len(_FALLBACK)])


def show(fig: go.Figure, *, window: bool = False) -> None:
    """Render a figure responsively inline (fills the cell, resizes with the window).

    `window=True` pops it into a resizable browser tab instead -- bigger, and hover works on
    every subplot (inline small-multiples cramp the right panels and drop their hover).
    """
    fig.show(config=CONFIG, renderer="browser" if window else None)


def _responsive(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        autosize=True,
        height=height,
        margin=dict(l=60, r=30, t=64, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    return fig


def metrics_table(result):
    """Formatted metrics DataFrame (a pandas Styler; renders as HTML in a notebook)."""
    rows = {n: o.metrics for n, o in result.outcomes.items()}
    cols = {
        "pr_auc": "PR-AUC", "recall_at": "recall@P.90", "best_f1": "best F1",
        "perbin_roc": "per-bin ROC", "lag": "lag", "n_pred": "n_pred", "n_true": "n_true",
    }
    df = pd.DataFrame(rows).T.reindex(columns=[c for c in cols if c in pd.DataFrame(rows).T.columns])
    df = df.rename(columns=cols)
    fmt = {k: "{:.3f}" for k in ("PR-AUC", "recall@P.90", "best F1", "per-bin ROC") if k in df.columns}
    fmt.update({k: "{:.0f}" for k in ("lag", "n_pred", "n_true") if k in df.columns})
    grad = [c for c in ("PR-AUC", "best F1") if c in df.columns]
    n_true = next(iter(result.outcomes.values())).metrics["n_true"]
    return (
        df.style.format(fmt, na_rep="—")
        .background_gradient(subset=grad, cmap="Blues", vmin=0, vmax=1)
        .set_caption(f"benchmark — tol=±{result.tol / result.fs:.2f}s, n_true={n_true}")
    )


def pr_figure(result) -> go.Figure:
    """Event-matched precision–recall, one line per detector, AP in the legend."""
    fig = go.Figure()
    for i, (n, o) in enumerate(result.outcomes.items()):
        rec, prec = o.curve
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines", name=f"{n} · AP={o.metrics['pr_auc']:.2f}",
            line=dict(color=_color(n, i), width=2),
            hovertemplate="recall %{x:.2f}<br>precision %{y:.2f}<extra>" + n + "</extra>",
        ))
    fig.add_hline(y=0.9, line=dict(color=_REF, dash="dot"),
                  annotation_text="precision 0.90", annotation_position="bottom left")
    fig.update_xaxes(title="recall", range=[0, 1.0])
    fig.update_yaxes(title="precision", range=[0, 1.02])
    fig.update_layout(title="event-matched precision–recall")
    return _responsive(fig, 480)


def trace_figure(result, *, t0_s: float = 120, dur_s: float = 25, show_quantile: float = 0.6) -> go.Figure:
    """Zoomed trace: grey verticals = true onsets, ▽ markers = each detector's confident hits."""
    fs, y = result.fs, result.y
    dt = 1.0 / fs
    i0, i1 = int(t0_s * fs), min(int((t0_s + dur_s) * fs), len(y))
    t = np.arange(i0, i1) * dt
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y[i0:i1], mode="lines", name="y",
                             line=dict(color=_INK, width=1),
                             hovertemplate="t %{x:.1f}s<br>y %{y:.2f}<extra></extra>"))
    for o in result.onsets[(result.onsets >= i0) & (result.onsets < i1)]:
        fig.add_vline(x=o * dt, line=dict(color=_GRID, width=1))
    ymax = float(y[i0:i1].max()) if i1 > i0 else 1.0
    for k, (n, o) in enumerate(result.outcomes.items()):
        d = o.times - o.lag
        if len(o.scores):
            d = d[o.scores > np.quantile(o.scores, show_quantile)]
        d = d[(d >= i0) & (d < i1)]
        fig.add_trace(go.Scatter(x=d * dt, y=np.full(len(d), ymax * (1.12 + 0.1 * k)),
                                 mode="markers", name=n,
                                 marker=dict(color=_color(n, k), size=9, symbol="triangle-down"),
                                 hovertemplate="t %{x:.1f}s<extra>" + n + "</extra>"))
    fig.update_xaxes(title="time (s)")
    fig.update_yaxes(title="y")
    fig.update_layout(title=f"trace {t0_s:g}–{t0_s + dur_s:g}s · grey = onsets, ▽ = detections")
    return _responsive(fig, 360)


def dashboard(result, *, t0_s: float = 120, dur_s: float = 25) -> go.Figure:
    """PR curve + trace panel stacked in one responsive figure."""
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.5, 0.5], vertical_spacing=0.13,
        subplot_titles=("event-matched precision–recall", f"trace {t0_s:g}–{t0_s + dur_s:g}s"),
    )
    for i, (n, o) in enumerate(result.outcomes.items()):
        rec, prec = o.curve
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                 name=f"{n} · AP={o.metrics['pr_auc']:.2f}",
                                 line=dict(color=_color(n, i), width=2)), row=1, col=1)
    fig.add_hline(y=0.9, line=dict(color=_REF, dash="dot"), row=1, col=1)

    fs, y = result.fs, result.y
    dt = 1.0 / fs
    i0, i1 = int(t0_s * fs), min(int((t0_s + dur_s) * fs), len(y))
    t = np.arange(i0, i1) * dt
    fig.add_trace(go.Scatter(x=t, y=y[i0:i1], mode="lines", showlegend=False,
                             line=dict(color=_INK, width=1)), row=2, col=1)
    for o in result.onsets[(result.onsets >= i0) & (result.onsets < i1)]:
        fig.add_vline(x=o * dt, line=dict(color=_GRID, width=1), row=2, col=1)
    ymax = float(y[i0:i1].max()) if i1 > i0 else 1.0
    for k, (n, o) in enumerate(result.outcomes.items()):
        d = o.times - o.lag
        d = d[(d >= i0) & (d < i1)]
        fig.add_trace(go.Scatter(x=d * dt, y=np.full(len(d), ymax * (1.12 + 0.1 * k)),
                                 mode="markers", showlegend=False,
                                 marker=dict(color=_color(n, k), size=9, symbol="triangle-down")),
                      row=2, col=1)
    fig.update_xaxes(title="recall", range=[0, 1.0], row=1, col=1)
    fig.update_yaxes(title="precision", range=[0, 1.02], row=1, col=1)
    fig.update_xaxes(title="time (s)", row=2, col=1)
    fig.update_yaxes(title="y", row=2, col=1)
    return _responsive(fig, 780)
