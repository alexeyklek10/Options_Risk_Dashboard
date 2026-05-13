"""Plotly chart factories shared across tabs.

Each function takes a small set of pandas / numpy / dataclass inputs and
returns a ``plotly.graph_objects.Figure`` already styled by the project's
dark theme (:mod:`ord.ui.theme`). Axis labels always include units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ord.analytics.gex import GEXResult
from ord.analytics.iv_surface import IVSurface
from ord.analytics.max_pain import MaxPainResult
from ord.ui.theme import (
    CALL_COLOR,
    GRID,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    PRIMARY,
    PUT_COLOR,
    apply_layout,
)


def iv_surface_3d(surface: IVSurface, title: str = "Implied volatility surface") -> go.Figure:
    fig = go.Figure(
        data=[
            go.Surface(
                z=surface.iv * 100.0,  # express as percent
                x=surface.strike_grid,
                y=surface.dte_grid,
                colorscale="Viridis",
                colorbar={"title": "IV (%)"},
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene={
            "xaxis": {"title": "Strike ($)", "gridcolor": GRID},
            "yaxis": {"title": "Days to expiry", "gridcolor": GRID},
            "zaxis": {"title": "IV (%)", "gridcolor": GRID},
        },
        height=600,
    )
    return apply_layout(fig)


def iv_smile_2d(observed: pd.DataFrame, expiry_label: str) -> go.Figure:
    calls = observed[observed["option_type"] == "call"].sort_values("strike")
    puts = observed[observed["option_type"] == "put"].sort_values("strike")
    fig = go.Figure()
    if not calls.empty:
        fig.add_scatter(
            x=calls["strike"],
            y=calls["implied_vol"] * 100,
            mode="lines+markers",
            name="Calls",
            line={"color": CALL_COLOR, "width": 2},
        )
    if not puts.empty:
        fig.add_scatter(
            x=puts["strike"],
            y=puts["implied_vol"] * 100,
            mode="lines+markers",
            name="Puts",
            line={"color": PUT_COLOR, "width": 2},
        )
    fig.update_layout(
        title=f"IV smile, {expiry_label}",
        xaxis_title="Strike ($)",
        yaxis_title="IV (annualized, %)",
        height=400,
    )
    return apply_layout(fig)


def greek_heatmap(chain_with_greeks: pd.DataFrame, greek: str, side: str = "call") -> go.Figure:
    df = chain_with_greeks[chain_with_greeks["option_type"] == side]
    if df.empty:
        fig = go.Figure()
        return apply_layout(fig)
    pivot = df.pivot_table(index="strike", columns="expiry", values=greek, aggfunc="mean")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(dtype=np.float64),
            x=[str(c) for c in pivot.columns],
            y=pivot.index,
            colorscale="RdBu",
            colorbar={"title": greek.capitalize()},
            zmid=0.0,
        )
    )
    fig.update_layout(
        title=f"{greek.capitalize()} -- {side}s",
        xaxis_title="Expiry",
        yaxis_title="Strike ($)",
        height=500,
    )
    return apply_layout(fig)


def gex_bars(gex_result: GEXResult) -> go.Figure:
    df = gex_result.per_strike
    if df.empty:
        fig = go.Figure()
        return apply_layout(fig)
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in df["gex_total"]]
    fig = go.Figure(
        data=go.Bar(
            x=df["strike"],
            y=df["gex_total"] / 1e6,  # $MM
            marker_color=colors,
            name="Net GEX",
        )
    )
    fig.add_vline(
        x=gex_result.underlying_price,
        line_dash="dash",
        line_color=NEUTRAL,
        annotation_text="Spot",
        annotation_position="top",
    )
    if gex_result.gamma_flip_strike is not None:
        fig.add_vline(
            x=gex_result.gamma_flip_strike,
            line_dash="dot",
            line_color=PRIMARY,
            annotation_text=f"Gamma flip = {gex_result.gamma_flip_strike:.2f}",
            annotation_position="top",
        )
    fig.update_layout(
        title=f"Gamma exposure (total {gex_result.total_gex / 1e6:+.1f} $MM per 1% move)",
        xaxis_title="Strike ($)",
        yaxis_title="GEX ($MM per 1% move)",
        height=420,
    )
    return apply_layout(fig)


def max_pain_bars(result: MaxPainResult, spot: float) -> go.Figure:
    fig = go.Figure(
        data=go.Bar(
            x=result.pain_curve["strike"],
            y=result.pain_curve["pain"] / 1e6,
            marker_color=PRIMARY,
        )
    )
    fig.add_vline(
        x=spot,
        line_dash="dash",
        line_color=NEUTRAL,
        annotation_text="Spot",
    )
    fig.add_vline(
        x=result.strike,
        line_dash="dot",
        line_color=POSITIVE,
        annotation_text=f"Max pain = {result.strike:.2f}",
    )
    fig.update_layout(
        title=f"Max pain -- {result.expiry.isoformat()}",
        xaxis_title="Strike ($)",
        yaxis_title="Total holder pain ($MM)",
        height=400,
    )
    return apply_layout(fig)


def oi_heatmap_chart(pivot: pd.DataFrame, side: str, spot: float) -> go.Figure:
    if pivot.empty:
        fig = go.Figure()
        return apply_layout(fig)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(dtype=np.float64),
            x=[str(c) for c in pivot.columns],
            y=pivot.index,
            colorscale="Blues" if side == "call" else "Reds",
            colorbar={"title": "Open interest"},
        )
    )
    fig.add_hline(
        y=spot,
        line_dash="dash",
        line_color=NEUTRAL,
        annotation_text="Spot",
        annotation_position="right",
    )
    fig.update_layout(
        title=f"Open interest -- {side}s",
        xaxis_title="Expiry",
        yaxis_title="Strike ($)",
        height=500,
    )
    return apply_layout(fig)


def skew_term_structure(per_expiry: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if per_expiry.empty:
        return apply_layout(fig)
    fig.add_scatter(
        x=per_expiry["expiry"].astype(str),
        y=per_expiry["atm_iv"] * 100,
        name="ATM IV (%)",
        line={"color": PRIMARY},
        mode="lines+markers",
    )
    fig.add_scatter(
        x=per_expiry["expiry"].astype(str),
        y=per_expiry["rr_25d"] * 100,
        name="25-delta RR (%)",
        line={"color": NEGATIVE, "dash": "dot"},
        mode="lines+markers",
        yaxis="y2",
    )
    fig.update_layout(
        title="Skew term structure",
        xaxis_title="Expiry",
        yaxis_title="ATM IV (annualized, %)",
        yaxis2={
            "title": "25-delta RR (%)",
            "overlaying": "y",
            "side": "right",
            "gridcolor": GRID,
        },
        height=400,
    )
    return apply_layout(fig)


def pcr_bars(per_expiry: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if per_expiry.empty:
        return apply_layout(fig)
    fig.add_bar(
        x=per_expiry["expiry"].astype(str),
        y=per_expiry["pcr_volume"].fillna(0),
        name="PCR (volume)",
        marker_color=PRIMARY,
    )
    fig.add_bar(
        x=per_expiry["expiry"].astype(str),
        y=per_expiry["pcr_oi"].fillna(0),
        name="PCR (open interest)",
        marker_color=NEUTRAL,
    )
    fig.update_layout(
        title="Put-call ratio per expiry",
        xaxis_title="Expiry",
        yaxis_title="PCR",
        barmode="group",
        height=400,
    )
    return apply_layout(fig)
