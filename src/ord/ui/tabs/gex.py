"""GEX tab: per-strike GEX bars with gamma-flip callout, cumulative GEX line."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ord.analytics.gex import gamma_exposure
from ord.ui.charts import gex_bars
from ord.ui.context import AppContext
from ord.ui.theme import GRID, NEUTRAL, PRIMARY, apply_layout


def render(ctx: AppContext) -> None:
    chain = ctx.filtered_chain
    if chain.empty:
        st.info("No options data available for the current filters.")
        return

    result = gamma_exposure(chain, r=ctx.rate, q=ctx.dividend_yield)
    if result.per_strike.empty:
        st.warning(
            "No liquid (positive IV + nonzero OI) contracts in the filtered chain to "
            "compute gamma exposure."
        )
        return

    st.plotly_chart(gex_bars(result), use_container_width=True)

    st.subheader("Cumulative GEX (walking down from highest strike)")
    df = result.per_strike.sort_values("strike", ascending=False).copy()
    df["cum_gex"] = df["gex_total"].cumsum()
    fig = go.Figure(
        data=go.Scatter(
            x=df["strike"],
            y=df["cum_gex"] / 1e6,
            mode="lines+markers",
            line={"color": PRIMARY, "width": 2},
        )
    )
    fig.add_hline(y=0.0, line_color=NEUTRAL, line_dash="dot")
    fig.add_vline(
        x=result.underlying_price, line_color=NEUTRAL, line_dash="dash", annotation_text="Spot"
    )
    if result.gamma_flip_strike is not None:
        fig.add_vline(
            x=result.gamma_flip_strike,
            line_color=PRIMARY,
            line_dash="dot",
            annotation_text=f"Gamma flip = {result.gamma_flip_strike:.2f}",
        )
    fig.update_layout(
        xaxis_title="Strike ($)",
        yaxis_title="Cumulative GEX ($MM per 1%)",
        height=380,
        xaxis={"gridcolor": GRID},
        yaxis={"gridcolor": GRID},
    )
    st.plotly_chart(apply_layout(fig), use_container_width=True)

    cols = st.columns(3)
    cols[0].metric("Total GEX ($MM / 1%)", f"{result.total_gex / 1e6:+.1f}")
    cols[1].metric(
        "Gamma flip",
        f"${result.gamma_flip_strike:,.2f}" if result.gamma_flip_strike else "n/a",
    )
    cols[2].metric(
        "Spot vs gamma flip",
        (
            f"{result.underlying_price - result.gamma_flip_strike:+.2f}"
            if result.gamma_flip_strike is not None
            else "n/a"
        ),
    )
    _ = np  # keep numpy reachable for downstream tabs that import this layout
