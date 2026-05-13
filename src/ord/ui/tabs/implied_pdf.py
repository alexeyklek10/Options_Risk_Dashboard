"""Implied PDF tab: Breeden-Litzenberger risk-neutral density per expiry."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from ord.analytics.implied_pdf import implied_pdf_all_expiries
from ord.ui.context import AppContext
from ord.ui.theme import GRID, NEUTRAL, PRIMARY, apply_layout


def render(ctx: AppContext) -> None:
    chain = ctx.filtered_chain
    if chain.empty:
        st.info("No options data available for the current filters.")
        return

    smoothing = st.slider(
        "Spline smoothing (s)",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        help="0 = exact interpolation. Higher values smooth out noisy quote curves.",
    )
    pdfs = implied_pdf_all_expiries(chain, r=ctx.rate, smoothing=smoothing)
    if not pdfs:
        st.warning(
            "No expiry has 4+ liquid call quotes in the filtered chain. "
            "Widen the strike window or pick a more liquid ticker."
        )
        return

    expiries = sorted(pdfs.keys())
    selected = st.selectbox(
        "Expiry",
        options=expiries,
        format_func=lambda d: d.isoformat(),
        key="pdf_expiry",
    )
    result = pdfs[selected]

    fig = go.Figure()
    fig.add_scatter(
        x=result.strikes,
        y=result.density,
        mode="lines",
        name="Risk-neutral density",
        line={"color": PRIMARY, "width": 2},
    )
    fig.add_vline(
        x=ctx.spot,
        line_dash="dash",
        line_color=NEUTRAL,
        annotation_text="Spot",
    )
    fig.update_layout(
        xaxis_title="Underlying at expiry ($)",
        yaxis_title="Density",
        height=420,
        title=f"Breeden-Litzenberger PDF -- {selected.isoformat()}",
        xaxis={"gridcolor": GRID},
        yaxis={"gridcolor": GRID},
    )
    st.plotly_chart(apply_layout(fig), use_container_width=True)

    if result.has_negative_density:
        st.warning(
            "The fitted density was non-positive at some points (the negative values "
            "have been clipped). This indicates calendar / butterfly arbitrage in the "
            "input call price curve. Increase the spline smoothing or pick a more "
            "liquid expiry."
        )
