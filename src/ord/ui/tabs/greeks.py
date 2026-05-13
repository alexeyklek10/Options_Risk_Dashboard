"""Greeks tab: per-strike heatmaps for the seven Greeks across expiries."""

from __future__ import annotations

import streamlit as st

from ord.analytics.greeks_dashboard import attach_greeks
from ord.ui.charts import greek_heatmap
from ord.ui.context import AppContext


def render(ctx: AppContext) -> None:
    chain = ctx.filtered_chain
    if chain.empty:
        st.info("No options data available for the current filters.")
        return

    st.subheader("Per-strike Greeks")
    with_greeks = attach_greeks(chain, r=ctx.rate, q=ctx.dividend_yield)

    cols = st.columns(2)
    side_left = cols[0].selectbox("Side", ["call", "put"], key="greeks_side_left")
    greek_left = cols[1].selectbox(
        "Greek",
        ["delta", "gamma", "vega", "theta", "vanna", "charm"],
        key="greeks_greek_left",
    )
    st.plotly_chart(greek_heatmap(with_greeks, greek_left, side_left), use_container_width=True)

    with st.expander("Secondary view"):
        cols2 = st.columns(2)
        side_right = cols2[0].selectbox("Side", ["call", "put"], key="greeks_side_right")
        greek_right = cols2[1].selectbox(
            "Greek",
            ["delta", "gamma", "vega", "theta", "vanna", "charm"],
            key="greeks_greek_right",
            index=1,
        )
        st.plotly_chart(
            greek_heatmap(with_greeks, greek_right, side_right), use_container_width=True
        )
