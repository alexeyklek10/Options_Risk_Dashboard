"""Max Pain and OI tab: per-expiry max-pain bars and per-side OI heatmap."""

from __future__ import annotations

import streamlit as st

from ord.analytics.max_pain import max_pain_all_expiries
from ord.analytics.oi_heatmap import oi_heatmap
from ord.ui.charts import max_pain_bars, oi_heatmap_chart
from ord.ui.context import AppContext


def render(ctx: AppContext) -> None:
    chain = ctx.filtered_chain
    if chain.empty:
        st.info("No options data available for the current filters.")
        return

    st.subheader("Max pain")
    pain = max_pain_all_expiries(chain)
    if not pain:
        st.warning("No open interest in the filtered chain to compute max pain.")
    else:
        expiries = sorted(pain.keys())
        selected = st.selectbox(
            "Expiry",
            options=expiries,
            format_func=lambda d: d.isoformat(),
            key="maxpain_expiry",
        )
        st.plotly_chart(max_pain_bars(pain[selected], ctx.spot), use_container_width=True)

    st.subheader("Open-interest heatmap")
    heatmap = oi_heatmap(chain)
    cols = st.columns(2)
    cols[0].plotly_chart(
        oi_heatmap_chart(heatmap.calls, "call", ctx.spot), use_container_width=True
    )
    cols[1].plotly_chart(oi_heatmap_chart(heatmap.puts, "put", ctx.spot), use_container_width=True)
