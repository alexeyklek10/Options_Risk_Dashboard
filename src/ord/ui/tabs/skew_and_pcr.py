"""Skew and PCR tab: ATM IV / 25-delta RR + BF term structure, PCR per expiry."""

from __future__ import annotations

import streamlit as st

from ord.analytics.put_call_ratio import put_call_ratio
from ord.analytics.skew import skew
from ord.ui.charts import pcr_bars, skew_term_structure
from ord.ui.context import AppContext


def render(ctx: AppContext) -> None:
    chain = ctx.filtered_chain
    if chain.empty:
        st.info("No options data available for the current filters.")
        return

    sk = skew(chain, r=ctx.rate, q=ctx.dividend_yield)
    pcr = put_call_ratio(chain)

    st.subheader("Skew term structure")
    st.plotly_chart(skew_term_structure(sk.per_expiry), use_container_width=True)

    cols = st.columns([2, 1])
    cols[0].plotly_chart(pcr_bars(pcr.per_expiry), use_container_width=True)

    with cols[1]:
        st.markdown("**Aggregate**")
        st.metric(
            "PCR (volume)",
            f"{pcr.by_volume_aggregate:.2f}" if pcr.by_volume_aggregate is not None else "n/a",
        )
        st.metric(
            "PCR (open interest)",
            f"{pcr.by_oi_aggregate:.2f}" if pcr.by_oi_aggregate is not None else "n/a",
        )

    with st.expander("Per-expiry skew table"):
        st.dataframe(sk.per_expiry, hide_index=True)
