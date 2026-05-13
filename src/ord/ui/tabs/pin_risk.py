"""Pin Risk tab: pin-likelihood scores + cumulative gamma vs strike."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ord.analytics.pin_risk import pin_risk
from ord.ui.context import AppContext


def render(ctx: AppContext) -> None:
    chain = ctx.filtered_chain
    if chain.empty:
        st.info("No options data available for the current filters.")
        return

    expiries = sorted(chain["expiry"].dropna().unique())
    selected = st.selectbox(
        "Expiry",
        options=expiries,
        format_func=lambda d: (d.date() if hasattr(d, "date") else d).isoformat(),
        key="pin_expiry",
    )
    sel_date = selected.date() if hasattr(selected, "date") else selected
    result = pin_risk(chain, sel_date, r=ctx.rate, q=ctx.dividend_yield)
    if result is None or not result.candidates:
        st.warning("No pin-risk candidates for this expiry (no open interest).")
        return

    rows = [
        {
            "strike": f"${c.strike:,.2f}",
            "score": f"{c.score:.1f}",
            "distance_pct": f"{c.distance_pct * 100:+.2f}%",
            "open_interest": c.oi,
            "gex_total": f"{c.gex_total / 1e6:+.2f} $MM",
        }
        for c in result.candidates
    ]
    st.subheader(f"Top pin candidates -- {sel_date.isoformat()}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    cols = st.columns(3)
    cols[0].metric("Spot", f"${result.spot:,.2f}")
    cols[1].metric(
        "Gamma flip",
        f"${result.gamma_flip_strike:,.2f}" if result.gamma_flip_strike else "n/a",
    )
    cols[2].metric("Top score", f"{result.candidates[0].score:.1f}")
