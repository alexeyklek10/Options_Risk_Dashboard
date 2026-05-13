"""IV vs RV tab: ATM IV minus 21-day realized vol, with optional yfinance history."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ord.analytics.iv_rv_spread import iv_rv_spread
from ord.ui.context import AppContext
from ord.ui.theme import GRID, NEGATIVE, NEUTRAL, POSITIVE, PRIMARY, apply_layout


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_history(ticker: str, period: str = "3mo") -> pd.Series:
    import yfinance as yf

    history = yf.Ticker(ticker).history(period=period, interval="1d")
    if history.empty:
        return pd.Series(dtype="float64")
    return history["Close"].astype("float64")


def render(ctx: AppContext) -> None:
    chain = ctx.filtered_chain
    if chain.empty:
        st.info("No options data available for the current filters.")
        return

    history = _fetch_history(ctx.ticker)
    if history.empty:
        st.warning(f"Could not fetch price history for {ctx.ticker}.")
        return

    result = iv_rv_spread(chain, history, r=ctx.rate, q=ctx.dividend_yield)
    cols = st.columns(3)
    cols[0].metric(
        "21D realized vol",
        f"{result.realized_vol_21d * 100:.1f}%" if result.realized_vol_21d else "n/a",
    )
    cols[1].metric(
        "ATM IV (>= 21 DTE)",
        f"{result.atm_iv_21d * 100:.1f}%" if result.atm_iv_21d else "n/a",
    )
    cols[2].metric(
        "Spread (IV - RV)",
        f"{result.spread * 100:+.1f}%" if result.spread is not None else "n/a",
    )
    if result.expiry_used:
        st.caption(f"Implied vol taken from the {result.expiry_used.isoformat()} expiry.")

    fig = go.Figure()
    fig.add_scatter(
        x=history.index,
        y=history.values,
        mode="lines",
        name=f"{ctx.ticker} close",
        line={"color": PRIMARY, "width": 2},
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Close ($)",
        height=320,
        xaxis={"gridcolor": GRID},
        yaxis={"gridcolor": GRID},
        title=f"{ctx.ticker} -- recent close",
    )
    st.plotly_chart(apply_layout(fig), use_container_width=True)

    if result.spread is not None:
        regime = (
            "IV is rich vs realized -- premium-sellers favored."
            if result.spread > 0.0
            else "IV is cheap vs realized -- premium-buyers favored."
        )
        color = POSITIVE if result.spread > 0.0 else NEGATIVE
        st.markdown(
            f"<span style='color:{color}'><strong>{regime}</strong></span>",
            unsafe_allow_html=True,
        )
    _ = NEUTRAL  # keep theme symbol reachable
