"""Strategy Builder tab: up to four legs, expiry P&L, P&L surface, position Greeks."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from ord.analytics.strategy_builder import Leg, evaluate
from ord.pricing.black_scholes import bs_price
from ord.ui.context import AppContext
from ord.ui.theme import GRID, PRIMARY, apply_layout

PRESETS: dict[str, str] = {
    "Long call": "Long ATM call.",
    "Long put": "Long ATM put.",
    "Vertical (call spread)": "Long ATM call, short OTM call.",
    "Iron condor": "Short OTM put, long further-OTM put, short OTM call, long further-OTM call.",
}


def _default_legs(
    preset: str,
    spot: float,
    T: float,  # noqa: N803 - quant notation
    sigma: float,
    r: float,
    q: float,
) -> list[Leg]:
    if preset == "Long call":
        c = bs_price(spot, spot, T, r, sigma, "call", q)
        return [Leg("call", spot, T, sigma, 1.0, c)]
    if preset == "Long put":
        p = bs_price(spot, spot, T, r, sigma, "put", q)
        return [Leg("put", spot, T, sigma, 1.0, p)]
    if preset == "Vertical (call spread)":
        k1, k2 = spot, spot * 1.05
        c1 = bs_price(spot, k1, T, r, sigma, "call", q)
        c2 = bs_price(spot, k2, T, r, sigma, "call", q)
        return [
            Leg("call", k1, T, sigma, 1.0, c1),
            Leg("call", k2, T, sigma, -1.0, c2),
        ]
    # iron condor
    k_pl, k_ps, k_cs, k_cl = spot * 0.90, spot * 0.95, spot * 1.05, spot * 1.10
    p_l = bs_price(spot, k_pl, T, r, sigma, "put", q)
    p_s = bs_price(spot, k_ps, T, r, sigma, "put", q)
    c_s = bs_price(spot, k_cs, T, r, sigma, "call", q)
    c_l = bs_price(spot, k_cl, T, r, sigma, "call", q)
    return [
        Leg("put", k_pl, T, sigma, 1.0, p_l),
        Leg("put", k_ps, T, sigma, -1.0, p_s),
        Leg("call", k_cs, T, sigma, -1.0, c_s),
        Leg("call", k_cl, T, sigma, 1.0, c_l),
    ]


def render(ctx: AppContext) -> None:
    if ctx.chain.empty:
        st.info("Fetch a chain first to construct a strategy.")
        return

    spot = ctx.spot
    cols = st.columns([1, 1, 1, 1])
    preset = cols[0].selectbox("Preset", list(PRESETS.keys()), index=0)
    expiry_days = cols[1].number_input(
        "Days to expiry", min_value=1, max_value=730, value=30, step=1
    )
    sigma = cols[2].number_input(
        "Sigma (per leg)", min_value=0.05, max_value=2.0, value=0.25, step=0.01
    )
    window_pct = cols[3].slider("Spot window (+/- %)", min_value=5, max_value=60, value=20, step=5)
    st.caption(PRESETS[preset])

    legs = _default_legs(preset, spot, expiry_days / 365.0, sigma, ctx.rate, ctx.dividend_yield)
    result = evaluate(
        legs,
        spot=spot,
        r=ctx.rate,
        q=ctx.dividend_yield,
        spot_window_pct=window_pct / 100.0,
        n_spots=121,
        n_times=12,
    )

    fig = go.Figure()
    fig.add_scatter(
        x=result.spot_grid,
        y=result.pnl_at_expiry,
        mode="lines",
        name="P&L at expiry (per share)",
        line={"color": PRIMARY, "width": 2},
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="#9CA3AF")
    fig.add_vline(x=spot, line_dash="dash", line_color="#9CA3AF", annotation_text="Spot")
    for b in result.breakevens:
        fig.add_vline(x=b, line_dash="dot", line_color="#3FB950", annotation_text=f"BE {b:.2f}")
    fig.update_layout(
        xaxis_title="Spot at expiry ($)",
        yaxis_title="P&L per share ($)",
        height=400,
        title=f"{preset} -- P&L at expiry",
        xaxis={"gridcolor": GRID},
        yaxis={"gridcolor": GRID},
    )
    st.plotly_chart(apply_layout(fig), use_container_width=True)

    surf = go.Figure(
        data=go.Surface(
            z=result.pnl_surface,
            x=result.spot_grid,
            y=result.time_grid * 365.0,
            colorscale="RdBu",
            cmid=0.0,
            colorbar={"title": "P&L"},
        )
    )
    surf.update_layout(
        scene={
            "xaxis": {"title": "Spot ($)", "gridcolor": GRID},
            "yaxis": {"title": "Days remaining", "gridcolor": GRID},
            "zaxis": {"title": "P&L per share ($)", "gridcolor": GRID},
        },
        title="P&L surface (spot x time)",
        height=500,
    )
    st.plotly_chart(apply_layout(surf), use_container_width=True)

    cols2 = st.columns(4)
    cols2[0].metric("Max profit ($/share)", f"{result.max_profit:+.2f}")
    cols2[1].metric("Max loss ($/share)", f"{result.max_loss:+.2f}")
    cols2[2].metric("Breakevens", ", ".join(f"{b:.2f}" for b in result.breakevens) or "n/a")
    cols2[3].metric("Position delta", f"{result.greeks['delta']:+.2f}")

    with st.expander("Position Greeks (per share)"):
        st.json({k: round(v, 4) for k, v in result.greeks.items()})
