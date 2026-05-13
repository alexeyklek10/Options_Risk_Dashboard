"""Overview tab: KPI tiles + plain-English summary.

KPIs: spot, nearest-expiry ATM IV, 30D ATM IV, PCR, max-pain (nearest expiry),
total GEX, gamma-flip distance from spot.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from ord.analytics.expected_move import expected_move
from ord.analytics.gex import gamma_exposure
from ord.analytics.max_pain import max_pain_all_expiries
from ord.analytics.put_call_ratio import put_call_ratio
from ord.analytics.skew import skew
from ord.ui.context import AppContext


def render(ctx: AppContext) -> None:
    st.subheader(f"{ctx.ticker} -- snapshot")
    chain = ctx.filtered_chain
    if chain.empty:
        st.info("No options data available for the current filters.")
        return

    pcr = put_call_ratio(chain)
    sk = skew(chain, r=ctx.rate, q=ctx.dividend_yield)
    pain = max_pain_all_expiries(chain)
    gex = gamma_exposure(chain, r=ctx.rate, q=ctx.dividend_yield)
    em = expected_move(chain)

    nearest_expiry = chain["expiry"].min()
    atm_iv_nearest = float("nan")
    if not sk.per_expiry.empty:
        nearest_row = sk.per_expiry.iloc[0]
        atm_iv_nearest = float(nearest_row["atm_iv"])

    # Approximate "30-day ATM IV": pick the expiry whose DTE is closest to 30.
    iv_30d = float("nan")
    if not sk.per_expiry.empty:
        per = sk.per_expiry.copy()
        dtes = chain.groupby("expiry")["dte"].first()
        per["dte"] = per["expiry"].map(dtes)
        per = per.dropna(subset=["dte"])
        if not per.empty:
            iv_30d = float(per.iloc[(per["dte"] - 30).abs().argsort()[:1]]["atm_iv"].iloc[0])

    nearest_pain = pain.get(
        nearest_expiry.date() if hasattr(nearest_expiry, "date") else nearest_expiry
    )

    nearest_em = float("nan")
    if not em.per_expiry.empty:
        nearest_em = float(em.per_expiry.iloc[0]["expected_move"])

    flip_distance: float | None = None
    if gex.gamma_flip_strike is not None and not np.isnan(ctx.spot):
        flip_distance = gex.gamma_flip_strike - ctx.spot

    cols = st.columns(4)
    cols[0].metric("Spot", f"${ctx.spot:,.2f}")
    cols[1].metric("ATM IV (nearest)", _fmt_pct(atm_iv_nearest))
    cols[2].metric("ATM IV (~30D)", _fmt_pct(iv_30d))
    cols[3].metric(
        "Expected move (nearest)",
        f"+/- ${nearest_em:,.2f}" if not np.isnan(nearest_em) else "n/a",
    )

    cols2 = st.columns(4)
    cols2[0].metric("PCR (volume)", _fmt(pcr.by_volume_aggregate))
    cols2[1].metric("PCR (OI)", _fmt(pcr.by_oi_aggregate))
    cols2[2].metric(
        "Max pain (nearest)",
        f"${nearest_pain.strike:,.2f}" if nearest_pain else "n/a",
    )
    cols2[3].metric(
        "Total GEX ($MM / 1%)",
        f"{gex.total_gex / 1e6:+.1f}" if not np.isnan(gex.total_gex) else "n/a",
    )

    summary = _build_summary(
        ctx.ticker,
        atm_iv_nearest,
        iv_30d,
        pcr.by_oi_aggregate,
        nearest_pain.strike if nearest_pain else None,
        ctx.spot,
        gex.total_gex,
        flip_distance,
    )
    st.markdown(summary)


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.2f}"


def _fmt_pct(value: float) -> str:
    if np.isnan(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _build_summary(
    ticker: str,
    atm_iv_near: float,
    atm_iv_30d: float,
    pcr_oi: float | None,
    max_pain: float | None,
    spot: float,
    total_gex: float,
    flip_distance: float | None,
) -> str:
    parts: list[str] = [f"**{ticker}** at ${spot:,.2f}."]
    if not np.isnan(atm_iv_near):
        parts.append(f"Nearest-expiry ATM IV is {atm_iv_near * 100:.1f}%.")
    if not np.isnan(atm_iv_30d) and atm_iv_30d != atm_iv_near:
        direction = "above" if atm_iv_30d > atm_iv_near else "below"
        parts.append(
            f"The ~30-day ATM IV ({atm_iv_30d * 100:.1f}%) sits {direction} the nearest expiry."
        )
    if pcr_oi is not None:
        bias = "bearish" if pcr_oi > 1.0 else "bullish" if pcr_oi < 0.7 else "balanced"
        parts.append(f"Open-interest PCR is {pcr_oi:.2f} -- a {bias} positioning skew.")
    if max_pain is not None and not np.isnan(spot):
        diff_pct = (max_pain - spot) / spot * 100
        parts.append(
            f"Max pain for the nearest expiry sits at ${max_pain:,.2f} "
            f"({diff_pct:+.1f}% from spot)."
        )
    if not np.isnan(total_gex):
        regime = "long-gamma (dampened vol)" if total_gex > 0 else "short-gamma (amplified vol)"
        parts.append(f"Total GEX is {total_gex / 1e6:+.1f} $MM per 1% move -- {regime}.")
    if flip_distance is not None:
        parts.append(f"Gamma-flip sits {flip_distance:+.2f} from spot.")
    return " ".join(parts)
