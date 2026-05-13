"""Entry point for the Options Risk Dashboard Streamlit app.

Streamlit Community Cloud runs this module directly. The full UI is assembled
from ``ord.ui.tabs.*`` and ``ord.ui.charts``; this file owns the sidebar and
the chain fetch (which is ``st.cache_data``-d for 15 min).

Default zero-secrets path: yfinance only. Tradier and Polygon providers
auto-enable when ``TRADIER_TOKEN`` / ``POLYGON_API_KEY`` env vars are set
(wiring lands in step 7); ``FRED_API_KEY`` is checked inside
:mod:`ord.utils.rates`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Streamlit Community Cloud does not install the local package by default
# (no `setup.py` step in its build), so we make `from ord.*` resolve by
# inserting the src layout onto sys.path before any project imports. Harmless
# locally (the editable install already covers it) and inside the Docker image
# (where PYTHONPATH=/app/src is set explicitly).
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ord.data.aggregator import ChainAggregator  # noqa: E402
from ord.data.base import DataProvider, ProviderUnavailableError  # noqa: E402
from ord.data.polygon_provider import PolygonProvider  # noqa: E402
from ord.data.tradier_provider import TradierProvider  # noqa: E402
from ord.data.yfinance_provider import YFinanceProvider  # noqa: E402
from ord.ui.context import AppContext  # noqa: E402
from ord.ui.tabs import (  # noqa: E402
    data_quality,
    implied_pdf,
    iv_vs_rv,
    max_pain_oi,
    pin_risk,
    skew_and_pcr,
    strategy_builder,
)
from ord.ui.tabs import gex as gex_tab  # noqa: E402
from ord.ui.tabs import greeks as greeks_tab  # noqa: E402
from ord.ui.tabs import iv_surface as iv_tab  # noqa: E402
from ord.ui.tabs import overview as overview_tab  # noqa: E402
from ord.utils.rates import get_risk_free_rate  # noqa: E402

logging.basicConfig(level=logging.INFO)


def _enabled_providers() -> list[DataProvider]:
    """Build the active provider list.

    yfinance is always available. Tradier and Polygon auto-enable when their
    respective env vars (``TRADIER_TOKEN``, ``POLYGON_API_KEY``) are set;
    otherwise they raise :class:`ProviderUnavailableError` on construction
    and are silently skipped.
    """
    providers: list[DataProvider] = [YFinanceProvider()]
    for cls in (TradierProvider, PolygonProvider):
        try:
            providers.append(cls())
        except ProviderUnavailableError:
            continue
    return providers


@st.cache_data(ttl=900, show_spinner="Fetching options chain...")
def _fetch_chain(ticker: str, max_expiries: int) -> dict[str, pd.DataFrame]:
    aggregator = ChainAggregator(_enabled_providers())
    bundle = aggregator.fetch(ticker, max_expiries=max_expiries)
    return {"consensus": bundle.consensus, **bundle.chains}


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_risk_free_rate() -> float:
    return get_risk_free_rate()


def _render_sidebar() -> tuple[str, int, int, int, float, float, float, float]:
    st.sidebar.markdown("## Inputs")
    ticker = st.sidebar.text_input("Ticker", value="SPY").upper().strip()

    st.sidebar.markdown("### Expiry window (DTE)")
    min_dte, max_dte = st.sidebar.slider(
        "Days to expiry",
        min_value=0,
        max_value=730,
        value=(0, 180),
        step=7,
    )

    st.sidebar.markdown("### Strike window (% of spot)")
    strike_lo, strike_hi = st.sidebar.slider(
        "Strike range",
        min_value=0.5,
        max_value=1.5,
        value=(0.80, 1.20),
        step=0.01,
        format="%.2f",
    )

    st.sidebar.markdown("### Rate and dividend")
    rate_default = _cached_risk_free_rate()
    rate = st.sidebar.number_input(
        "Risk-free rate",
        min_value=0.0,
        max_value=0.25,
        value=float(rate_default),
        step=0.0025,
        format="%.4f",
    )
    div_yield = st.sidebar.number_input(
        "Dividend yield (q)", min_value=0.0, max_value=0.15, value=0.0, step=0.001, format="%.4f"
    )

    st.sidebar.markdown("---")
    max_expiries = st.sidebar.number_input(
        "Max expirations to fetch", min_value=1, max_value=40, value=10, step=1
    )

    refreshed = st.sidebar.button("Refresh data", use_container_width=True)
    if refreshed:
        _fetch_chain.clear()
        _cached_risk_free_rate.clear()
        st.toast("Cache cleared; next fetch will hit the providers.")

    return (
        ticker,
        int(min_dte),
        int(max_dte),
        int(max_expiries),
        float(strike_lo),
        float(strike_hi),
        float(rate),
        float(div_yield),
    )


def main() -> None:
    st.set_page_config(
        page_title="Options Risk Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Options Risk Dashboard")
    st.caption("Research and educational tool. Not investment advice.")

    (
        ticker,
        min_dte,
        max_dte,
        max_expiries,
        strike_lo,
        strike_hi,
        rate,
        div_yield,
    ) = _render_sidebar()

    if not ticker:
        st.info("Enter a US-listed ticker in the sidebar to begin.")
        return

    try:
        chains = _fetch_chain(ticker, max_expiries=max_expiries)
    except Exception as exc:
        st.error(f"Failed to fetch options chain for {ticker}: {exc}")
        return

    consensus = chains.pop("consensus", pd.DataFrame())
    if consensus.empty:
        st.warning(
            f"No options data returned for {ticker}. The ticker may have no listed "
            "options, or the provider returned an empty response."
        )
        return

    ctx = AppContext(
        ticker=ticker,
        chain=consensus,
        chains_by_provider=chains,
        rate=rate,
        dividend_yield=div_yield,
        min_dte=min_dte,
        max_dte=max_dte,
        strike_lo_pct=strike_lo,
        strike_hi_pct=strike_hi,
    )

    (
        tab_overview,
        tab_iv,
        tab_greeks,
        tab_skew,
        tab_max_pain,
        tab_gex,
        tab_iv_rv,
        tab_pdf,
        tab_pin,
        tab_strategy,
        tab_dq,
    ) = st.tabs(
        [
            "Overview",
            "IV Surface",
            "Greeks",
            "Skew and PCR",
            "Max Pain / OI",
            "GEX",
            "IV vs RV",
            "Implied PDF",
            "Pin Risk",
            "Strategy Builder",
            "Data Quality",
        ]
    )
    with tab_overview:
        overview_tab.render(ctx)
    with tab_iv:
        iv_tab.render(ctx)
    with tab_greeks:
        greeks_tab.render(ctx)
    with tab_skew:
        skew_and_pcr.render(ctx)
    with tab_max_pain:
        max_pain_oi.render(ctx)
    with tab_gex:
        gex_tab.render(ctx)
    with tab_iv_rv:
        iv_vs_rv.render(ctx)
    with tab_pdf:
        implied_pdf.render(ctx)
    with tab_pin:
        pin_risk.render(ctx)
    with tab_strategy:
        strategy_builder.render(ctx)
    with tab_dq:
        data_quality.render(ctx)


if __name__ == "__main__":
    main()
