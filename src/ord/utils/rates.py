"""Risk-free rate fetcher with graceful, zero-secrets fallbacks.

Resolution order:

1. **yfinance ``^IRX``** (13-week T-bill yield, primary). yfinance quotes
   ``^IRX`` as a percent; this module divides by 100. No API key required.
2. **FRED ``DGS3MO``** (optional, only when ``FRED_API_KEY`` is set).
3. **Hardcoded ``0.04``** (fallback, with a one-time logged warning).

Callers can override via the ``rate`` sidebar in the Streamlit app. The result
is cached at the call site (``st.cache_data(ttl=86400)``) for 24 hours.
"""

from __future__ import annotations

import logging
import os
from typing import Final

import requests

_LOG = logging.getLogger(__name__)
_DEFAULT_RATE: Final[float] = 0.04
_FRED_BASE: Final[str] = "https://api.stlouisfed.org/fred/series/observations"
_warned_fallback = False


def _try_yfinance_irx() -> float | None:
    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover - yfinance is in requirements.txt
        return None
    try:
        ticker = yf.Ticker("^IRX")
        # Use the most recent close. yfinance returns the percent quote.
        history = ticker.history(period="5d", interval="1d")
        if history.empty:
            return None
        close_pct = float(history["Close"].dropna().iloc[-1])
        return close_pct / 100.0
    except Exception as exc:
        _LOG.debug("yfinance ^IRX fetch failed: %s", exc)
        return None


def _try_fred_dgs3mo(api_key: str) -> float | None:
    params: dict[str, str] = {
        "series_id": "DGS3MO",
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": "1",
    }
    try:
        resp = requests.get(_FRED_BASE, params=params, timeout=10)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        if not observations:
            return None
        value = observations[0].get("value")
        if value in (None, ".", ""):
            return None
        return float(value) / 100.0
    except (requests.RequestException, ValueError, KeyError) as exc:
        _LOG.debug("FRED DGS3MO fetch failed: %s", exc)
        return None


def get_risk_free_rate() -> float:
    """Return the current 3-month risk-free rate as a fraction (e.g. ``0.0525``)."""
    global _warned_fallback

    rate = _try_yfinance_irx()
    if rate is not None and 0.0 <= rate <= 0.25:
        return rate

    fred_key = os.environ.get("FRED_API_KEY")
    if fred_key:
        rate = _try_fred_dgs3mo(fred_key)
        if rate is not None and 0.0 <= rate <= 0.25:
            return rate

    if not _warned_fallback:
        _LOG.warning(
            "Risk-free rate fetch failed (yfinance ^IRX and FRED both unavailable). "
            "Falling back to hardcoded %.4f. Suppressing further warnings.",
            _DEFAULT_RATE,
        )
        _warned_fallback = True
    return _DEFAULT_RATE
