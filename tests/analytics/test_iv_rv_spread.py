"""IV vs realized vol spread."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ord.analytics.iv_rv_spread import iv_rv_spread, realized_vol


def test_realized_vol_returns_none_when_insufficient_data() -> None:
    prices = pd.Series([100.0, 101.0])
    assert realized_vol(prices, window=21) is None


def test_realized_vol_matches_closed_form() -> None:
    # Constant 1% daily log return -> annualized vol = 0.01 * sqrt(252).
    rng = np.random.default_rng(7)
    rets = np.full(30, 0.01)
    prices = pd.Series(100.0 * np.exp(np.cumsum(rets)))
    rv = realized_vol(prices, window=21)
    assert rv is not None
    assert abs(rv - 0.01 * np.sqrt(252)) < 1e-6
    _ = rng  # silence pyright


def test_iv_rv_spread_computes_spread(synthetic_chain: pd.DataFrame) -> None:
    rng = np.random.default_rng(11)
    rets = rng.normal(0.0, 0.012, size=30)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(rets)),
        index=pd.date_range("2026-04-01", periods=30),
    )
    result = iv_rv_spread(synthetic_chain, prices, r=0.04, q=0.0)
    assert result.realized_vol_21d is not None
    assert result.atm_iv_21d is not None
    assert result.spread == result.atm_iv_21d - result.realized_vol_21d


def test_iv_rv_spread_handles_empty_chain() -> None:
    prices = pd.Series([100.0] * 25, index=pd.date_range("2026-04-01", periods=25))
    result = iv_rv_spread(pd.DataFrame(), prices)
    assert result.spread is None
    assert result.atm_iv_21d is None
    assert result.realized_vol_21d == 0.0  # zero returns -> zero vol


def test_iv_rv_spread_handles_no_long_dated_expiry() -> None:
    # Chain only has one short-dated expiry (under 21 DTE).
    rows = pd.DataFrame(
        {
            "expiry": ["2026-05-14"] * 4,
            "dte": [1, 1, 1, 1],
            "strike": [99.0, 100.0, 99.0, 100.0],
            "option_type": ["call", "call", "put", "put"],
            "implied_vol": [0.25, 0.24, 0.26, 0.25],
            "underlying_price": [100.0] * 4,
        }
    )
    rv_history = pd.Series([100.0] * 25)
    result = iv_rv_spread(rows, rv_history)
    assert result.atm_iv_21d is None
    assert result.spread is None
