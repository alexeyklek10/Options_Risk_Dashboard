"""Gamma exposure: per-strike series, totals, gamma-flip detection."""

from __future__ import annotations

import pandas as pd

from ord.analytics.gex import gamma_exposure


def test_gex_returns_one_row_per_strike(synthetic_chain: pd.DataFrame) -> None:
    result = gamma_exposure(synthetic_chain)
    n_strikes = synthetic_chain["strike"].nunique()
    assert len(result.per_strike) == n_strikes


def test_gex_call_is_positive_and_put_is_negative(synthetic_chain: pd.DataFrame) -> None:
    # SqueezeMetrics convention: dealers short calls, long puts.
    result = gamma_exposure(synthetic_chain)
    call_sums = result.per_strike["gex_call"].dropna()
    put_sums = result.per_strike["gex_put"].dropna()
    assert (call_sums >= 0).all()
    assert (put_sums <= 0).all()


def test_gex_total_is_signed(synthetic_chain: pd.DataFrame) -> None:
    result = gamma_exposure(synthetic_chain)
    # Fixture has equal call/put OI at every strike, so total nets to ~0.
    assert abs(result.total_gex) < 1e6  # well below any single strike contribution


def test_gex_includes_underlying_price(synthetic_chain: pd.DataFrame) -> None:
    result = gamma_exposure(synthetic_chain)
    assert result.underlying_price == 100.0


def test_gex_empty_chain() -> None:
    result = gamma_exposure(pd.DataFrame())
    assert result.per_strike.empty
    assert result.gamma_flip_strike is None


def test_gex_chain_without_iv_returns_empty_per_strike() -> None:
    chain = pd.DataFrame(
        {
            "expiry": ["2026-06-19"] * 2,
            "dte": [37, 37],
            "strike": [100.0, 105.0],
            "option_type": ["call", "put"],
            "open_interest": [100, 100],
            "implied_vol": [None, None],
            "underlying_price": [100.0, 100.0],
        }
    )
    result = gamma_exposure(chain)
    assert result.per_strike.empty


def test_gamma_flip_detected_on_asymmetric_chain() -> None:
    # Build a chain where puts dominate at the low strikes (negative GEX) and
    # calls dominate at the high strikes (positive GEX): cumulative GEX walking
    # down from the top should cross zero somewhere in the middle.
    rows: list[dict[str, object]] = []
    for strike, call_oi, put_oi in [
        (90.0, 10, 5000),
        (100.0, 10, 1000),
        (110.0, 5000, 10),
    ]:
        for option_type, oi in [("call", call_oi), ("put", put_oi)]:
            rows.append(
                {
                    "expiry": "2026-06-19",
                    "dte": 30,
                    "strike": strike,
                    "option_type": option_type,
                    "open_interest": oi,
                    "implied_vol": 0.25,
                    "underlying_price": 100.0,
                }
            )
    chain = pd.DataFrame(rows)
    result = gamma_exposure(chain)
    assert result.gamma_flip_strike is not None
    assert 90.0 <= result.gamma_flip_strike <= 110.0


def test_gamma_flip_none_when_cumulative_never_crosses_zero() -> None:
    # All-call chain: GEX is positive everywhere, cum_gex never crosses.
    rows: list[dict[str, object]] = []
    for strike in (100.0, 105.0, 110.0):
        rows.append(
            {
                "expiry": "2026-06-19",
                "dte": 30,
                "strike": strike,
                "option_type": "call",
                "open_interest": 1000,
                "implied_vol": 0.2,
                "underlying_price": 100.0,
            }
        )
    result = gamma_exposure(pd.DataFrame(rows))
    assert result.gamma_flip_strike is None
