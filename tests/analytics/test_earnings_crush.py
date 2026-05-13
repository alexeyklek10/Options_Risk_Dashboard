"""Earnings IV crush estimator."""

from __future__ import annotations

from datetime import date

import pandas as pd

from ord.analytics.earnings_crush import earnings_crush


def test_returns_no_event_vol_when_chain_lacks_two_post_expiries() -> None:
    chain = pd.DataFrame(
        {
            "expiry": [date(2026, 6, 19)] * 2,
            "dte": [37, 37],
            "strike": [100.0, 100.0],
            "option_type": ["call", "put"],
            "implied_vol": [0.4, 0.4],
            "underlying_price": [100.0, 100.0],
        }
    )
    result = earnings_crush(chain, earnings_date=date(2026, 6, 1))
    assert result.event_vol is None


def test_event_vol_positive_with_realistic_term_structure() -> None:
    # Construct a chain that satisfies the no-arb requirement: total variance
    # to the longer expiry > total variance to the shorter expiry.
    # iv_short=0.50, T_short=15/365, iv_long=0.40, T_long=37/365.
    chain = pd.DataFrame(
        [
            *[
                {
                    "expiry": date(2026, 5, 30),
                    "dte": 15,
                    "strike": float(s),
                    "option_type": ot,
                    "implied_vol": 0.50,
                    "underlying_price": 100.0,
                }
                for s in (95, 100, 105)
                for ot in ("call", "put")
            ],
            *[
                {
                    "expiry": date(2026, 6, 19),
                    "dte": 37,
                    "strike": float(s),
                    "option_type": ot,
                    "implied_vol": 0.40,
                    "underlying_price": 100.0,
                }
                for s in (95, 100, 105)
                for ot in ("call", "put")
            ],
        ]
    )
    result = earnings_crush(chain, earnings_date=date(2026, 5, 16), as_of=date(2026, 5, 13))
    assert result.event_vol is not None
    assert result.event_vol > 0
    assert result.post_event_vol is not None
    assert result.iv_pre == 0.50
    assert result.iv_post == 0.40


def test_event_vol_returns_none_for_calendar_arb_inputs() -> None:
    # iv_short much higher than iv_long causes implied negative post-event variance.
    chain = pd.DataFrame(
        [
            *[
                {
                    "expiry": date(2026, 5, 30),
                    "dte": 15,
                    "strike": float(s),
                    "option_type": ot,
                    "implied_vol": 0.80,  # absurdly high
                    "underlying_price": 100.0,
                }
                for s in (95, 100, 105)
                for ot in ("call", "put")
            ],
            *[
                {
                    "expiry": date(2026, 6, 19),
                    "dte": 37,
                    "strike": float(s),
                    "option_type": ot,
                    "implied_vol": 0.20,  # absurdly low for a longer-dated expiry
                    "underlying_price": 100.0,
                }
                for s in (95, 100, 105)
                for ot in ("call", "put")
            ],
        ]
    )
    result = earnings_crush(chain, earnings_date=date(2026, 5, 16), as_of=date(2026, 5, 13))
    # Calendar-arb input -> cannot decompose.
    assert result.event_vol is None


def test_historical_moves_computes_pct_change() -> None:
    prices = pd.Series(
        [100.0, 110.0, 110.0, 95.0, 95.0],
        index=pd.date_range("2026-01-01", periods=5),
    )
    earnings_dates = [date(2026, 1, 1), date(2026, 1, 3)]
    chain = pd.DataFrame()  # empty chain forces empty event_vol path
    result = earnings_crush(
        chain,
        earnings_date=None,
        underlying_history=prices,
        historical_earnings_dates=earnings_dates,
        n_earnings=2,
    )
    assert not result.historical_moves.empty
    # 2026-01-01 -> 2026-01-02: 100 -> 110 = +10%.
    # 2026-01-03 -> 2026-01-04: 110 -> 95 = -13.6%.
    moves = result.historical_moves.to_dict()
    assert abs(moves[date(2026, 1, 1)] - 0.10) < 1e-6
    assert abs(moves[date(2026, 1, 3)] - (95.0 - 110.0) / 110.0) < 1e-6


def test_empty_history_returns_empty_moves() -> None:
    result = earnings_crush(
        pd.DataFrame(),
        earnings_date=None,
        underlying_history=pd.Series(dtype="float64"),
        historical_earnings_dates=[date(2026, 1, 1)],
    )
    assert result.historical_moves.empty
