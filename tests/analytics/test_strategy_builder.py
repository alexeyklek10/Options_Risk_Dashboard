"""Multi-leg strategy P&L surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from ord.analytics.strategy_builder import Leg, evaluate
from ord.pricing.black_scholes import bs_price


def test_long_call_payoff_is_hockey_stick() -> None:
    spot = 100.0
    sigma = 0.2
    T = 0.5
    r = 0.03
    entry = bs_price(spot, 100.0, T, r, sigma, "call")
    leg = Leg(
        leg_type="call", strike=100.0, expiry_T=T, sigma=sigma, quantity=1.0, entry_price=entry
    )
    result = evaluate([leg], spot=spot, r=r, n_spots=51, n_times=2)
    # Max profit unbounded -> equals the max of the spot grid - K - entry.
    assert result.max_profit == pytest.approx(max(result.spot_grid) - 100.0 - entry, abs=1e-6)
    # Max loss equals premium paid.
    assert result.max_loss == pytest.approx(-entry, abs=1e-6)


def test_long_straddle_has_two_breakevens() -> None:
    spot = 100.0
    sigma = 0.25
    T = 0.5
    r = 0.03
    c = bs_price(spot, 100.0, T, r, sigma, "call")
    p = bs_price(spot, 100.0, T, r, sigma, "put")
    legs = [
        Leg("call", 100.0, T, sigma, 1.0, c),
        Leg("put", 100.0, T, sigma, 1.0, p),
    ]
    result = evaluate(legs, spot=spot, r=r, spot_window_pct=0.5, n_spots=201)
    assert len(result.breakevens) == 2
    # Symmetric: breakevens are roughly equidistant from spot.
    diffs = sorted(abs(b - spot) for b in result.breakevens)
    assert abs(diffs[0] - diffs[1]) < 1.0


def test_evaluate_requires_at_least_one_leg() -> None:
    with pytest.raises(ValueError):
        evaluate([], spot=100.0)


def test_greeks_sum_across_legs() -> None:
    spot = 100.0
    sigma = 0.2
    T = 0.5
    r = 0.03
    legs = [
        Leg("call", 100.0, T, sigma, 1.0, bs_price(spot, 100.0, T, r, sigma, "call")),
        Leg("call", 110.0, T, sigma, -1.0, bs_price(spot, 110.0, T, r, sigma, "call")),
    ]
    result = evaluate(legs, spot=spot, r=r, n_spots=11, n_times=2)
    # Bull call spread -> positive net delta, bounded.
    assert result.greeks["delta"] > 0
    assert result.greeks["delta"] < 1.0


def test_stock_leg_only_pnl_is_linear() -> None:
    leg = Leg("stock", strike=0.0, expiry_T=0.0, sigma=0.0, quantity=100.0, entry_price=100.0)
    result = evaluate([leg], spot=100.0, n_spots=11, n_times=2)
    # Long 100 shares at 100 -> PnL = 100 * (spot - 100). Should be linear.
    pnl = result.pnl_at_expiry
    slopes = np.diff(pnl) / np.diff(result.spot_grid)
    assert np.allclose(slopes, 100.0, atol=1e-6)
