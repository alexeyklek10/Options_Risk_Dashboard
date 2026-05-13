"""Implied-volatility solver: round-trip + arb-violation behavior.

Round trip on the Sobol grid: compute the BSM price at sigma, invert back via
the solver, assert the recovered sigma matches the input to within 1e-6.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ord.pricing.black_scholes import bs_price
from ord.pricing.greeks import vega
from ord.pricing.iv_solver import implied_vol

# Vega floor below which IV is not mathematically recoverable to 1e-6 precision:
# the price-vs-sigma curve is flat, so many sigmas produce the same price within
# machine epsilon. Cells at deep moneyness with low sigma and short T routinely
# land here on a wide Sobol grid.
_VEGA_RECOVERY_FLOOR = 1.0e-4


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_iv_round_trip_on_sobol_grid(sobol_grid: np.ndarray, option_type: str) -> None:
    max_err = 0.0
    skipped_degenerate = 0
    misses_in_recoverable_region = 0
    tested = 0
    for row in sobol_grid:
        S, k_over_s, T, r, sigma, q = (float(x) for x in row)
        K = S * k_over_s
        if vega(S, K, T, r, sigma, q) < _VEGA_RECOVERY_FLOOR:
            skipped_degenerate += 1
            continue
        price = bs_price(S, K, T, r, sigma, option_type, q)  # type: ignore[arg-type]
        recovered = implied_vol(price, S, K, T, r, option_type, q)  # type: ignore[arg-type]
        if recovered is None:
            misses_in_recoverable_region += 1
            continue
        tested += 1
        max_err = max(max_err, abs(recovered - sigma))
    assert tested > 900, f"too many cells skipped/missed; tested only {tested} of 1024"
    assert misses_in_recoverable_region == 0, (
        f"IV solver returned None in {misses_in_recoverable_region} "
        f"recoverable cells (vega above {_VEGA_RECOVERY_FLOOR})"
    )
    assert max_err < 1.0e-6, (
        f"IV {option_type} max round-trip err {max_err:.3e} "
        f"(tested={tested}, skipped_degenerate={skipped_degenerate})"
    )


def test_iv_returns_none_below_intrinsic() -> None:
    S, K, T, r, q = 100.0, 90.0, 1.0, 0.05, 0.0
    intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    # Price 1 dollar below the lower bound is unambiguously arb-violating.
    assert implied_vol(intrinsic - 1.0, S, K, T, r, "call", q) is None


def test_iv_returns_none_above_upper_bound() -> None:
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    upper = S * math.exp(-q * T)
    # Price above S * exp(-qT) is impossible for a European call.
    assert implied_vol(upper + 1.0, S, K, T, r, "call", q) is None


def test_iv_returns_none_at_zero_time_to_expiry() -> None:
    assert implied_vol(5.0, 100.0, 100.0, 0.0, 0.05, "call") is None


def test_iv_solver_handles_otm_call() -> None:
    # Far-OTM call: the initial Manaster-Koehler guess can land outside the
    # neighbourhood of the true root; the Brent fallback must catch it.
    S, K, T, r, q = 100.0, 150.0, 0.5, 0.03, 0.02
    target_sigma = 0.45
    price = bs_price(S, K, T, r, target_sigma, "call", q)
    recovered = implied_vol(price, S, K, T, r, "call", q)
    assert recovered is not None
    assert abs(recovered - target_sigma) < 1.0e-6


def test_iv_solver_handles_low_vol_atm() -> None:
    # Very low vol ATM is conditioned well (large vega-to-price ratio).
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.0, 0.0
    target_sigma = 0.06
    price = bs_price(S, K, T, r, target_sigma, "put", q)
    recovered = implied_vol(price, S, K, T, r, "put", q)
    assert recovered is not None
    assert abs(recovered - target_sigma) < 1.0e-6
