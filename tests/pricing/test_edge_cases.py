"""Edge-case coverage for pricing modules.

Covers expiry boundaries (T <= 0, sigma <= 0) across every Greek, and the
implied-vol solver branches: Brent fallback when Newton drifts out of bracket,
the intrinsic-boundary early return, and the arb-violation warning.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from ord.pricing.black_scholes import bs_price, bs_price_vec
from ord.pricing.greeks import charm, delta, gamma, rho, theta, vanna, vega
from ord.pricing.iv_solver import implied_vol

# ---------------------------------------------------------------------------
# Greeks at expiry / zero vol
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("S", "K", "option_type", "expected"),
    [
        (110.0, 100.0, "call", 1.0),  # ITM call
        (90.0, 100.0, "call", 0.0),  # OTM call
        (90.0, 100.0, "put", -1.0),  # ITM put
        (110.0, 100.0, "put", 0.0),  # OTM put
    ],
)
def test_delta_at_expiry(S: float, K: float, option_type: str, expected: float) -> None:
    assert delta(S, K, 0.0, 0.05, 0.2, option_type, 0.0) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "greek_fn",
    [
        lambda S, K: gamma(S, K, 0.0, 0.05, 0.2),
        lambda S, K: vega(S, K, 0.0, 0.05, 0.2),
        lambda S, K: vanna(S, K, 0.0, 0.05, 0.2),
    ],
)
def test_pure_greeks_zero_at_expiry(greek_fn) -> None:  # type: ignore[no-untyped-def]
    assert greek_fn(100.0, 100.0) == 0.0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_rho_charm_zero_at_expiry(option_type: str) -> None:
    assert theta(100.0, 100.0, 0.0, 0.05, 0.2, option_type, 0.0) == 0.0  # type: ignore[arg-type]
    assert rho(100.0, 100.0, 0.0, 0.05, 0.2, option_type, 0.0) == 0.0  # type: ignore[arg-type]
    assert charm(100.0, 100.0, 0.0, 0.05, 0.2, option_type, 0.0) == 0.0  # type: ignore[arg-type]


def test_all_greeks_zero_when_sigma_is_zero() -> None:
    # T > 0 but sigma == 0 should also fall through the early-return guard.
    assert gamma(100.0, 100.0, 1.0, 0.05, 0.0) == 0.0
    assert vega(100.0, 100.0, 1.0, 0.05, 0.0) == 0.0
    assert vanna(100.0, 100.0, 1.0, 0.05, 0.0) == 0.0
    assert theta(100.0, 100.0, 1.0, 0.05, 0.0, "call") == 0.0
    assert rho(100.0, 100.0, 1.0, 0.05, 0.0, "call") == 0.0
    assert charm(100.0, 100.0, 1.0, 0.05, 0.0, "call") == 0.0
    assert delta(100.0, 100.0, 1.0, 0.05, 0.0, "call") == 0.0  # K == S edge: not ITM


def test_bs_price_vec_with_mixed_expired_cells() -> None:
    S = np.array([100.0, 100.0, 100.0])
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([1.0, 0.0, 0.0])  # second and third are at expiry
    r = np.array([0.05, 0.05, 0.05])
    sigma = np.array([0.2, 0.2, 0.0])  # third also has zero vol
    q = np.array([0.0, 0.0, 0.0])

    call_prices = bs_price_vec(S, K, T, r, sigma, "call", q)
    # First cell: live BS price; second cell: intrinsic max(100-100, 0)=0;
    # third cell: intrinsic max(100-110, 0)=0.
    assert call_prices[0] > 0
    assert call_prices[1] == 0.0
    assert call_prices[2] == 0.0

    put_prices = bs_price_vec(S, K, T, r, sigma, "put", q)
    assert put_prices[1] == 0.0
    assert put_prices[2] == 10.0  # max(110-100, 0)


# ---------------------------------------------------------------------------
# IV solver branches
# ---------------------------------------------------------------------------


def test_iv_at_intrinsic_returns_none() -> None:
    # Price exactly at lower arb bound: BS not invertible (sigma -> 0 limit only).
    S, K, T, r, q = 100.0, 90.0, 0.5, 0.04, 0.0
    lower = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    assert implied_vol(lower, S, K, T, r, "call", q) is None


def test_iv_logs_warning_on_arb_violation(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="ord.pricing.iv_solver"):
        result = implied_vol(-1.0, 100.0, 100.0, 1.0, 0.05, "call", 0.0)
    assert result is None
    assert any("outside arb bounds" in record.message for record in caplog.records)


def test_iv_brent_fallback_when_newton_drifts_out_of_bracket() -> None:
    # Construct a case where the Manaster-Koehler initial guess is far from the
    # true root so Newton's first step is large; Brent must clean up. Deep OTM
    # put with moderate sigma fits the bill.
    S, K, T, r, q = 100.0, 60.0, 0.25, 0.05, 0.0
    true_sigma = 0.55
    price = bs_price(S, K, T, r, true_sigma, "put", q)
    recovered = implied_vol(price, S, K, T, r, "put", q)
    assert recovered is not None
    assert abs(recovered - true_sigma) < 1.0e-6


def test_iv_returns_none_when_bracket_does_not_change_sign() -> None:
    # Price above what sigma=5 produces but still below the upper arb bound
    # (S*exp(-qT)): the arb-bounds pre-check passes but no sigma in [1e-6, 5]
    # reaches the target, so f(SIGMA_LOW) and f(SIGMA_HIGH) have the same sign
    # and Brent returns None.
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    upper_limit_price = bs_price(S, K, T, r, 4.99, "call", q)
    assert implied_vol(upper_limit_price + 0.1, S, K, T, r, "call", q) is None


def test_iv_brent_fallback_runs_when_newton_runs_out_of_iterations() -> None:
    # maxiter=1 forces Newton to terminate after one step, exercising the
    # Brent fallback path explicitly.
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    target_sigma = 0.25
    price = bs_price(S, K, T, r, target_sigma, "call", q)
    recovered = implied_vol(price, S, K, T, r, "call", q, max_iter=1)
    assert recovered is not None
    assert abs(recovered - target_sigma) < 1.0e-6
