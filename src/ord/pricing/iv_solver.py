"""Implied-volatility solver.

Newton-Raphson on vega using the Manaster-Koehler (1982) initial guess, falling
back to Brent's method (``scipy.optimize.brentq``) on the bracket ``[1e-6, 5.0]``
when Newton diverges, vega is too small, or a step lands outside that bracket.

Returns ``None`` for prices that violate the no-arbitrage intrinsic bounds.
"""

from __future__ import annotations

import logging
import math

from scipy.optimize import brentq

from ord.pricing.black_scholes import OptionType, bs_price
from ord.pricing.greeks import vega

_LOG = logging.getLogger(__name__)

SIGMA_LOW = 1.0e-6
SIGMA_HIGH = 5.0
VEGA_FLOOR = 1.0e-10
INITIAL_GUESS_FLOOR = 0.05
INITIAL_GUESS_CEIL = 2.0


def _arb_bounds(
    S: float, K: float, T: float, r: float, option_type: OptionType, q: float
) -> tuple[float, float]:
    """European-option no-arbitrage price bounds (lower, upper)."""
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    if option_type == "call":
        lower = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        lower = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r
    return lower, upper


def _manaster_koehler(S: float, K: float, T: float, r: float) -> float:
    """Manaster & Koehler (1982) closed-form initial guess for the IV solver.

    Clamped to [INITIAL_GUESS_FLOOR, INITIAL_GUESS_CEIL] to avoid degenerate
    starts (e.g. exactly zero at the money with r = 0).
    """
    raw = math.sqrt(abs(math.log(S / K) + r * T) * 2.0 / T)
    return max(INITIAL_GUESS_FLOOR, min(raw, INITIAL_GUESS_CEIL))


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    q: float = 0.0,
    tol: float = 1.0e-8,
    max_iter: int = 100,
) -> float | None:
    """Solve for the Black-Scholes implied volatility.

    Returns ``None`` when the input price violates the no-arbitrage bounds, or
    when no IV in ``[1e-6, 5.0]`` reproduces ``price`` to within ``tol`` even
    after the Brent fallback.
    """
    if T <= 0.0:
        return None

    lower, upper = _arb_bounds(S, K, T, r, option_type, q)
    # Allow a tiny float-precision slack on the bounds.
    slack = 1.0e-12
    if price < lower - slack or price > upper + slack:
        _LOG.warning(
            "implied_vol: price %.10g outside arb bounds [%.10g, %.10g] "
            "(S=%g, K=%g, T=%g, r=%g, q=%g, %s)",
            price,
            lower,
            upper,
            S,
            K,
            T,
            r,
            q,
            option_type,
        )
        return None
    # At the intrinsic boundary, BS is not invertible -- sigma -> 0 limit only.
    if price <= lower + slack:
        return None

    sigma = _manaster_koehler(S, K, T, r)

    # Newton-Raphson on vega. Convergence is checked on the change in sigma,
    # not on the residual price -- in flat-vega regions a tiny price residual
    # can coexist with a sigma very far from the true root, so a price-tolerance
    # exit returns garbage. Quadratic convergence near the root makes the
    # sigma-step criterion reliable.
    for _ in range(max_iter):
        p = bs_price(S, K, T, r, sigma, option_type, q)
        diff = p - price
        v = vega(S, K, T, r, sigma, q)
        if v < VEGA_FLOOR:  # pragma: no cover - hard to trigger deterministically; defensive guard
            break
        sigma_next = sigma - diff / v
        if not (SIGMA_LOW < sigma_next < SIGMA_HIGH):
            break
        if abs(sigma_next - sigma) < tol:
            return sigma_next
        sigma = sigma_next

    # Brent's method fallback on the full bracket.
    def f(s: float) -> float:
        return bs_price(S, K, T, r, s, option_type, q) - price

    f_lo = f(SIGMA_LOW)
    f_hi = f(SIGMA_HIGH)
    if f_lo * f_hi > 0.0:
        # No sign change in the bracket; solver can't converge.
        return None
    try:
        root = brentq(f, SIGMA_LOW, SIGMA_HIGH, xtol=tol, maxiter=200)
    except (
        ValueError,
        RuntimeError,
    ):  # pragma: no cover - defensive; pre-check rules out the only known trigger
        return None
    return float(root)
