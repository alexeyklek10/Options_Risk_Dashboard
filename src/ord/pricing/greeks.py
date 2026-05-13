"""European-option Greeks: delta, gamma, vega, theta, rho, vanna, charm.

All Greeks per-share (not scaled by contract multiplier).

Units convention (matches the standard textbook convention; conversion notes
spell out how to map to the per-1-vol-point / per-day quotes most platforms
display):

- delta    dimensionless; ``d(price)/d(S)``.
- gamma    per dollar of S; ``d^2(price)/d(S)^2``.
- vega     PER 1.00 of sigma. Divide by 100 for "per 1 vol point" (e.g. per 0.01).
- theta    PER YEAR of T. Divide by 365 for per-calendar-day theta.
- rho      PER 1.00 of r. Divide by 100 for "per 1 percentage point" of rate.
- vanna    ``d(delta)/d(sigma)`` = ``d(vega)/d(S)``. Per 1.00 of sigma.
- charm    ``d(delta)/d(T)`` (positive when delta grows as more time-to-expiry).
           Per YEAR. Divide by 365 for per-day.

Formulas implemented from first principles. Validated against
py_vollib.black_scholes_merton.greeks.analytical on a Sobol grid in tests, with
the appropriate unit rescaling. Vanna and charm have no py_vollib counterpart
so they are validated by central-difference numerical differentiation against
the analytical delta with h = 1e-5.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from ord.pricing.black_scholes import OptionType, d1_d2

FloatArray = npt.NDArray[np.float64]


def _phi(x: float) -> float:
    """Standard normal PDF (scalar)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _phi_vec(x: FloatArray) -> FloatArray:
    """Standard normal PDF (vectorized)."""
    result: FloatArray = (np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)).astype(np.float64)
    return result


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
) -> float:
    """Delta: sensitivity of price to spot, per share."""
    if T <= 0.0 or sigma <= 0.0:
        # At expiry, delta is a step: 1 ITM, 0 OTM (or -1 / 0 for puts).
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    disc_q = math.exp(-q * T)
    if option_type == "call":
        return float(disc_q * norm.cdf(d1))
    return float(disc_q * (norm.cdf(d1) - 1.0))


def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Gamma: sensitivity of delta to spot. Same for calls and puts (put-call symmetry)."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return float(math.exp(-q * T) * _phi(d1) / (S * sigma * math.sqrt(T)))


def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Vega per 1.00 of sigma. Same for calls and puts."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return float(S * math.exp(-q * T) * _phi(d1) * math.sqrt(T))


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
) -> float:
    """Theta per YEAR. Divide by 365 for per-calendar-day theta."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    common = -S * disc_q * _phi(d1) * sigma / (2.0 * math.sqrt(T))
    if option_type == "call":
        return float(common - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1))
    return float(common + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1))


def rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
) -> float:
    """Rho per 1.00 of rate."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_r = math.exp(-r * T)
    if option_type == "call":
        return float(K * T * disc_r * norm.cdf(d2))
    return float(-K * T * disc_r * norm.cdf(-d2))


def vanna(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Vanna: d(delta)/d(sigma) = d(vega)/d(S). Same for calls and puts."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return float(-math.exp(-q * T) * _phi(d1) * d2 / sigma)


def charm(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
) -> float:
    """Charm: d(delta)/d(T) in per-YEAR units. Divide by 365 for per-day.

    Sign convention: positive charm means delta grows with more time-to-expiry.
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    sqrt_T = math.sqrt(T)
    disc_q = math.exp(-q * T)
    # Term that is shared between call and put.
    shared = (
        disc_q * _phi(d1) * (2.0 * (r - q) * T - d2 * sigma * sqrt_T) / (2.0 * T * sigma * sqrt_T)
    )
    if option_type == "call":
        return float(-q * disc_q * norm.cdf(d1) + shared)
    return float(q * disc_q * norm.cdf(-d1) + shared)


# ---------------------------------------------------------------------------
# Vectorized variants
# ---------------------------------------------------------------------------


def _d1_d2_vec(
    S: FloatArray,
    K: FloatArray,
    T: FloatArray,
    r: FloatArray,
    sigma: FloatArray,
    q: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1.astype(np.float64), d2.astype(np.float64)


def delta_vec(
    S: FloatArray,
    K: FloatArray,
    T: FloatArray,
    r: FloatArray,
    sigma: FloatArray,
    option_type: OptionType,
    q: FloatArray,
) -> FloatArray:
    d1, _ = _d1_d2_vec(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    if option_type == "call":
        result: FloatArray = (disc_q * norm.cdf(d1)).astype(np.float64)
    else:
        result = (disc_q * (norm.cdf(d1) - 1.0)).astype(np.float64)
    return result


def gamma_vec(
    S: FloatArray,
    K: FloatArray,
    T: FloatArray,
    r: FloatArray,
    sigma: FloatArray,
    q: FloatArray,
) -> FloatArray:
    d1, _ = _d1_d2_vec(S, K, T, r, sigma, q)
    result: FloatArray = (np.exp(-q * T) * _phi_vec(d1) / (S * sigma * np.sqrt(T))).astype(
        np.float64
    )
    return result


def vega_vec(
    S: FloatArray,
    K: FloatArray,
    T: FloatArray,
    r: FloatArray,
    sigma: FloatArray,
    q: FloatArray,
) -> FloatArray:
    d1, _ = _d1_d2_vec(S, K, T, r, sigma, q)
    result: FloatArray = (S * np.exp(-q * T) * _phi_vec(d1) * np.sqrt(T)).astype(np.float64)
    return result
