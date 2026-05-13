"""Black-Scholes pricing for European options on equities with continuous dividend yield.

Formulas (Black & Scholes 1973, Merton 1973):

    d1 = (ln(S/K) + (r - q + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    Call = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
    Put  = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)

N(.) is the standard normal CDF.

Parameter convention used throughout the package:
    S       spot price of the underlying
    K       strike price
    T       time to expiration in years (positive; T <= 0 returns intrinsic)
    r       continuously compounded risk-free rate, annualized
    sigma   annualized volatility (as a fraction; 0.20 means 20 percent)
    q       continuous dividend yield, annualized (default 0)
    option_type  "call" or "put"

Validated against py_vollib.black_scholes_merton on a Sobol grid to within
1e-8 absolute in tests/pricing/test_black_scholes.py. py_vollib lives in
requirements-dev.txt only and is never imported from src/.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

OptionType = Literal["call", "put"]
FloatArray = npt.NDArray[np.float64]
FloatLike = float | FloatArray


def _intrinsic(S: float, K: float, option_type: OptionType) -> float:
    if option_type == "call":
        return max(S - K, 0.0)
    return max(K - S, 0.0)


def d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> tuple[float, float]:
    """Return the Black-Scholes ``d1`` and ``d2`` for a single (S, K, T, r, sigma, q)."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
) -> float:
    """Black-Scholes-Merton price of a European call or put (scalar).

    Per-share fair value, not multiplied by contract size. ``T <= 0`` collapses
    to intrinsic.
    """
    if T <= 0.0 or sigma <= 0.0:
        return _intrinsic(S, K, option_type)

    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    if option_type == "call":
        return float(S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2))
    return float(K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1))


def bs_price_vec(
    S: FloatLike,
    K: FloatLike,
    T: FloatLike,
    r: FloatLike,
    sigma: FloatLike,
    option_type: OptionType,
    q: FloatLike = 0.0,
) -> FloatArray:
    """Vectorized Black-Scholes-Merton price.

    All array inputs must broadcast; ``option_type`` is a single flag applied
    across all elements. ``T <= 0`` or ``sigma <= 0`` cells fall back to
    intrinsic value.
    """
    S_a = np.asarray(S, dtype=np.float64)
    K_a = np.asarray(K, dtype=np.float64)
    T_a = np.asarray(T, dtype=np.float64)
    r_a = np.asarray(r, dtype=np.float64)
    sigma_a = np.asarray(sigma, dtype=np.float64)
    q_a = np.asarray(q, dtype=np.float64)

    live = (T_a > 0.0) & (sigma_a > 0.0)
    # Clip to avoid log(0)/div-by-zero in dead cells; we'll mask them out anyway.
    T_safe = np.where(live, T_a, 1.0)
    sigma_safe = np.where(live, sigma_a, 1.0)

    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S_a / K_a) + (r_a - q_a + 0.5 * sigma_safe * sigma_safe) * T_safe) / (
        sigma_safe * sqrt_T
    )
    d2 = d1 - sigma_safe * sqrt_T
    disc_q = np.exp(-q_a * T_safe)
    disc_r = np.exp(-r_a * T_safe)

    if option_type == "call":
        live_price = S_a * disc_q * norm.cdf(d1) - K_a * disc_r * norm.cdf(d2)
        intrinsic = np.maximum(S_a - K_a, 0.0)
    else:
        live_price = K_a * disc_r * norm.cdf(-d2) - S_a * disc_q * norm.cdf(-d1)
        intrinsic = np.maximum(K_a - S_a, 0.0)

    result: FloatArray = np.where(live, live_price, intrinsic).astype(np.float64)
    return result
