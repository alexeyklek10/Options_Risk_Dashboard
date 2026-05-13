"""Volatility skew: ATM IV, 25-delta risk reversal, 25-delta butterfly, near-ATM skope.

25-delta strikes are interpolated rather than rounded to the nearest listed
strike so the metric is comparable across expiries that have different listed
grids.

Per expiry we report:

- ``atm_iv`` -- IV at K = S, linearly interpolated across the listed strikes
  (separately for calls and puts, then averaged).
- ``rr_25d`` -- 25-delta risk reversal: ``IV(25d_call) - IV(25d_put)``. A
  negative number means OTM puts are bid (skew tilted toward downside fear).
- ``bf_25d`` -- 25-delta butterfly:
  ``(IV(25d_call) + IV(25d_put)) / 2 - atm_iv``. Positive = wings are bid
  relative to ATM (excess kurtosis priced in).
- ``slope`` -- OLS slope of IV vs log(K/S) in a +/- 10% window around spot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ord.pricing.greeks import delta as delta_fn


@dataclass(frozen=True)
class SkewResult:
    """Per-expiry skew metrics."""

    per_expiry: pd.DataFrame  # columns: expiry, atm_iv, rr_25d, bf_25d, slope


def _interp_iv_at_strike(
    strikes: npt.NDArray[np.float64], ivs: npt.NDArray[np.float64], k_target: float
) -> float | None:
    if len(strikes) < 2:
        return None
    sorted_idx = np.argsort(strikes)
    strikes_sorted = strikes[sorted_idx]
    ivs_sorted = ivs[sorted_idx]
    if k_target < strikes_sorted[0] or k_target > strikes_sorted[-1]:
        return None
    return float(np.interp(k_target, strikes_sorted, ivs_sorted))


def _find_25d_strike(
    strikes: npt.NDArray[np.float64],
    ivs: npt.NDArray[np.float64],
    spot: float,
    T: float,
    r: float,
    q: float,
    option_type: Literal["call", "put"],
) -> float | None:
    """Find the strike whose absolute delta is 0.25 by linear interpolation."""
    if len(strikes) < 2:
        return None
    target_abs = 0.25
    deltas = np.array(
        [
            abs(delta_fn(spot, float(k), T, r, float(v), option_type, q))
            for k, v in zip(strikes, ivs, strict=False)
        ]
    )
    # Find consecutive strikes whose deltas bracket the target.
    sorted_idx = np.argsort(strikes)
    strikes_sorted = strikes[sorted_idx]
    deltas_sorted = deltas[sorted_idx]
    for i in range(len(strikes_sorted) - 1):
        a, b = deltas_sorted[i], deltas_sorted[i + 1]
        if (a - target_abs) * (b - target_abs) <= 0:
            # Linear interpolation in (delta, strike) space.
            if a == b:
                return float(strikes_sorted[i])
            t = (target_abs - a) / (b - a)
            return float(strikes_sorted[i] + t * (strikes_sorted[i + 1] - strikes_sorted[i]))
    return None


def _slope_near_atm(
    strikes: npt.NDArray[np.float64], ivs: npt.NDArray[np.float64], spot: float
) -> float | None:
    """OLS slope of IV vs log(K/S) within +/- 10% of spot."""
    window = np.abs(strikes / spot - 1.0) <= 0.10
    if window.sum() < 3:
        return None
    x = np.log(strikes[window] / spot)
    y = ivs[window]
    # OLS slope via numpy polyfit (degree 1).
    coeffs = np.polyfit(x, y, deg=1)
    return float(coeffs[0])


def _skew_for_expiry(
    side: pd.DataFrame,
    spot: float,
    expiry: date,
    r: float,
    q: float,
) -> dict[str, object]:
    calls = side[side["option_type"] == "call"]
    puts = side[side["option_type"] == "put"]
    call_strikes = calls["strike"].to_numpy(dtype=np.float64)
    call_ivs = calls["implied_vol"].to_numpy(dtype=np.float64)
    put_strikes = puts["strike"].to_numpy(dtype=np.float64)
    put_ivs = puts["implied_vol"].to_numpy(dtype=np.float64)

    atm_call = _interp_iv_at_strike(call_strikes, call_ivs, spot)
    atm_put = _interp_iv_at_strike(put_strikes, put_ivs, spot)
    if atm_call is None and atm_put is None:
        atm = float("nan")
    elif atm_call is None:
        atm = float(atm_put)  # type: ignore[arg-type]
    elif atm_put is None:
        atm = float(atm_call)
    else:
        atm = (atm_call + atm_put) / 2.0

    dte_days = int(side["dte"].iloc[0])
    T = max(dte_days, 1) / 365.0

    rr = bf = float("nan")
    if len(call_strikes) >= 2 and len(put_strikes) >= 2:
        k_25c = _find_25d_strike(call_strikes, call_ivs, spot, T, r, q, "call")
        k_25p = _find_25d_strike(put_strikes, put_ivs, spot, T, r, q, "put")
        iv_25c = _interp_iv_at_strike(call_strikes, call_ivs, k_25c) if k_25c is not None else None
        iv_25p = _interp_iv_at_strike(put_strikes, put_ivs, k_25p) if k_25p is not None else None
        if iv_25c is not None and iv_25p is not None:
            rr = iv_25c - iv_25p
            if not np.isnan(atm):
                bf = (iv_25c + iv_25p) / 2.0 - atm

    combined_strikes = np.concatenate([call_strikes, put_strikes])
    combined_ivs = np.concatenate([call_ivs, put_ivs])
    slope = _slope_near_atm(combined_strikes, combined_ivs, spot)

    return {
        "expiry": expiry,
        "atm_iv": atm,
        "rr_25d": rr,
        "bf_25d": bf,
        "slope": float("nan") if slope is None else slope,
    }


def skew(chain: pd.DataFrame, r: float = 0.04, q: float = 0.0) -> SkewResult:
    """Compute per-expiry skew metrics for the chain."""
    if chain.empty:
        return SkewResult(
            per_expiry=pd.DataFrame(columns=["expiry", "atm_iv", "rr_25d", "bf_25d", "slope"])
        )
    df = chain[chain["implied_vol"].notna() & (chain["implied_vol"] > 0)].copy()
    if df.empty:
        return SkewResult(
            per_expiry=pd.DataFrame(columns=["expiry", "atm_iv", "rr_25d", "bf_25d", "slope"])
        )

    spot = float(df["underlying_price"].iloc[0])
    rows: list[dict[str, object]] = []
    for expiry, group in df.groupby("expiry"):
        exp_date = expiry.date() if hasattr(expiry, "date") else expiry
        rows.append(_skew_for_expiry(group, spot, exp_date, r, q))
    return SkewResult(per_expiry=pd.DataFrame(rows))
