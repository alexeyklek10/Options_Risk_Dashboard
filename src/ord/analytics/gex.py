"""Gamma exposure (GEX) under the SqueezeMetrics dealer-sign convention.

Convention (assumed dealer position; see methodology notebook for the
discussion): dealers are SHORT calls and LONG puts. Following SqueezeMetrics
(2017), positive total GEX indicates a long-gamma market regime (dampened
realized vol), and negative total GEX indicates a short-gamma regime
(amplified realized vol).

Per-strike GEX in dollars per 1 percent move::

    GEX_call(K) =  gamma(K) * OI_call(K) * 100 * S**2 * 0.01
    GEX_put(K)  = -gamma(K) * OI_put(K)  * 100 * S**2 * 0.01
    total(K)    = GEX_call(K) + GEX_put(K)

Outputs include the per-strike GEX series, the gamma-flip level (the strike
at which cumulative GEX crosses zero walking from the highest strike
downward), and a zero-gamma level interpolated at spot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ord.pricing.greeks import gamma as gamma_fn

CONTRACT_MULTIPLIER: float = 100.0
PCT_MOVE: float = 0.01


@dataclass(frozen=True)
class GEXResult:
    """Per-strike GEX aggregated across the chain."""

    per_strike: pd.DataFrame  # columns: strike, gex_call, gex_put, gex_total
    total_gex: float
    gamma_flip_strike: float | None
    underlying_price: float


def gamma_exposure(chain: pd.DataFrame, r: float = 0.04, q: float = 0.0) -> GEXResult:
    """Compute per-strike GEX and the gamma-flip level."""
    if chain.empty:
        return GEXResult(
            per_strike=pd.DataFrame(columns=["strike", "gex_call", "gex_put", "gex_total"]),
            total_gex=float("nan"),
            gamma_flip_strike=None,
            underlying_price=float("nan"),
        )

    df = chain[chain["implied_vol"].notna() & (chain["implied_vol"] > 0)].copy()
    if df.empty:
        spot_fallback = float(chain["underlying_price"].iloc[0])
        return GEXResult(
            per_strike=pd.DataFrame(columns=["strike", "gex_call", "gex_put", "gex_total"]),
            total_gex=float("nan"),
            gamma_flip_strike=None,
            underlying_price=spot_fallback,
        )
    df["open_interest"] = df["open_interest"].fillna(0).astype(np.int64)
    spot = float(df["underlying_price"].iloc[0])

    scale = CONTRACT_MULTIPLIER * spot * spot * PCT_MOVE

    def _per_row_gex(row: pd.Series) -> float:
        t = max(int(row["dte"]), 1) / 365.0
        g = gamma_fn(spot, float(row["strike"]), t, r, float(row["implied_vol"]), q)
        signed = g if row["option_type"] == "call" else -g
        return float(signed * int(row["open_interest"]) * scale)

    df["gex"] = df.apply(_per_row_gex, axis=1)

    calls = df[df["option_type"] == "call"].groupby("strike")["gex"].sum()
    puts = df[df["option_type"] == "put"].groupby("strike")["gex"].sum()
    per_strike = (
        pd.concat([calls.rename("gex_call"), puts.rename("gex_put")], axis=1)
        .fillna(0.0)
        .reset_index()
    )
    per_strike["gex_total"] = per_strike["gex_call"] + per_strike["gex_put"]
    per_strike = per_strike.sort_values("strike").reset_index(drop=True)

    total = float(per_strike["gex_total"].sum())
    flip = _gamma_flip(per_strike)

    return GEXResult(
        per_strike=per_strike,
        total_gex=total,
        gamma_flip_strike=flip,
        underlying_price=spot,
    )


def _gamma_flip(per_strike: pd.DataFrame) -> float | None:
    """Strike where cumulative GEX crosses zero walking from highest strike downward."""
    if per_strike.empty:
        return None
    desc = per_strike.iloc[::-1].copy()
    desc["cum_gex"] = desc["gex_total"].cumsum()
    strikes = desc["strike"].to_numpy(dtype=np.float64)
    cum = desc["cum_gex"].to_numpy(dtype=np.float64)
    if (cum >= 0).all() or (cum <= 0).all():
        return None
    for i in range(len(cum) - 1):
        if cum[i] * cum[i + 1] <= 0:
            if cum[i] == cum[i + 1]:
                return float(strikes[i])
            # Linear interpolation in (cum_gex, strike) space.
            t = cum[i] / (cum[i] - cum[i + 1])
            return float(strikes[i] + t * (strikes[i + 1] - strikes[i]))
    return None
