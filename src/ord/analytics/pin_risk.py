"""Pin-risk likelihood score for the nearest weekly / monthly expiry.

Three-signal score, normalized to 0-100:

1. Distance from spot: strikes within +/-2 percent of spot get full marks;
   linearly decays out to +/-5 percent.
2. Open interest decile: strikes in the top decile of the expiry's OI
   distribution get full marks; below the top quartile scores zero.
3. Gamma exposure proximity to the gamma-flip level: strikes within
   1 percent of the flip get full marks; linearly decays out to 5 percent.

Each signal contributes one third of the score.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ord.analytics.gex import gamma_exposure


@dataclass(frozen=True)
class PinCandidate:
    """One strike with its pin-likelihood score and the underlying signals."""

    strike: float
    score: float
    distance_pct: float
    oi: int
    gex_total: float


@dataclass(frozen=True)
class PinRiskResult:
    """Per-expiry pin-risk ranking."""

    expiry: date
    candidates: list[PinCandidate]
    spot: float
    gamma_flip_strike: float | None


def _distance_score(strike: float, spot: float) -> float:
    distance = abs(strike - spot) / spot
    if distance <= 0.02:
        return 1.0
    if distance >= 0.05:
        return 0.0
    return 1.0 - (distance - 0.02) / 0.03


def _oi_score(oi: int, decile_threshold: float, quartile_threshold: float) -> float:
    if oi >= decile_threshold:
        return 1.0
    if oi <= quartile_threshold:
        return 0.0
    span = decile_threshold - quartile_threshold
    if span <= 0:
        return 0.0
    return float((oi - quartile_threshold) / span)


def _gex_score(strike: float, flip: float | None) -> float:
    if flip is None:
        return 0.0
    distance = abs(strike - flip) / flip if flip > 0 else float("inf")
    if distance <= 0.01:
        return 1.0
    if distance >= 0.05:
        return 0.0
    return 1.0 - (distance - 0.01) / 0.04


def pin_risk(
    chain: pd.DataFrame,
    expiry: date,
    r: float = 0.04,
    q: float = 0.0,
    top_n: int = 10,
) -> PinRiskResult | None:
    if chain.empty or "expiry" not in chain.columns:
        return None
    expiry_chain = chain[chain["expiry"] == expiry]
    if expiry_chain.empty:
        return None
    spot = float(expiry_chain["underlying_price"].iloc[0])

    gex_result = gamma_exposure(expiry_chain, r=r, q=q)
    flip = gex_result.gamma_flip_strike
    gex_by_strike = gex_result.per_strike.set_index("strike")["gex_total"]

    oi_by_strike = expiry_chain.groupby("strike")["open_interest"].sum().fillna(0).astype(np.int64)
    if oi_by_strike.empty:
        return PinRiskResult(expiry=expiry, candidates=[], spot=spot, gamma_flip_strike=flip)
    decile = float(np.quantile(oi_by_strike, 0.90))
    quartile = float(np.quantile(oi_by_strike, 0.75))

    candidates: list[PinCandidate] = []
    for strike, oi in oi_by_strike.items():
        d_score = _distance_score(float(strike), spot)
        o_score = _oi_score(int(oi), decile, quartile)
        g_score = _gex_score(float(strike), flip)
        total = 100.0 * (d_score + o_score + g_score) / 3.0
        gex_val = float(gex_by_strike.get(strike, 0.0))
        candidates.append(
            PinCandidate(
                strike=float(strike),
                score=total,
                distance_pct=(float(strike) - spot) / spot,
                oi=int(oi),
                gex_total=gex_val,
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return PinRiskResult(
        expiry=expiry,
        candidates=candidates[:top_n],
        spot=spot,
        gamma_flip_strike=flip,
    )
