"""Max-pain: strike that minimizes dollar pain to option holders if the stock pins.

Per-expiry argmin over the candidate strike set of::

    pain(K) = sum_calls OI_i * max(K - K_i, 0) * 100
            + sum_puts  OI_j * max(K_j - K, 0) * 100

The contract multiplier (100) is factored in for dollar units. The candidate
strike set is the union of all strikes that have nonzero open interest in the
expiry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

CONTRACT_MULTIPLIER: float = 100.0


@dataclass(frozen=True)
class MaxPainResult:
    """Max-pain result for one expiry."""

    expiry: date
    strike: float
    dollar_pain: float
    pain_curve: pd.DataFrame  # columns: strike, pain


def max_pain_for_expiry(chain: pd.DataFrame, expiry: date) -> MaxPainResult | None:
    """Compute max-pain for a single expiration; return ``None`` if no OI."""
    expiry_chain = chain[chain["expiry"] == expiry]
    expiry_chain = expiry_chain[expiry_chain["open_interest"].fillna(0) > 0]
    if expiry_chain.empty:
        return None

    strikes = np.sort(expiry_chain["strike"].unique())
    pains = np.empty(len(strikes), dtype=np.float64)
    calls = expiry_chain[expiry_chain["option_type"] == "call"]
    puts = expiry_chain[expiry_chain["option_type"] == "put"]
    call_strikes = calls["strike"].to_numpy(dtype=np.float64)
    call_oi = calls["open_interest"].fillna(0).to_numpy(dtype=np.float64)
    put_strikes = puts["strike"].to_numpy(dtype=np.float64)
    put_oi = puts["open_interest"].fillna(0).to_numpy(dtype=np.float64)

    for i, k in enumerate(strikes):
        call_pain = float(np.sum(call_oi * np.maximum(k - call_strikes, 0.0)))
        put_pain = float(np.sum(put_oi * np.maximum(put_strikes - k, 0.0)))
        pains[i] = (call_pain + put_pain) * CONTRACT_MULTIPLIER

    idx = int(np.argmin(pains))
    return MaxPainResult(
        expiry=expiry,
        strike=float(strikes[idx]),
        dollar_pain=float(pains[idx]),
        pain_curve=pd.DataFrame({"strike": strikes, "pain": pains}),
    )


def max_pain_all_expiries(chain: pd.DataFrame) -> dict[date, MaxPainResult]:
    """Map of ``expiry -> MaxPainResult`` for every expiry with OI."""
    out: dict[date, MaxPainResult] = {}
    for expiry in chain["expiry"].dropna().unique():
        d = expiry.date() if hasattr(expiry, "date") else expiry
        result = max_pain_for_expiry(chain, d)
        if result is not None:
            out[d] = result
    return out
