"""Open-interest heatmaps: (strike x expiry) matrices for calls and puts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OIHeatmap:
    """Pivot tables of open interest per strike per expiry."""

    calls: pd.DataFrame  # index=strike, columns=expiry, values=OI
    puts: pd.DataFrame
    underlying_price: float


def oi_heatmap(chain: pd.DataFrame) -> OIHeatmap:
    """Return a per-side (strike x expiry) open-interest pivot."""
    if chain.empty:
        empty = pd.DataFrame()
        return OIHeatmap(calls=empty, puts=empty, underlying_price=float("nan"))

    spot = float(chain["underlying_price"].iloc[0])
    df = chain.copy()
    df["open_interest"] = df["open_interest"].fillna(0).astype(np.int64)

    def _pivot(side: str) -> pd.DataFrame:
        side_df = df[df["option_type"] == side]
        if side_df.empty:
            return pd.DataFrame()
        return side_df.pivot_table(
            index="strike", columns="expiry", values="open_interest", aggfunc="sum"
        ).fillna(0)

    return OIHeatmap(calls=_pivot("call"), puts=_pivot("put"), underlying_price=spot)
