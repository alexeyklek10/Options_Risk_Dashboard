"""Application context passed to each tab's ``render`` function.

The Streamlit entry point assembles this once per run from the sidebar inputs
and the (cached) chain pull. Tabs treat it as read-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AppContext:
    """Read-only bundle of inputs available to every tab."""

    ticker: str
    chain: pd.DataFrame
    chains_by_provider: dict[str, pd.DataFrame]
    rate: float
    dividend_yield: float
    min_dte: int
    max_dte: int
    strike_lo_pct: float
    strike_hi_pct: float

    @property
    def filtered_chain(self) -> pd.DataFrame:
        """Chain filtered to the sidebar's DTE and strike-percent window."""
        if self.chain.empty:
            return self.chain
        spot = float(self.chain["underlying_price"].iloc[0])
        return self.chain[
            (self.chain["dte"] >= self.min_dte)
            & (self.chain["dte"] <= self.max_dte)
            & (self.chain["strike"] >= self.strike_lo_pct * spot)
            & (self.chain["strike"] <= self.strike_hi_pct * spot)
        ].copy()

    @property
    def spot(self) -> float:
        if self.chain.empty:
            return float("nan")
        return float(self.chain["underlying_price"].iloc[0])
