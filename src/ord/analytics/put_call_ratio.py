"""Put/call ratio: per-expiry and aggregate, by volume and by open interest.

A 20-day rolling PCR helper is provided for use with cached historical
aggregate snapshots (see :mod:`ord.data.cache`); on first run with no
history the rolling output is empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PCRResult:
    """Put-call ratios for one chain snapshot."""

    by_volume_aggregate: float | None
    by_oi_aggregate: float | None
    per_expiry: pd.DataFrame  # columns: expiry, pcr_volume, pcr_oi


def _ratio(puts_sum: float, calls_sum: float) -> float | None:
    if calls_sum <= 0:
        return None
    return float(puts_sum / calls_sum)


def put_call_ratio(chain: pd.DataFrame) -> PCRResult:
    """Compute volume- and OI-weighted PCR per expiry plus the aggregate."""
    if chain.empty:
        return PCRResult(
            by_volume_aggregate=None,
            by_oi_aggregate=None,
            per_expiry=pd.DataFrame(columns=["expiry", "pcr_volume", "pcr_oi"]),
        )

    df = chain.copy()
    df["volume"] = df["volume"].fillna(0)
    df["open_interest"] = df["open_interest"].fillna(0)

    grouped = df.groupby(["expiry", "option_type"], dropna=False)[["volume", "open_interest"]].sum()
    grouped = grouped.unstack(fill_value=0)
    expiries: list[dict[str, object]] = []
    for expiry, row in grouped.iterrows():
        v_calls = float(row.get(("volume", "call"), 0))
        v_puts = float(row.get(("volume", "put"), 0))
        oi_calls = float(row.get(("open_interest", "call"), 0))
        oi_puts = float(row.get(("open_interest", "put"), 0))
        expiries.append(
            {
                "expiry": expiry,
                "pcr_volume": _ratio(v_puts, v_calls),
                "pcr_oi": _ratio(oi_puts, oi_calls),
            }
        )
    per_expiry = pd.DataFrame(expiries)

    total_calls_vol = float(df.loc[df["option_type"] == "call", "volume"].sum())
    total_puts_vol = float(df.loc[df["option_type"] == "put", "volume"].sum())
    total_calls_oi = float(df.loc[df["option_type"] == "call", "open_interest"].sum())
    total_puts_oi = float(df.loc[df["option_type"] == "put", "open_interest"].sum())

    return PCRResult(
        by_volume_aggregate=_ratio(total_puts_vol, total_calls_vol),
        by_oi_aggregate=_ratio(total_puts_oi, total_calls_oi),
        per_expiry=per_expiry,
    )


def rolling_pcr(history: pd.DataFrame, window: int = 20) -> pd.Series:
    """20-day rolling mean of an aggregate PCR time series.

    ``history`` must have a sorted ``DatetimeIndex`` and a numeric ``pcr``
    column. Returns an empty series if there are fewer than ``window``
    observations.
    """
    if history.empty or "pcr" not in history.columns:
        return pd.Series(dtype="float64")
    if len(history) < window:
        return pd.Series(dtype="float64")
    return history["pcr"].rolling(window=window).mean().dropna()
