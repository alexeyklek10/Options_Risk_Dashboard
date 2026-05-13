"""Open-interest heatmap pivot tables."""

from __future__ import annotations

import pandas as pd

from ord.analytics.oi_heatmap import oi_heatmap


def test_oi_heatmap_pivots_per_side(synthetic_chain: pd.DataFrame) -> None:
    h = oi_heatmap(synthetic_chain)
    assert h.calls.shape[0] > 0  # at least one strike row
    assert h.puts.shape[0] > 0
    assert h.calls.shape[1] == 2  # two expiries
    assert h.puts.shape[1] == 2


def test_oi_heatmap_concentration_at_atm(synthetic_chain: pd.DataFrame) -> None:
    h = oi_heatmap(synthetic_chain)
    # Fixture concentrates near-expiry OI at K=100; the max value of the calls
    # column for that expiry must be at the K=100 row.
    near_expiry = synthetic_chain["expiry"].min()
    col = h.calls[near_expiry]
    assert col.idxmax() == 100.0


def test_oi_heatmap_empty_chain() -> None:
    empty = pd.DataFrame()
    h = oi_heatmap(empty)
    assert h.calls.empty
    assert h.puts.empty
