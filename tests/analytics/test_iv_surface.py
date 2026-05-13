"""IV-surface observed scatter + cubic-interpolated regular grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ord.analytics.iv_surface import interpolated_surface, observed_surface


def test_observed_surface_drops_missing_iv(synthetic_chain: pd.DataFrame) -> None:
    chain = synthetic_chain.copy()
    chain.loc[chain.index[0], "implied_vol"] = None
    chain.loc[chain.index[1], "implied_vol"] = 0.0
    scatter = observed_surface(chain)
    assert (scatter["implied_vol"] > 0).all()
    assert scatter["implied_vol"].notna().all()


def test_observed_surface_can_filter_by_side(synthetic_chain: pd.DataFrame) -> None:
    calls_only = observed_surface(synthetic_chain, side="call")
    assert (calls_only["option_type"] == "call").all()


def test_observed_surface_handles_empty_chain() -> None:
    empty = pd.DataFrame(columns=["strike", "dte", "implied_vol", "option_type"])
    out = observed_surface(empty)
    assert out.empty


def test_interpolated_surface_returns_regular_grid(synthetic_chain: pd.DataFrame) -> None:
    surf = interpolated_surface(synthetic_chain, n_strikes=10, n_dtes=5)
    assert surf is not None
    assert surf.iv.shape == (5, 10)
    assert surf.strike_grid.shape == (10,)
    assert surf.dte_grid.shape == (5,)


def test_interpolated_surface_smile_shape(synthetic_chain: pd.DataFrame) -> None:
    surf = interpolated_surface(synthetic_chain, side="call", n_strikes=11, n_dtes=3)
    assert surf is not None
    # IV at the wings should be higher than at ATM for our v-shaped smile.
    atm_idx = int(np.argmin(np.abs(surf.strike_grid - 100.0)))
    iv_atm = float(surf.iv[1, atm_idx])
    iv_left = float(surf.iv[1, 0])
    iv_right = float(surf.iv[1, -1])
    assert iv_left > iv_atm
    assert iv_right > iv_atm


def test_interpolated_surface_returns_none_when_insufficient_obs() -> None:
    tiny = pd.DataFrame(
        [
            {"strike": 100.0, "dte": 30, "implied_vol": 0.2, "option_type": "call"},
            {"strike": 105.0, "dte": 30, "implied_vol": 0.21, "option_type": "call"},
        ]
    )
    # observed_surface needs the full schema; build directly here.
    surf = interpolated_surface(tiny.assign(**{c: 0 for c in ["bid"]}), n_strikes=5, n_dtes=2)
    assert surf is None
