"""Implied-volatility surface across strikes and days-to-expiry.

Two output forms:

- ``observed_surface`` returns the raw, irregular (strike, DTE, IV) scatter
  exactly as it appears in the chain (one row per liquid option). This is what
  the 2D smile / smirk plots consume.
- ``interpolated_surface`` regrids the observations onto a regular
  ``(strike_grid, dte_grid)`` mesh via ``scipy.interpolate.griddata`` with cubic
  interpolation, falling back to nearest-neighbor for cells outside the convex
  hull of the observations. Returned as a dense ``IVSurface`` dataclass that
  the 3D Plotly surface trace consumes directly.

Rows with missing or non-positive IV are dropped before interpolation. The
caller is responsible for choosing whether to feed in provider IV or the
recomputed IV from :mod:`ord.pricing.iv_solver`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import griddata

FloatArray = npt.NDArray[np.float64]


@dataclass
class IVSurface:
    """Regular-grid implied-volatility surface.

    ``iv`` is a 2D array with shape ``(len(dte_grid), len(strike_grid))``.
    Cells outside the convex hull of the observations are filled by
    nearest-neighbor interpolation; ``mask`` is True where the cell required
    such extrapolation.
    """

    strike_grid: FloatArray
    dte_grid: FloatArray
    iv: FloatArray
    mask_extrapolated: npt.NDArray[np.bool_]


def observed_surface(
    chain: pd.DataFrame, side: Literal["call", "put", "both"] = "both"
) -> pd.DataFrame:
    """Return the irregular IV scatter from a chain.

    Drops rows with missing or non-positive implied vol. ``side="both"`` keeps
    calls and puts.
    """
    if chain.empty:
        return chain.iloc[0:0][["strike", "dte", "implied_vol", "option_type"]].copy()

    df = chain.copy()
    if side != "both":
        df = df[df["option_type"] == side]
    df = df[df["implied_vol"].notna() & (df["implied_vol"] > 0)]
    return df[["strike", "dte", "implied_vol", "option_type"]].reset_index(drop=True)


def interpolated_surface(
    chain: pd.DataFrame,
    strike_grid: FloatArray | None = None,
    dte_grid: FloatArray | None = None,
    side: Literal["call", "put", "both"] = "both",
    n_strikes: int = 40,
    n_dtes: int = 20,
) -> IVSurface | None:
    """Cubic-interpolate the observed IV scatter onto a regular grid.

    Returns ``None`` when there are fewer than 4 observations (insufficient
    for a cubic interpolation -- Delaunay needs at least 4 non-coplanar
    points).
    """
    obs = observed_surface(chain, side=side)
    if len(obs) < 4:
        return None

    strikes = obs["strike"].to_numpy(dtype=np.float64)
    dtes = obs["dte"].to_numpy(dtype=np.float64)
    ivs = obs["implied_vol"].to_numpy(dtype=np.float64)

    if strike_grid is None:
        strike_grid = np.linspace(strikes.min(), strikes.max(), n_strikes)
    if dte_grid is None:
        dte_grid = np.linspace(dtes.min(), dtes.max(), n_dtes)

    mesh_k, mesh_t = np.meshgrid(strike_grid, dte_grid)
    points = np.column_stack([strikes, dtes])
    cubic = griddata(points, ivs, (mesh_k, mesh_t), method="cubic")
    nearest = griddata(points, ivs, (mesh_k, mesh_t), method="nearest")
    extrapolated = np.isnan(cubic)
    iv_grid: FloatArray = np.where(extrapolated, nearest, cubic).astype(np.float64)

    return IVSurface(
        strike_grid=np.asarray(strike_grid, dtype=np.float64),
        dte_grid=np.asarray(dte_grid, dtype=np.float64),
        iv=iv_grid,
        mask_extrapolated=extrapolated,
    )
