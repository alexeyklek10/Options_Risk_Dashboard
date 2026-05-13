"""Risk-neutral implied probability density via Breeden-Litzenberger (1978).

The risk-neutral density of the underlying at expiry T is the discounted
second derivative of the call price with respect to strike::

    f(K) = exp(r * T) * d^2 C / dK^2

We fit a univariate smoothing spline through the call-price-vs-strike curve
on the listed strikes, take the analytical second derivative, normalize the
result to integrate to one over the strike grid, and warn if the density is
non-positive anywhere (a sign of arbitrage in the input prices).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import UnivariateSpline

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ImpliedPDFResult:
    """Risk-neutral PDF for one expiry."""

    expiry: date
    strikes: FloatArray
    density: FloatArray
    has_negative_density: bool


def implied_pdf_for_expiry(
    chain: pd.DataFrame,
    expiry: date,
    r: float = 0.04,
    smoothing: float = 0.0,
    n_grid: int = 200,
) -> ImpliedPDFResult | None:
    """Compute the risk-neutral PDF for one expiry.

    Returns ``None`` when there are fewer than 4 call quotes (insufficient
    for a cubic spline).
    """
    side = chain[(chain["expiry"] == expiry) & (chain["option_type"] == "call")]
    side = side[side["mid"].notna() & (side["mid"] > 0)]
    if len(side) < 4:
        return None
    side = side.sort_values("strike")
    strikes = side["strike"].to_numpy(dtype=np.float64)
    prices = side["mid"].to_numpy(dtype=np.float64)
    dte = int(side["dte"].iloc[0])
    T = max(dte, 1) / 365.0

    # Fit a smoothing spline (cubic). UnivariateSpline requires strictly
    # increasing x; duplicated strikes are averaged.
    if len(np.unique(strikes)) != len(strikes):
        df = pd.DataFrame({"strike": strikes, "price": prices})
        df = df.groupby("strike", as_index=False).mean()
        strikes = df["strike"].to_numpy(dtype=np.float64)
        prices = df["price"].to_numpy(dtype=np.float64)
    if len(strikes) < 4:
        return None

    spline = UnivariateSpline(strikes, prices, k=3, s=smoothing)
    grid = np.linspace(strikes[0], strikes[-1], n_grid)
    second_deriv = spline.derivative(n=2)(grid)
    density: FloatArray = (np.exp(r * T) * second_deriv).astype(np.float64)

    has_negative = bool((density < 0).any())
    density = np.clip(density, 0.0, None)
    area = float(trapezoid(density, grid))
    if area > 0:
        density = density / area

    return ImpliedPDFResult(
        expiry=expiry,
        strikes=grid,
        density=density,
        has_negative_density=has_negative,
    )


def implied_pdf_all_expiries(
    chain: pd.DataFrame, r: float = 0.04, smoothing: float = 0.0, n_grid: int = 200
) -> dict[date, ImpliedPDFResult]:
    out: dict[date, ImpliedPDFResult] = {}
    for expiry in chain["expiry"].dropna().unique():
        exp_date = expiry.date() if hasattr(expiry, "date") else expiry
        result = implied_pdf_for_expiry(chain, exp_date, r=r, smoothing=smoothing, n_grid=n_grid)
        if result is not None:
            out[exp_date] = result
    return out
