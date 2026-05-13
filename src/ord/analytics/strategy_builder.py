"""Multi-leg strategy P&L surfaces.

A position is a list of ``Leg`` records. Each leg is a single option contract
(``"call"`` / ``"put"`` / ``"stock"``) with a signed quantity (positive long,
negative short). Per-leg expiries can differ -- the surface reprices each
unexpired leg via Black-Scholes at the future spot and time.

Outputs:

- ``pnl_at_expiry``: P&L as a function of spot at the latest leg's expiry
  (all options have settled), expressed per share of underlying (callers
  multiply by 100 for per-contract dollars).
- ``pnl_surface``: 2D grid (spot x time-to-expiry) of mark-to-market P&L,
  pricing un-expired option legs via Black-Scholes at the implied vol the
  caller supplies per leg.
- Aggregate breakevens, max profit, max loss, and current Greek totals are
  attached as separate fields on ``StrategyResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from ord.pricing.black_scholes import bs_price
from ord.pricing.greeks import charm, delta, gamma, rho, theta, vanna, vega

FloatArray = npt.NDArray[np.float64]
LegType = Literal["call", "put", "stock"]


@dataclass(frozen=True)
class Leg:
    """One leg of a multi-leg strategy.

    ``quantity`` is signed (positive = long, negative = short). For
    ``"stock"`` legs, ``strike`` and ``expiry_T`` are ignored.
    """

    leg_type: LegType
    strike: float
    expiry_T: float  # noqa: N815 - paired with module-wide quant T notation  # years to expiry at the snapshot the position was opened
    sigma: float  # implied vol the caller wants used for repricing un-expired legs
    quantity: float
    entry_price: float  # price paid (call/put) or stock cost basis


@dataclass(frozen=True)
class StrategyResult:
    """P&L diagnostics for a multi-leg position."""

    spot_grid: FloatArray
    pnl_at_expiry: FloatArray
    pnl_surface: FloatArray  # shape (len(time_grid), len(spot_grid))
    time_grid: FloatArray  # years remaining until the last leg expires
    breakevens: list[float] = field(default_factory=list)
    max_profit: float = float("nan")
    max_loss: float = float("nan")
    greeks: dict[str, float] = field(default_factory=dict)


def _leg_payoff_at_spot(leg: Leg, spot: FloatArray) -> FloatArray:
    if leg.leg_type == "stock":
        result: FloatArray = (leg.quantity * (spot - leg.entry_price)).astype(np.float64)
        return result
    if leg.leg_type == "call":
        intrinsic = np.maximum(spot - leg.strike, 0.0)
    else:
        intrinsic = np.maximum(leg.strike - spot, 0.0)
    return (leg.quantity * (intrinsic - leg.entry_price)).astype(np.float64)


def _leg_mark_at(leg: Leg, spot: FloatArray, t_remaining: float, r: float, q: float) -> FloatArray:
    if leg.leg_type == "stock":
        return (leg.quantity * (spot - leg.entry_price)).astype(np.float64)
    # The leg's T at snapshot was leg.expiry_T; remaining time is bounded above by that.
    t = min(t_remaining, leg.expiry_T)
    prices = np.array(
        [bs_price(float(s), leg.strike, max(t, 0.0), r, leg.sigma, leg.leg_type, q) for s in spot]
    )
    return (leg.quantity * (prices - leg.entry_price)).astype(np.float64)


def _greeks_aggregate(legs: list[Leg], spot: float, r: float, q: float) -> dict[str, float]:
    totals: dict[str, float] = {
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0,
        "rho": 0.0,
        "vanna": 0.0,
        "charm": 0.0,
    }
    for leg in legs:
        if leg.leg_type == "stock":
            totals["delta"] += leg.quantity
            continue
        t = leg.expiry_T
        totals["delta"] += leg.quantity * delta(spot, leg.strike, t, r, leg.sigma, leg.leg_type, q)
        totals["gamma"] += leg.quantity * gamma(spot, leg.strike, t, r, leg.sigma, q)
        totals["vega"] += leg.quantity * vega(spot, leg.strike, t, r, leg.sigma, q)
        totals["theta"] += leg.quantity * theta(spot, leg.strike, t, r, leg.sigma, leg.leg_type, q)
        totals["rho"] += leg.quantity * rho(spot, leg.strike, t, r, leg.sigma, leg.leg_type, q)
        totals["vanna"] += leg.quantity * vanna(spot, leg.strike, t, r, leg.sigma, q)
        totals["charm"] += leg.quantity * charm(spot, leg.strike, t, r, leg.sigma, leg.leg_type, q)
    return totals


def _breakevens(spot_grid: FloatArray, pnl: FloatArray) -> list[float]:
    crossings: list[float] = []
    for i in range(len(spot_grid) - 1):
        if pnl[i] * pnl[i + 1] <= 0 and pnl[i] != pnl[i + 1]:
            t = pnl[i] / (pnl[i] - pnl[i + 1])
            crossings.append(float(spot_grid[i] + t * (spot_grid[i + 1] - spot_grid[i])))
    return crossings


def evaluate(
    legs: list[Leg],
    spot: float,
    spot_grid: FloatArray | None = None,
    time_grid: FloatArray | None = None,
    r: float = 0.04,
    q: float = 0.0,
    spot_window_pct: float = 0.30,
    n_spots: int = 100,
    n_times: int = 20,
) -> StrategyResult:
    """Evaluate the strategy across the spot range and (optionally) a time grid."""
    if not legs:
        raise ValueError("at least one leg required")

    if spot_grid is None:
        lo = spot * (1.0 - spot_window_pct)
        hi = spot * (1.0 + spot_window_pct)
        spot_grid = np.linspace(lo, hi, n_spots)
    if time_grid is None:
        max_T = max((leg.expiry_T for leg in legs if leg.leg_type != "stock"), default=0.0)
        time_grid = np.linspace(0.0, max_T, n_times)

    pnl_expiry = sum(
        (_leg_payoff_at_spot(leg, spot_grid) for leg in legs), np.zeros_like(spot_grid)
    )

    pnl_surface = np.zeros((len(time_grid), len(spot_grid)), dtype=np.float64)
    for i, t_remaining in enumerate(time_grid):
        contribution = sum(
            (_leg_mark_at(leg, spot_grid, float(t_remaining), r, q) for leg in legs),
            np.zeros_like(spot_grid),
        )
        pnl_surface[i] = contribution

    return StrategyResult(
        spot_grid=spot_grid,
        pnl_at_expiry=pnl_expiry,
        pnl_surface=pnl_surface,
        time_grid=time_grid,
        breakevens=_breakevens(spot_grid, pnl_expiry),
        max_profit=float(np.max(pnl_expiry)),
        max_loss=float(np.min(pnl_expiry)),
        greeks=_greeks_aggregate(legs, spot, r, q),
    )
