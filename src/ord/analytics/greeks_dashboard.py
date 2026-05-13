"""Per-strike, per-expiry Greeks recomputed from the hand-rolled pricer.

For each chain row this attaches the seven Greeks
(``delta``, ``gamma``, ``vega``, ``theta``, ``rho``, ``vanna``, ``charm``)
computed from ``ord.pricing`` using:

- ``T = dte / 365`` (calendar-day convention matches the schema).
- ``sigma = implied_vol`` if present, otherwise the IV recomputed from the
  mid price via :func:`ord.pricing.iv_solver.implied_vol` (callers can opt
  out via ``use_provider_iv=True``).
- A risk-free rate ``r`` supplied by the caller; the dashboard pulls it
  via :func:`ord.utils.rates.get_risk_free_rate`.
- A continuous dividend yield ``q`` defaulting to 0 (callers fetch via
  yfinance ``info["dividendYield"]`` and pass through).

Rows where IV cannot be resolved are kept with ``NaN`` Greeks.
"""

from __future__ import annotations

import pandas as pd

from ord.pricing.black_scholes import OptionType
from ord.pricing.greeks import charm, delta, gamma, rho, theta, vanna, vega
from ord.pricing.iv_solver import implied_vol

GREEK_COLUMNS: list[str] = ["delta", "gamma", "vega", "theta", "rho", "vanna", "charm"]


def attach_greeks(
    chain: pd.DataFrame,
    r: float,
    q: float = 0.0,
    use_provider_iv: bool = True,
) -> pd.DataFrame:
    """Return the chain with seven Greek columns appended.

    Parameters
    ----------
    chain
        Normalized chain DataFrame (see :class:`ord.data.base.ChainRow`).
    r
        Risk-free rate, continuously compounded annualized fraction.
    q
        Continuous dividend yield, annualized fraction.
    use_provider_iv
        If True, prefer the provider's ``implied_vol`` field. If False or
        the field is missing, recompute IV from ``mid`` via the hand-rolled
        solver.
    """
    if chain.empty:
        out = chain.copy()
        for col in GREEK_COLUMNS:
            out[col] = pd.array([], dtype="Float64")
        return out

    df = chain.copy()
    iv_used = _resolve_iv_per_row(df, r=r, q=q, use_provider_iv=use_provider_iv)

    delta_v: list[float | None] = []
    gamma_v: list[float | None] = []
    vega_v: list[float | None] = []
    theta_v: list[float | None] = []
    rho_v: list[float | None] = []
    vanna_v: list[float | None] = []
    charm_v: list[float | None] = []

    for (_, row), sigma in zip(df.iterrows(), iv_used, strict=False):
        if sigma is None or sigma <= 0:
            delta_v.append(None)
            gamma_v.append(None)
            vega_v.append(None)
            theta_v.append(None)
            rho_v.append(None)
            vanna_v.append(None)
            charm_v.append(None)
            continue
        s = float(row["underlying_price"])
        k = float(row["strike"])
        t = float(row["dte"]) / 365.0
        ot: OptionType = row["option_type"]
        delta_v.append(delta(s, k, t, r, sigma, ot, q))
        gamma_v.append(gamma(s, k, t, r, sigma, q))
        vega_v.append(vega(s, k, t, r, sigma, q))
        theta_v.append(theta(s, k, t, r, sigma, ot, q))
        rho_v.append(rho(s, k, t, r, sigma, ot, q))
        vanna_v.append(vanna(s, k, t, r, sigma, q))
        charm_v.append(charm(s, k, t, r, sigma, ot, q))

    df["delta"] = pd.array(delta_v, dtype="Float64")
    df["gamma"] = pd.array(gamma_v, dtype="Float64")
    df["vega"] = pd.array(vega_v, dtype="Float64")
    df["theta"] = pd.array(theta_v, dtype="Float64")
    df["rho"] = pd.array(rho_v, dtype="Float64")
    df["vanna"] = pd.array(vanna_v, dtype="Float64")
    df["charm"] = pd.array(charm_v, dtype="Float64")
    return df


def _resolve_iv_per_row(
    df: pd.DataFrame, r: float, q: float, use_provider_iv: bool
) -> list[float | None]:
    resolved: list[float | None] = []
    for _, row in df.iterrows():
        provider_iv = row.get("implied_vol")
        if (
            use_provider_iv
            and provider_iv is not None
            and not pd.isna(provider_iv)
            and float(provider_iv) > 0
        ):
            resolved.append(float(provider_iv))
            continue
        mid = row.get("mid")
        if mid is None or pd.isna(mid) or float(mid) <= 0:
            resolved.append(None)
            continue
        t = float(row["dte"]) / 365.0
        ot: OptionType = row["option_type"]
        recovered = implied_vol(
            float(mid),
            float(row["underlying_price"]),
            float(row["strike"]),
            t,
            r,
            ot,
            q,
        )
        resolved.append(recovered)
    return resolved
