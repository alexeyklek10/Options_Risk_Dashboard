"""Hand-rolled Black-Scholes pricer, Greeks, and implied-volatility solver.

py_vollib is never imported here -- it lives in ``requirements-dev.txt`` and is
used only by ``tests/pricing/`` as a reference oracle.
"""

from __future__ import annotations

from ord.pricing.black_scholes import (
    OptionType,
    bs_price,
    bs_price_vec,
    d1_d2,
)
from ord.pricing.greeks import (
    charm,
    delta,
    delta_vec,
    gamma,
    gamma_vec,
    rho,
    theta,
    vanna,
    vega,
    vega_vec,
)
from ord.pricing.iv_solver import implied_vol

__all__ = [
    "OptionType",
    "bs_price",
    "bs_price_vec",
    "charm",
    "d1_d2",
    "delta",
    "delta_vec",
    "gamma",
    "gamma_vec",
    "implied_vol",
    "rho",
    "theta",
    "vanna",
    "vega",
    "vega_vec",
]
