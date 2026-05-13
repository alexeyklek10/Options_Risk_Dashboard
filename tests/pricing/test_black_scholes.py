"""Black-Scholes pricer validated against py_vollib.black_scholes_merton.

The grid is the Sobol sample defined in ``conftest.py``. The tolerance follows
BUILD_PROMPT section 6.4: 1e-8 absolute across all grid cells.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ord.pricing.black_scholes import bs_price, bs_price_vec

_BSM = pytest.importorskip("py_vollib.black_scholes_merton")


def _max_price_error(grid: np.ndarray, option_type: str) -> float:
    flag = "c" if option_type == "call" else "p"
    bsm = _BSM.black_scholes_merton
    errs = np.empty(grid.shape[0], dtype=np.float64)
    for i, row in enumerate(grid):
        S, k_over_s, T, r, sigma, q = row
        K = float(S * k_over_s)
        ours = bs_price(float(S), K, float(T), float(r), float(sigma), option_type, float(q))
        theirs = bsm(flag, float(S), K, float(T), float(r), float(sigma), float(q))
        errs[i] = abs(ours - theirs)
    return float(errs.max())


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bs_price_matches_py_vollib_on_sobol_grid(sobol_grid: np.ndarray, option_type: str) -> None:
    max_err = _max_price_error(sobol_grid, option_type)
    assert max_err < 1.0e-8, f"{option_type} max abs error {max_err:.3e}"


def test_bs_price_handles_zero_time_to_expiry() -> None:
    # At expiry, price collapses to intrinsic.
    assert bs_price(100.0, 90.0, 0.0, 0.05, 0.2, "call") == 10.0
    assert bs_price(100.0, 110.0, 0.0, 0.05, 0.2, "call") == 0.0
    assert bs_price(100.0, 110.0, 0.0, 0.05, 0.2, "put") == 10.0
    assert bs_price(100.0, 90.0, 0.0, 0.05, 0.2, "put") == 0.0


def test_bs_price_handles_zero_sigma() -> None:
    # Zero vol collapses to (deterministic) intrinsic value at expiry.
    assert bs_price(100.0, 90.0, 1.0, 0.0, 0.0, "call") == 10.0
    assert bs_price(100.0, 110.0, 1.0, 0.0, 0.0, "put") == 10.0


def test_put_call_parity_on_sobol_grid(sobol_grid: np.ndarray) -> None:
    # C - P = S * exp(-qT) - K * exp(-rT). Holds exactly under BSM.
    max_err = 0.0
    for row in sobol_grid:
        S, k_over_s, T, r, sigma, q = (float(x) for x in row)
        K = S * k_over_s
        c = bs_price(S, K, T, r, sigma, "call", q)
        p = bs_price(S, K, T, r, sigma, "put", q)
        parity = S * math.exp(-q * T) - K * math.exp(-r * T)
        max_err = max(max_err, abs((c - p) - parity))
    assert max_err < 1.0e-10, f"put-call parity max abs error {max_err:.3e}"


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bs_price_vec_matches_scalar(sobol_grid: np.ndarray, option_type: str) -> None:
    S = sobol_grid[:, 0]
    K = S * sobol_grid[:, 1]
    T = sobol_grid[:, 2]
    r = sobol_grid[:, 3]
    sigma = sobol_grid[:, 4]
    q = sobol_grid[:, 5]

    vec = bs_price_vec(S, K, T, r, sigma, option_type, q)
    scalar = np.array(
        [
            bs_price(
                float(S[i]),
                float(K[i]),
                float(T[i]),
                float(r[i]),
                float(sigma[i]),
                option_type,
                float(q[i]),
            )
            for i in range(len(S))
        ]
    )
    assert np.allclose(vec, scalar, atol=1.0e-12, rtol=0.0)
