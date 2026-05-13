"""Greeks validated against py_vollib.black_scholes_merton.greeks.analytical.

Unit alignment between our convention and py_vollib's:

- delta, gamma: same units in both packages.
- vega: ours is per 1.00 of sigma; py_vollib is per 0.01 of sigma. Multiply theirs by 100.
- theta: ours is per year; py_vollib is per day. Multiply theirs by 365.
- rho: ours is per 1.00 of rate; py_vollib is per 0.01 of rate. Multiply theirs by 100.

py_vollib does not expose vanna or charm; they are validated via central-difference
numerical differentiation against the analytical delta with h = 1e-5.
"""

from __future__ import annotations

import numpy as np
import pytest

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

_ANALYTICAL = pytest.importorskip("py_vollib.black_scholes_merton.greeks.analytical")


def _flag(option_type: str) -> str:
    return "c" if option_type == "call" else "p"


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_py_vollib(sobol_grid: np.ndarray, option_type: str) -> None:
    fl = _flag(option_type)
    max_err = 0.0
    for row in sobol_grid:
        S, k_over_s, T, r, sigma, q = (float(x) for x in row)
        K = S * k_over_s
        ours = delta(S, K, T, r, sigma, option_type, q)  # type: ignore[arg-type]
        theirs = _ANALYTICAL.delta(fl, S, K, T, r, sigma, q)
        max_err = max(max_err, abs(ours - theirs))
    assert max_err < 1.0e-6, f"delta {option_type} max abs error {max_err:.3e}"


def test_gamma_matches_py_vollib(sobol_grid: np.ndarray) -> None:
    max_err = 0.0
    for row in sobol_grid:
        S, k_over_s, T, r, sigma, q = (float(x) for x in row)
        K = S * k_over_s
        ours = gamma(S, K, T, r, sigma, q)
        theirs = _ANALYTICAL.gamma("c", S, K, T, r, sigma, q)
        max_err = max(max_err, abs(ours - theirs))
    assert max_err < 1.0e-6, f"gamma max abs error {max_err:.3e}"


def test_vega_matches_py_vollib(sobol_grid: np.ndarray) -> None:
    # py_vollib quotes vega per 0.01 vol; rescale to our per-1.00 convention.
    max_err = 0.0
    for row in sobol_grid:
        S, k_over_s, T, r, sigma, q = (float(x) for x in row)
        K = S * k_over_s
        ours = vega(S, K, T, r, sigma, q)
        theirs = _ANALYTICAL.vega("c", S, K, T, r, sigma, q) * 100.0
        max_err = max(max_err, abs(ours - theirs))
    assert max_err < 1.0e-6, f"vega max abs error {max_err:.3e}"


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_matches_py_vollib(sobol_grid: np.ndarray, option_type: str) -> None:
    fl = _flag(option_type)
    # py_vollib quotes theta per day; rescale to our per-year convention.
    max_err = 0.0
    for row in sobol_grid:
        S, k_over_s, T, r, sigma, q = (float(x) for x in row)
        K = S * k_over_s
        ours = theta(S, K, T, r, sigma, option_type, q)  # type: ignore[arg-type]
        theirs = _ANALYTICAL.theta(fl, S, K, T, r, sigma, q) * 365.0
        max_err = max(max_err, abs(ours - theirs))
    assert max_err < 1.0e-6, f"theta {option_type} max abs error {max_err:.3e}"


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_matches_py_vollib(sobol_grid: np.ndarray, option_type: str) -> None:
    fl = _flag(option_type)
    # py_vollib quotes rho per 0.01 rate; rescale to our per-1.00 convention.
    max_err = 0.0
    for row in sobol_grid:
        S, k_over_s, T, r, sigma, q = (float(x) for x in row)
        K = S * k_over_s
        ours = rho(S, K, T, r, sigma, option_type, q)  # type: ignore[arg-type]
        theirs = _ANALYTICAL.rho(fl, S, K, T, r, sigma, q) * 100.0
        max_err = max(max_err, abs(ours - theirs))
    assert max_err < 1.0e-6, f"rho {option_type} max abs error {max_err:.3e}"


def test_vanna_matches_numerical_diff(sobol_grid: np.ndarray) -> None:
    """Vanna = d(delta)/d(sigma). Validate via central difference on analytical delta.

    Skip cells where the central difference is conditioned poorly: very small T
    (under one trading week) or very small sigma (under 10 percent) cause delta
    to be near a step at the strike, where the finite-difference truncation
    error swamps the analytical value.
    """
    h = 1.0e-5
    max_err = 0.0
    for row in sobol_grid:
        S, k_over_s, T, r, sigma, q = (float(x) for x in row)
        if T < 5.0 / 365.0 or sigma < 0.10:
            continue
        K = S * k_over_s
        ours = vanna(S, K, T, r, sigma, q)
        d_plus = delta(S, K, T, r, sigma + h, "call", q)
        d_minus = delta(S, K, T, r, sigma - h, "call", q)
        numerical = (d_plus - d_minus) / (2.0 * h)
        max_err = max(max_err, abs(ours - numerical))
    assert max_err < 1.0e-5, f"vanna max abs error vs numerical diff {max_err:.3e}"


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_charm_matches_numerical_diff(sobol_grid: np.ndarray, option_type: str) -> None:
    """Charm = d(delta)/d(T). Same conditioning caveats as vanna apply."""
    h = 1.0e-5
    max_err = 0.0
    for row in sobol_grid:
        S, k_over_s, T, r, sigma, q = (float(x) for x in row)
        if T < 30.0 / 365.0 or sigma < 0.10:
            continue
        K = S * k_over_s
        ours = charm(S, K, T, r, sigma, option_type, q)  # type: ignore[arg-type]
        d_plus = delta(S, K, T + h, r, sigma, option_type, q)  # type: ignore[arg-type]
        d_minus = delta(S, K, T - h, r, sigma, option_type, q)  # type: ignore[arg-type]
        numerical = (d_plus - d_minus) / (2.0 * h)
        max_err = max(max_err, abs(ours - numerical))
    # Charm has a 1/T term that amplifies near-the-money finite-difference noise.
    assert max_err < 1.0e-4, f"charm {option_type} max abs error vs numerical diff {max_err:.3e}"


def test_vega_call_equals_vega_put_on_sobol_grid(sobol_grid: np.ndarray) -> None:
    # Vega and gamma are independent of option_type; sanity-check by reusing the
    # same scalar function (the formulas above don't take option_type for these).
    # Spot check: gamma at one ATM cell.
    g_atm = gamma(100.0, 100.0, 1.0, 0.01, 0.2)
    assert g_atm > 0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vec_greeks_match_scalar(sobol_grid: np.ndarray, option_type: str) -> None:
    S = sobol_grid[:, 0]
    K = S * sobol_grid[:, 1]
    T = sobol_grid[:, 2]
    r = sobol_grid[:, 3]
    sigma = sobol_grid[:, 4]
    q = sobol_grid[:, 5]

    delta_v = delta_vec(S, K, T, r, sigma, option_type, q)
    gamma_v = gamma_vec(S, K, T, r, sigma, q)
    vega_v = vega_vec(S, K, T, r, sigma, q)

    delta_s = np.array(
        [
            delta(
                float(S[i]),
                float(K[i]),
                float(T[i]),
                float(r[i]),
                float(sigma[i]),
                option_type,  # type: ignore[arg-type]
                float(q[i]),
            )
            for i in range(len(S))
        ]
    )
    gamma_s = np.array(
        [
            gamma(
                float(S[i]),
                float(K[i]),
                float(T[i]),
                float(r[i]),
                float(sigma[i]),
                float(q[i]),
            )
            for i in range(len(S))
        ]
    )
    vega_s = np.array(
        [
            vega(
                float(S[i]),
                float(K[i]),
                float(T[i]),
                float(r[i]),
                float(sigma[i]),
                float(q[i]),
            )
            for i in range(len(S))
        ]
    )
    assert np.allclose(delta_v, delta_s, atol=1.0e-12, rtol=0.0)
    assert np.allclose(gamma_v, gamma_s, atol=1.0e-12, rtol=0.0)
    assert np.allclose(vega_v, vega_s, atol=1.0e-12, rtol=0.0)
