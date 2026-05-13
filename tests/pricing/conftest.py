"""Shared fixtures for the pricing test suite.

Builds a 1024-point Sobol grid over the (S, K/S, T, r, sigma, q) hypercube
specified in BUILD_PROMPT section 6.4. The grid is generated once per session
and reused across all pricing/greeks/IV tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import qmc

# Hypercube bounds: (S, K/S, T, r, sigma, q).
_LO = np.array([50.0, 0.5, 1.0 / 365.0, 0.0, 0.05, 0.0])
_HI = np.array([500.0, 1.5, 2.0, 0.08, 1.0, 0.04])


@pytest.fixture(scope="session")
def sobol_grid() -> np.ndarray:
    """1024-point Sobol sample over the BUILD_PROMPT section 6.4 hypercube.

    Columns: ``S``, ``K_over_S``, ``T``, ``r``, ``sigma``, ``q``.
    """
    sampler = qmc.Sobol(d=6, scramble=True, seed=42)
    # 2**10 = 1024 points; Sobol gives best low-discrepancy at powers of two.
    sample = sampler.random_base2(m=10)
    return np.asarray(qmc.scale(sample, _LO, _HI), dtype=np.float64)
