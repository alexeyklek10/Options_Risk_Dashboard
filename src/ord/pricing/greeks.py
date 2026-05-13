"""European-option Greeks: delta, gamma, vega, theta, rho, vanna, charm.

All Greeks per-unit (not scaled by contract multiplier). Scalar and vectorized
variants. Units are documented per function (e.g. theta annualized; divide by
365 for per-day). Vanna and charm are validated against numerical differentiation
in tests because py_vollib does not expose them directly.

Implemented in step 2 of the build sequence.
"""

from __future__ import annotations
