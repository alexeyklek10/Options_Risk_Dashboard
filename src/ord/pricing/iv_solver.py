"""Implied volatility solver.

Newton-Raphson on vega with the Manaster-Koehler (1982) initial guess; falls back
to Brent's method (``scipy.optimize.brentq``) on the bracket ``[1e-6, 5.0]`` when
Newton diverges or vega is too small. Returns ``None`` for prices that violate
the no-arbitrage intrinsic bounds.

Implemented in step 2 of the build sequence.
"""

from __future__ import annotations
