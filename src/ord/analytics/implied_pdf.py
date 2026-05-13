"""Risk-neutral implied probability density via Breeden-Litzenberger (1978).

The risk-neutral density of the underlying at expiry T is the discounted second
derivative of call price with respect to strike::

    f(K) = exp(r * T) * d^2 C / dK^2

Implementation fits a smoothing spline through the call-price-vs-strike curve,
takes the second derivative, normalizes, and warns when the resulting density
is non-positive at any point (indicates arbitrage violations in input prices).

Implemented in step 6 of the build sequence.
"""

from __future__ import annotations
