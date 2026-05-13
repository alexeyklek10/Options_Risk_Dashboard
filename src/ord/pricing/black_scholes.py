"""Black-Scholes pricing for European options on equities with continuous dividend yield.

Exposes scalar (``bs_price``) and vectorized (``bs_price_vec``) pricers. All formulas
are implemented from first principles; the unit tests validate against py_vollib on a
1000-point Sobol grid to within 1e-8.

Implemented in step 2 of the build sequence.
"""

from __future__ import annotations
