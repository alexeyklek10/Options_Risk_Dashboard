"""Verify the py_vollib reference oracle is importable and functional.

The hand-rolled pricer in ``src/ord/pricing/`` is validated against py_vollib on
a Sobol grid in step 2. This test catches the most common Windows breakage
mode -- the ``_testcapi`` shim in ``conftest.py`` failing to be applied before
py_vollib imports -- so the pricing tests do not start with a misleading
``ModuleNotFoundError``.
"""

from __future__ import annotations

import math


def test_py_vollib_prices_a_call() -> None:
    from py_vollib.black_scholes import black_scholes

    price = black_scholes("c", 100.0, 100.0, 1.0, 0.01, 0.2)
    assert math.isclose(price, 8.4333186901, abs_tol=1e-8)


def test_py_vollib_round_trips_iv() -> None:
    from py_vollib.black_scholes import black_scholes
    from py_vollib.black_scholes.implied_volatility import implied_volatility

    price = black_scholes("c", 100.0, 100.0, 1.0, 0.01, 0.2)
    recovered = implied_volatility(price, 100.0, 100.0, 1.0, 0.01, "c")
    assert math.isclose(recovered, 0.2, abs_tol=1e-6)
