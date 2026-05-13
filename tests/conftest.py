"""Shared pytest fixtures and configuration.

Includes a small ``_testcapi`` shim required for ``py_vollib`` / ``vollib`` to
import on CPython distributions that do not ship the private test module
(notably the Windows python.org installers for 3.11+). ``py_lets_be_rational``,
the C-accelerated dependency both packages share, does:

    from _testcapi import DBL_MIN, DBL_MAX

at import time -- but those values are simply ``sys.float_info.min`` and
``sys.float_info.max``. The shim must be installed in ``sys.modules`` before
any code path imports py_vollib, so it lives here rather than in individual
test files. py_vollib is only used in ``tests/`` as a reference oracle for the
hand-rolled pricer in ``src/ord/pricing/``; it is never imported from ``src/``.

Populated with concrete fixtures (Sobol-grid factory for pricing, synthetic-chain
loader for analytics) as those land in subsequent build steps.
"""

from __future__ import annotations

import sys
import types
import warnings


def _install_testcapi_shim() -> None:
    if "_testcapi" in sys.modules:
        return
    shim = types.ModuleType("_testcapi")
    shim.DBL_MIN = sys.float_info.min  # type: ignore[attr-defined]
    shim.DBL_MAX = sys.float_info.max  # type: ignore[attr-defined]
    sys.modules["_testcapi"] = shim


_install_testcapi_shim()

warnings.filterwarnings(
    "ignore",
    message="py_vollib is deprecated.*",
    category=DeprecationWarning,
)
