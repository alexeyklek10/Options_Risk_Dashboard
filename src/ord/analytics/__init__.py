"""Analytics modules. Each is a pure function or small class with no global state.

Every analytic accepts a normalized chain DataFrame and returns either a DataFrame
or a typed result object. Every analytic has its own unit test in
``tests/analytics/`` using deterministic fixtures.
"""

from __future__ import annotations
