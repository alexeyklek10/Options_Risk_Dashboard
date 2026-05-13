"""ord: Options Risk Dashboard analytics package.

Submodules:

- ``ord.data``      Chain providers, aggregator, cross-source validator, parquet cache.
- ``ord.pricing``   Hand-rolled Black-Scholes pricer, Greeks, and implied-vol solver.
                    Validated against py_vollib in tests; never imports it at runtime.
- ``ord.analytics`` IV surface, skew, max pain, GEX, Breeden-Litzenberger implied PDF,
                    pin risk, earnings crush, multi-leg strategy P&L.
- ``ord.ui``        Streamlit tabs and shared Plotly chart factories.
- ``ord.utils``     Shared helpers (market-time, risk-free rate fetch with fallbacks).

See the repository README for the full architecture diagram and build status.
"""

from __future__ import annotations

__version__ = "0.1.0"
