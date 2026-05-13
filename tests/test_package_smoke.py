"""Package-layout smoke test.

Imports every top-level submodule of ``ord`` and asserts the package version.
Catches accidental breakage of the public layout (missing ``__init__``, removed
submodule, syntax error in a stub).
"""

from __future__ import annotations

import importlib

import pytest

import ord

EXPECTED_MODULES = [
    "ord.analytics",
    "ord.analytics.earnings_crush",
    "ord.analytics.expected_move",
    "ord.analytics.gex",
    "ord.analytics.greeks_dashboard",
    "ord.analytics.implied_pdf",
    "ord.analytics.iv_rv_spread",
    "ord.analytics.iv_surface",
    "ord.analytics.max_pain",
    "ord.analytics.oi_heatmap",
    "ord.analytics.pin_risk",
    "ord.analytics.put_call_ratio",
    "ord.analytics.skew",
    "ord.analytics.strategy_builder",
    "ord.data",
    "ord.data.aggregator",
    "ord.data.base",
    "ord.data.cache",
    "ord.data.polygon_provider",
    "ord.data.tradier_provider",
    "ord.data.validator",
    "ord.data.yfinance_provider",
    "ord.pricing",
    "ord.pricing.black_scholes",
    "ord.pricing.greeks",
    "ord.pricing.iv_solver",
    "ord.ui",
    "ord.ui.charts",
    "ord.ui.tabs",
    "ord.ui.tabs.data_quality",
    "ord.ui.tabs.gex",
    "ord.ui.tabs.greeks",
    "ord.ui.tabs.implied_pdf",
    "ord.ui.tabs.iv_surface",
    "ord.ui.tabs.iv_vs_rv",
    "ord.ui.tabs.max_pain_oi",
    "ord.ui.tabs.overview",
    "ord.ui.tabs.pin_risk",
    "ord.ui.tabs.skew_and_pcr",
    "ord.ui.tabs.strategy_builder",
    "ord.ui.theme",
    "ord.utils",
    "ord.utils.rates",
    "ord.utils.time",
]


@pytest.mark.parametrize("module_name", EXPECTED_MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_version_is_set() -> None:
    assert ord.__version__
    parts = ord.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])
