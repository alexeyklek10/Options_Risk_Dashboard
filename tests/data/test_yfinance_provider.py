"""YFinanceProvider: unit tests with a stubbed yfinance and an opt-in integration test.

The integration test (``@pytest.mark.integration``) is skipped by default in
CI; run it locally with ``pytest -m integration`` when verifying live SPY
fetches.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest

from ord.data import yfinance_provider as yfp
from ord.data.base import CHAIN_COLUMNS, ProviderRateLimitError


# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


def test_coerce_optional_float_handles_nan_none_and_value() -> None:
    assert yfp._coerce_optional_float(float("nan")) is None
    assert yfp._coerce_optional_float(None) is None
    assert yfp._coerce_optional_float("not-a-float") is None
    assert yfp._coerce_optional_float("1.5") == 1.5


def test_coerce_optional_int_handles_nan_none_and_value() -> None:
    assert yfp._coerce_optional_int(float("nan")) is None
    assert yfp._coerce_optional_int(None) is None
    assert yfp._coerce_optional_int("not-an-int") is None
    assert yfp._coerce_optional_int(3.7) == 3


def test_midpoint_requires_positive_quotes() -> None:
    assert yfp._midpoint(1.0, 1.2) == pytest.approx(1.1)
    assert yfp._midpoint(None, 1.2) is None
    assert yfp._midpoint(1.0, None) is None
    assert yfp._midpoint(0.0, 1.2) is None
    assert yfp._midpoint(1.0, 0.0) is None


# ---------------------------------------------------------------------------
# Provider tests with stubbed yfinance
# ---------------------------------------------------------------------------


class _StubFastInfo:
    def __init__(self, last_price: float) -> None:
        self._lp = last_price

    def __getitem__(self, key: str) -> float:
        if key != "last_price":
            raise KeyError(key)
        return self._lp


class _StubTicker:
    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame, options: list[str]) -> None:
        self.calls = calls
        self.puts = puts
        self.options = tuple(options)
        self.fast_info = _StubFastInfo(101.25)

    def option_chain(self, expiry: str) -> "_StubChain":  # noqa: ARG002
        return _StubChain(self.calls, self.puts)

    def history(self, period: str, interval: str) -> pd.DataFrame:  # noqa: ARG002
        return pd.DataFrame({"Close": [100.0]})


class _StubChain:
    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame) -> None:
        self.calls = calls
        self.puts = puts


def _raw_side(*, strikes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strike": strikes,
            "bid": [1.0 for _ in strikes],
            "ask": [1.1 for _ in strikes],
            "lastPrice": [1.05 for _ in strikes],
            "volume": [10 for _ in strikes],
            "openInterest": [100 for _ in strikes],
            "impliedVolatility": [0.25 for _ in strikes],
        }
    )


@pytest.fixture()
def stub_yf(monkeypatch: pytest.MonkeyPatch) -> _StubTicker:
    calls = _raw_side(strikes=[95.0, 100.0, 105.0])
    puts = _raw_side(strikes=[95.0, 100.0, 105.0])
    stub = _StubTicker(calls=calls, puts=puts, options=["2026-06-19", "2026-07-17"])

    class _StubYF:
        Ticker = lambda _ticker: stub  # noqa: E731

    monkeypatch.setattr(yfp, "yfinance", _StubYF, raising=False)

    # YFinanceProvider.__init__ also does ``import yfinance``; intercept it.
    import sys

    monkeypatch.setitem(sys.modules, "yfinance", _StubYF)
    return stub


def test_get_underlying_price_uses_fast_info(stub_yf: _StubTicker) -> None:  # noqa: ARG001
    provider = yfp.YFinanceProvider()
    assert provider.get_underlying_price("TST") == pytest.approx(101.25)


def test_get_expirations_returns_parsed_dates(stub_yf: _StubTicker) -> None:  # noqa: ARG001
    provider = yfp.YFinanceProvider()
    expiries = provider.get_expirations("TST")
    assert expiries == [date(2026, 6, 19), date(2026, 7, 17)]


def test_get_chain_normalizes_columns(stub_yf: _StubTicker) -> None:  # noqa: ARG001
    provider = yfp.YFinanceProvider()
    df = provider.get_chain("TST", date(2026, 6, 19))
    assert list(df.columns) == CHAIN_COLUMNS
    # 3 calls + 3 puts.
    assert len(df) == 6
    assert set(df["option_type"]) == {"call", "put"}
    # midpoint computed correctly.
    assert df["mid"].iloc[0] == pytest.approx(1.05)


def test_get_chain_raises_rate_limit_error_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenTicker:
        options = ("2026-06-19",)
        fast_info = _StubFastInfo(100.0)

        def option_chain(self, expiry: str) -> None:  # noqa: ARG002
            raise RuntimeError("rate limit")

        def history(self, period: str, interval: str) -> pd.DataFrame:  # noqa: ARG002
            return pd.DataFrame({"Close": [100.0]})

    class _StubYF:
        Ticker = lambda _t: _BrokenTicker()  # noqa: E731

    import sys

    monkeypatch.setitem(sys.modules, "yfinance", _StubYF)
    monkeypatch.setattr(yfp, "yfinance", _StubYF, raising=False)
    provider = yfp.YFinanceProvider()
    with pytest.raises(ProviderRateLimitError):
        provider.get_chain("TST", date(2026, 6, 19))


def test_get_underlying_price_falls_back_to_history(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Ticker:
        options = ("2026-06-19",)

        class _BrokenInfo:
            def __getitem__(self, key: str) -> float:  # noqa: ARG002
                raise KeyError(key)

        fast_info = _BrokenInfo()

        def history(self, period: str, interval: str) -> pd.DataFrame:  # noqa: ARG002
            return pd.DataFrame({"Close": [99.5]})

    class _StubYF:
        Ticker = lambda _t: _Ticker()  # noqa: E731

    import sys

    monkeypatch.setitem(sys.modules, "yfinance", _StubYF)
    provider = yfp.YFinanceProvider()
    assert provider.get_underlying_price("TST") == pytest.approx(99.5)


def test_get_underlying_price_raises_when_history_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Ticker:
        class _BrokenInfo:
            def __getitem__(self, key: str) -> float:  # noqa: ARG002
                raise KeyError(key)

        fast_info = _BrokenInfo()

        def history(self, period: str, interval: str) -> pd.DataFrame:  # noqa: ARG002
            return pd.DataFrame()

    class _StubYF:
        Ticker = lambda _t: _Ticker()  # noqa: E731

    import sys

    monkeypatch.setitem(sys.modules, "yfinance", _StubYF)
    provider = yfp.YFinanceProvider()
    with pytest.raises(ProviderRateLimitError):
        provider.get_underlying_price("TST")


# ---------------------------------------------------------------------------
# Live integration (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ORD_INTEGRATION") != "1",
    reason="integration tests opt-in via ORD_INTEGRATION=1",
)
def test_live_spy_full_chain_smoke() -> None:
    provider = yfp.YFinanceProvider()
    df = provider.get_full_chain("SPY", max_expiries=2)
    assert not df.empty
    assert set(df["option_type"]) == {"call", "put"}
    assert (df["strike"] > 0).all()
