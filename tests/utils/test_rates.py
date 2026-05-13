"""Risk-free rate fetcher fallback chain: yfinance ^IRX, FRED, hardcoded 0.04."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from ord.utils import rates


@pytest.fixture(autouse=True)
def _reset_warning_flag() -> None:
    rates._warned_fallback = False


class _StubTicker:
    def __init__(self, close_series: pd.Series) -> None:
        self._series = close_series

    def history(self, period: str, interval: str) -> pd.DataFrame:
        return pd.DataFrame({"Close": self._series})


def _stub_yf_with_close(monkeypatch: pytest.MonkeyPatch, close_pct: float | None) -> None:
    series = (
        pd.Series([close_pct], dtype="float64")
        if close_pct is not None
        else pd.Series([], dtype="float64")
    )

    class _StubYF:
        @staticmethod
        def Ticker(ticker: str) -> _StubTicker:  # noqa: N802
            return _StubTicker(series)

    monkeypatch.setattr(rates, "_try_yfinance_irx", lambda: rates._try_yfinance_irx.__wrapped__())
    import sys

    monkeypatch.setitem(sys.modules, "yfinance", _StubYF)


def test_irx_primary_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rates, "_try_yfinance_irx", lambda: 0.0525)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert rates.get_risk_free_rate() == pytest.approx(0.0525)


def test_falls_back_to_fred_when_irx_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rates, "_try_yfinance_irx", lambda: None)
    monkeypatch.setenv("FRED_API_KEY", "fake-key")
    monkeypatch.setattr(rates, "_try_fred_dgs3mo", lambda _key: 0.041)
    assert rates.get_risk_free_rate() == pytest.approx(0.041)


def test_falls_back_to_hardcoded_when_all_sources_fail(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(rates, "_try_yfinance_irx", lambda: None)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with caplog.at_level("WARNING", logger="ord.utils.rates"):
        rate = rates.get_risk_free_rate()
    assert rate == pytest.approx(rates._DEFAULT_RATE)
    assert any("Falling back to hardcoded" in r.message for r in caplog.records)


def test_warning_is_emitted_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rates, "_try_yfinance_irx", lambda: None)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    rates.get_risk_free_rate()
    # Second call must not re-warn.
    assert rates._warned_fallback is True


def test_rejects_out_of_range_irx_value(monkeypatch: pytest.MonkeyPatch) -> None:
    # If yfinance returns a nonsense rate (e.g. negative or >25%), drop through.
    monkeypatch.setattr(rates, "_try_yfinance_irx", lambda: 1.5)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert rates.get_risk_free_rate() == pytest.approx(rates._DEFAULT_RATE)


def test_rejects_out_of_range_fred_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rates, "_try_yfinance_irx", lambda: None)
    monkeypatch.setenv("FRED_API_KEY", "fake")
    monkeypatch.setattr(rates, "_try_fred_dgs3mo", lambda _: -0.5)
    assert rates.get_risk_free_rate() == pytest.approx(rates._DEFAULT_RATE)


def test_try_yfinance_irx_uses_close(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the real ``_try_yfinance_irx`` codepath with a stubbed yfinance.
    class _Stub:
        @staticmethod
        def Ticker(name: str) -> _StubTicker:  # noqa: N802 - mirrors yfinance.Ticker
            return _StubTicker(pd.Series([5.23], dtype="float64"))

    import sys

    monkeypatch.setitem(sys.modules, "yfinance", _Stub)
    assert rates._try_yfinance_irx() == pytest.approx(0.0523)


def test_try_fred_dgs3mo_uses_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"observations": [{"value": "4.25"}]}

    def fake_get(*_args: Any, **_kwargs: Any) -> _Resp:
        return _Resp()

    monkeypatch.setattr(rates.requests, "get", fake_get)
    assert rates._try_fred_dgs3mo("fake") == pytest.approx(0.0425)


def test_try_fred_dgs3mo_handles_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"observations": [{"value": "."}]}  # FRED's missing-data sentinel

    monkeypatch.setattr(rates.requests, "get", lambda *_a, **_k: _Resp())
    assert rates._try_fred_dgs3mo("fake") is None
