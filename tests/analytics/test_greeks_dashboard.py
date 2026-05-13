"""Greeks attach correctly to a chain; IV fall-through to recomputed sigma works."""

from __future__ import annotations

import pandas as pd

from ord.analytics.greeks_dashboard import GREEK_COLUMNS, attach_greeks


def test_attach_greeks_adds_all_seven_columns(synthetic_chain: pd.DataFrame) -> None:
    out = attach_greeks(synthetic_chain, r=0.04, q=0.0)
    for col in GREEK_COLUMNS:
        assert col in out.columns
    # ATM call: delta should be roughly 0.5.
    atm_call = out[(out["strike"] == 100.0) & (out["option_type"] == "call")].iloc[0]
    assert 0.4 < float(atm_call["delta"]) < 0.7


def test_attach_greeks_empty_chain_returns_typed_empty() -> None:
    empty = pd.DataFrame(columns=["strike", "underlying_price"])
    out = attach_greeks(empty, r=0.04)
    for col in GREEK_COLUMNS:
        assert col in out.columns


def test_attach_greeks_recomputes_when_provider_iv_missing(
    synthetic_chain: pd.DataFrame,
) -> None:
    chain = synthetic_chain.copy()
    chain["implied_vol"] = None  # force fall-through to recomputation from mid
    out = attach_greeks(chain, r=0.04, q=0.0, use_provider_iv=True)
    # Most rows must still have Greeks because mid is populated in the fixture.
    populated = out["delta"].notna().sum()
    assert populated > 0.6 * len(out), f"only {populated}/{len(out)} rows got greeks"


def test_attach_greeks_returns_nan_when_iv_unrecoverable(
    synthetic_chain: pd.DataFrame,
) -> None:
    chain = synthetic_chain.copy()
    chain["implied_vol"] = None
    chain["mid"] = None  # no price to recompute IV from
    out = attach_greeks(chain, r=0.04, q=0.0)
    assert out["delta"].isna().all()


def test_attach_greeks_put_call_delta_signs(synthetic_chain: pd.DataFrame) -> None:
    out = attach_greeks(synthetic_chain, r=0.04)
    call_deltas = out[out["option_type"] == "call"]["delta"].dropna()
    put_deltas = out[out["option_type"] == "put"]["delta"].dropna()
    assert (call_deltas >= 0).all()
    assert (put_deltas <= 0).all()


def test_attach_greeks_gamma_is_positive(synthetic_chain: pd.DataFrame) -> None:
    out = attach_greeks(synthetic_chain, r=0.04)
    g = out["gamma"].dropna()
    assert (g >= 0).all()
    assert g.max() > 0


def test_attach_greeks_use_provider_iv_false_recomputes_everything(
    synthetic_chain: pd.DataFrame,
) -> None:
    out = attach_greeks(synthetic_chain, r=0.04, use_provider_iv=False)
    # All rows have mid in the fixture, so recomputation should succeed broadly.
    assert out["delta"].notna().sum() > 0.6 * len(out)
