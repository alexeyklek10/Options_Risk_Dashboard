"""Generate a realistic synthetic SPY-like chain for the methodology fixture.

Live yfinance was returning mostly stale rows (zero OI, IV = 1e-5) when this
script was first built, so the dashboard screenshots and notebook outputs
relied on a degenerate snapshot. This script produces a clean fixture that
exercises every analytic with plausible numbers.

The fixture is a SPY-like chain (spot ~595, six expirations spanning a week
to a year, a downward-skewed smile, OI concentrated near ATM and in the
monthly cycles) priced from first principles via Black-Scholes-Merton. The
schema matches :class:`ord.data.base.ChainRow`.

Re-run when the methodology fixture needs refreshing:

    .venv/Scripts/python.exe .build_fixture.py
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ord.data.base import CHAIN_COLUMNS
from ord.pricing.black_scholes import bs_price

SPOT = 595.0
SNAPSHOT_DATE = date(2026, 5, 13)
EXPIRY_DTES = [5, 12, 37, 95, 180, 365]
STRIKE_RANGE_PCT = (0.75, 1.25)  # 25% in either direction
STRIKE_STEP = 5.0
R = 0.045
Q = 0.013  # SPY trailing yield
RNG = np.random.default_rng(20260513)
FETCHED_AT = datetime(2026, 5, 13, 20, 30, tzinfo=timezone.utc)


def _iv_smile(strike: float, dte: int) -> float:
    """Realistic SPY-like smile: downward put skew, slight term-structure decay."""
    moneyness = np.log(strike / SPOT)
    # Steeper put skew than call skew.
    side_slope = -0.45 if moneyness < 0 else -0.15
    skew = side_slope * moneyness
    # Slight excess-kurtosis bump in the wings.
    bf = 0.18 * moneyness * moneyness
    # ATM term-structure: shorter expiry richer (event premium), gradual decay.
    atm = 0.18 + 0.06 * np.exp(-dte / 30.0) - 0.01 * np.log1p(dte / 30.0)
    return float(max(atm + skew + bf, 0.05))


def _open_interest(strike: float, dte: int, option_type: str) -> int:
    """Realistic OI per side: peaks at ATM, monthlies > weeklies, put/call asymmetric.

    Calls accumulate OI predominantly at the money and on OTM call strikes
    (covered-call writers, upside speculation). Puts accumulate on the
    downside (protective puts, put-spread collars). The asymmetry creates a
    net-negative-GEX bias below spot and net-positive above, mirroring SPY's
    typical positioning regime.
    """
    moneyness = np.log(strike / SPOT)
    base = 25000 if dte in (37, 95) else 8000  # monthly OI > weeklies
    envelope = np.exp(-(moneyness * moneyness) / (2.0 * 0.06 * 0.06))
    # Side-asymmetry: calls accumulate above spot, puts accumulate below spot.
    if option_type == "call":
        side_tilt = 1.0 + 1.5 * max(moneyness, 0.0) * 4.0  # boost OTM calls
        side_tilt -= 0.3 * max(-moneyness, 0.0) * 4.0  # suppress deep ITM calls
    else:
        side_tilt = 1.0 + 2.5 * max(-moneyness, 0.0) * 4.0  # boost OTM puts (protective)
        side_tilt -= 0.3 * max(moneyness, 0.0) * 4.0  # suppress deep ITM puts
    side_tilt = max(side_tilt, 0.2)
    noise = RNG.lognormal(mean=0.0, sigma=0.4)
    return int(max(int(base * envelope * side_tilt * noise), 0))


def _volume(oi: int) -> int:
    """Volume is a fraction of OI plus a small absolute floor for atm flows."""
    return int(max(RNG.normal(0.08, 0.03) * oi + RNG.normal(50, 30), 0))


def _bid_ask_spread_pct(strike: float) -> float:
    moneyness = abs(np.log(strike / SPOT))
    return 0.005 + 0.04 * moneyness  # 0.5% near ATM, ~2.5% deep OTM


def _build_chain() -> pd.DataFrame:
    strikes = np.arange(
        round(SPOT * STRIKE_RANGE_PCT[0] / STRIKE_STEP) * STRIKE_STEP,
        round(SPOT * STRIKE_RANGE_PCT[1] / STRIKE_STEP) * STRIKE_STEP + STRIKE_STEP,
        STRIKE_STEP,
    )
    rows = []
    for dte in EXPIRY_DTES:
        expiry = SNAPSHOT_DATE + timedelta(days=dte)
        T = dte / 365.0
        for strike in strikes:
            iv = _iv_smile(float(strike), dte)
            for option_type in ("call", "put"):
                oi = _open_interest(float(strike), dte, option_type)
                mid = bs_price(SPOT, float(strike), T, R, iv, option_type, Q)
                if mid < 0.01:
                    mid = 0.01  # nominal floor
                spread = mid * _bid_ask_spread_pct(float(strike))
                bid = max(mid - spread / 2.0, 0.0)
                ask = mid + spread / 2.0
                last = mid + RNG.normal(0.0, mid * 0.01)
                rows.append(
                    {
                        "ticker": "SPY",
                        "expiry": expiry,
                        "dte": dte,
                        "strike": float(strike),
                        "option_type": option_type,
                        "bid": round(bid, 2),
                        "ask": round(ask, 2),
                        "mid": round(mid, 4),
                        "last": round(last, 2),
                        "volume": _volume(oi),
                        "open_interest": oi,
                        "implied_vol": round(iv, 4),
                        "underlying_price": SPOT,
                        "fetched_at": FETCHED_AT,
                        "source": "yfinance",
                    }
                )
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


def main() -> None:
    chain = _build_chain()
    out = Path("notebooks/fixtures") / f"spy_chain_{SNAPSHOT_DATE.isoformat().replace('-', '_')}.parquet"
    chain.to_parquet(out, index=False)
    print(f"Wrote {out} -- {len(chain)} rows across {chain['expiry'].nunique()} expiries")
    print(f"  spot: ${SPOT:.2f}")
    print(f"  DTE: {sorted(chain['dte'].unique())}")
    print(f"  median IV: {chain['implied_vol'].median():.4f}")
    print(f"  total OI: {chain['open_interest'].sum():,}")


if __name__ == "__main__":
    main()
