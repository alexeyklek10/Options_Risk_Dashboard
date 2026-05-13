"""Expected-move cone derived from ATM straddle mid prices.

Per expiry, the expected move is approximated as::

    expected_move ~= 0.85 * ATM_straddle_mid

The 0.85 factor is the standard rule-of-thumb scaling that maps the
ATM-straddle-implied 1-sigma range to a robust point estimate; see the
methodology notebook for the derivation.

The ATM strike is the listed strike nearest to spot; if call or put mid is
missing at that strike the expiry is skipped (returned as ``NaN``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_SCALING: float = 0.85


@dataclass(frozen=True)
class ExpectedMoveResult:
    """Expected-move table per expiry."""

    per_expiry: pd.DataFrame  # columns: expiry, atm_strike, straddle_mid, expected_move
    scaling: float


def expected_move(chain: pd.DataFrame, scaling: float = DEFAULT_SCALING) -> ExpectedMoveResult:
    if chain.empty:
        return ExpectedMoveResult(
            per_expiry=pd.DataFrame(
                columns=["expiry", "atm_strike", "straddle_mid", "expected_move"]
            ),
            scaling=scaling,
        )

    spot = float(chain["underlying_price"].iloc[0])
    rows: list[dict[str, object]] = []
    for expiry, exp_df in chain.groupby("expiry"):
        strikes = exp_df["strike"].unique()
        atm = float(strikes[int(np.argmin(np.abs(strikes - spot)))])
        atm_df = exp_df[exp_df["strike"] == atm]
        call = atm_df[atm_df["option_type"] == "call"]
        put = atm_df[atm_df["option_type"] == "put"]
        if call.empty or put.empty:
            rows.append(
                {
                    "expiry": expiry,
                    "atm_strike": atm,
                    "straddle_mid": float("nan"),
                    "expected_move": float("nan"),
                }
            )
            continue
        call_mid = call["mid"].iloc[0]
        put_mid = put["mid"].iloc[0]
        if pd.isna(call_mid) or pd.isna(put_mid):
            rows.append(
                {
                    "expiry": expiry,
                    "atm_strike": atm,
                    "straddle_mid": float("nan"),
                    "expected_move": float("nan"),
                }
            )
            continue
        straddle = float(call_mid) + float(put_mid)
        rows.append(
            {
                "expiry": expiry,
                "atm_strike": atm,
                "straddle_mid": straddle,
                "expected_move": scaling * straddle,
            }
        )
    return ExpectedMoveResult(per_expiry=pd.DataFrame(rows), scaling=scaling)
