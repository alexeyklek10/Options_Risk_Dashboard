"""Strategy Builder tab: up to 4 legs, expiry P&L, P&L surface, position Greeks.

Implementation lands in step 6 of the build sequence.
"""

from __future__ import annotations

import streamlit as st

from ord.ui.context import AppContext


def render(_ctx: AppContext) -> None:
    st.info("Strategy builder view lands in step 6.")
