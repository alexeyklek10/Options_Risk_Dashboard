"""Implied PDF tab: Breeden-Litzenberger density per expiry.

Implementation lands in step 6 of the build sequence.
"""

from __future__ import annotations

import streamlit as st

from ord.ui.context import AppContext


def render(_ctx: AppContext) -> None:
    st.info("Implied PDF (Breeden-Litzenberger) view lands in step 6.")
