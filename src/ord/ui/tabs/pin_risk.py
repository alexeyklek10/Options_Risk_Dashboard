"""Pin Risk tab: pin-likelihood scores + cumulative gamma vs strike.

Implementation lands in step 6 of the build sequence.
"""

from __future__ import annotations

import streamlit as st

from ord.ui.context import AppContext


def render(_ctx: AppContext) -> None:
    st.info("Pin-risk view lands in step 6.")
