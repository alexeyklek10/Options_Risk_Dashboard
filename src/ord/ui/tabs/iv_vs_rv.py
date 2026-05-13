"""IV vs RV tab: spread time series if cached, current snapshot otherwise.

Implementation lands in step 6 of the build sequence.
"""

from __future__ import annotations

import streamlit as st

from ord.ui.context import AppContext


def render(_ctx: AppContext) -> None:
    st.info("IV vs RV view lands in step 6 of the build sequence.")
