"""Gamma exposure (GEX) using the SqueezeMetrics dealer-sign convention.

Sign convention (positive GEX = long-gamma regime, dampened realized vol;
negative GEX = short-gamma, amplified realized vol) is itself a methodology
choice. The convention and its assumptions are spelled out in the methodology
notebook and in this module's docstring at implementation time.

Outputs per-strike dollar GEX, the cumulative GEX curve, and the gamma flip
level (the strike at which cumulative GEX crosses zero walking down from the
highest strike).

Implemented in step 6 of the build sequence.
"""

from __future__ import annotations
