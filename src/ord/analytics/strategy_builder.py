"""Multi-leg strategy P&L surfaces.

User defines up to four legs as ``(option_type, strike, expiry, signed_quantity)``.
Generates: P&L at expiry across a spot range, P&L surface across (spot, time),
breakevens, max profit / max loss, current position Greeks. Preset templates
include long call, long put, vertical, iron condor, butterfly, calendar,
straddle, strangle.

Implemented in step 6 of the build sequence.
"""

from __future__ import annotations
