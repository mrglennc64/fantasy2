"""Simulate a DK Pick6 line from a sportsbook line.

No historical Pick6 boards exist, so a Pick6 backtest must either use the
sportsbook line as-is or SIMULATE the Pick6 line. Measured on the one day we
have both (2026-07-05, 12 pitchers): Pick6 - sportsbook delta had mean +0.08 K,
sd 0.28 — 11/12 identical, 1 off by a full run. So sportsbook is a close proxy,
but the occasional ±1.0 flips a leg's edge, which is exactly the risk to model.

This draws a DETERMINISTIC per-(pitcher,date) jitter from that distribution
(hash-seeded so backtests are reproducible), snaps to the nearest 0.5, and
exposes a `softness` knob for sensitivity analysis (how does ROI move if Pick6
lines run systematically softer/tighter than sportsbook?).

*** These are SIMULATED lines. Results are sensitivity analysis, not evidence.
The only real Pick6 numbers come from capturing live boards. Re-fit MEAS_* as
more real boards accrue (compare data/pick6_board_*.csv to that day's slate). ***
"""
from __future__ import annotations

import hashlib
import math

MEAS_MEAN = 0.08   # measured Pick6 - sportsbook mean delta (n=12, 2026-07-05)
MEAS_SD = 0.28     # measured sd


def _det_normal(name: str, date: str) -> float:
    """Deterministic standard-normal from a hash of (name, date) via Box-Muller."""
    h = hashlib.md5(f"{name}|{date}".encode()).hexdigest()
    u1 = (int(h[:8], 16) + 1) / (16 ** 8 + 1)
    u2 = (int(h[8:16], 16) + 1) / (16 ** 8 + 1)
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def sim_pick6_line(book_line: float, name: str, date: str,
                   softness: float = 0.0, sd: float = MEAS_SD) -> float:
    """Simulated DK Pick6 line = book line + measured jitter, snapped to 0.5.

    softness shifts the whole distribution (e.g. -0.5 = Pick6 sets lines half a K
    below sportsbook, making Overs easier); default uses the measured mean.
    """
    delta = MEAS_MEAN + softness + _det_normal(name, date) * sd
    return round((book_line + delta) * 2) / 2
