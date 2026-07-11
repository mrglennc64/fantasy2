"""Probability calibration fitted on the model's OWN graded history.

    p' = sigmoid(alpha + beta * logit(p))     per group (pitcher / batter)

This layer replaces what the old line-anchor did for honesty, without
touching the market: alpha/beta are fitted on (stated p, outcome) pairs from
the frozen record — the line never enters the fit. beta < 1 shrinks
systematic overconfidence; beta = 0 is the degenerate-but-honest statement
that the confidence RANKING has not yet demonstrated skill, so every row
states the group's realized base rate. beta is monotone-constrained (>= 0)
so a calibrated probability can never contradict the displayed lean.

Provenance (fitted 2026-07-11, walk-forward on 119 frozen-mean pitcher rows):
    pitcher: alpha=+0.120, beta=0.00 -> flat ~53.0% (realized 52.9%)
    batter:  identity (alpha=0, beta=1) — mean fixes of 7/9 still settling;
             the 0.70 ceiling in markets.py remains its guard until this
             layer is fitted per-market on the growing graded log.
Re-fit with the weekly cron report as the record grows; raise beta only when
the walk-forward supports it.
"""
from __future__ import annotations

import math

# group -> (alpha, beta), beta >= 0
GROUPS = {
    "pitcher": (0.120, 0.00),
    "batter": (0.0, 1.0),
}


def calibrate(group: str, p: float) -> float:
    """Calibrated probability for the model's chosen side. Never dips below
    50% (the lean itself is the mean's call; this layer only sizes trust)."""
    a, b = GROUPS.get(group, (0.0, 1.0))
    p = min(max(p, 0.5 + 1e-9), 1.0 - 1e-9)
    z = a + b * math.log(p / (1.0 - p))
    return max(0.5, 1.0 / (1.0 + math.exp(-z)))
