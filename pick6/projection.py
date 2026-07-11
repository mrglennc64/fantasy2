"""Line-free mean correction for the strikeout projection.

ARCHITECTURAL RULE (2026-07-11): the market can be a benchmark; it cannot be
the model's brain. The published line must never be used as:
  - expected mean
  - model input feature
  - calibration target
  - probability anchor
  - regression baseline
It appears exactly twice downstream, both legitimate: as the threshold the
probability is ABOUT (P(actual > line) is the question being answered), and
as a reference column next to the prediction on the dashboard.

The correction here is a global affine recentering fitted ONLY on frozen
(projection, actual) pairs — no line anywhere in the fit:

    mu' = AFFINE_A + AFFINE_B * mu_raw

Provenance (refit 2026-07-11 on 190 frozen pre-game starts, June mlb-edge
log + frozen slate archives, settled vs StatsAPI boxscores):
    full-sample NB MLE:  mu' = +2.25 + 0.50*mu
    walk-forward:        stated 61.1%  realized 52.9%  (raw was 50.4%)
The affine correction fixes the mean bias (raw over-projects, worst at high
mu); the remaining stated-vs-realized gap is handled downstream by
pick6/calibrate.py, which is fitted on the model's own graded history —
also line-free. Both re-fit as frozen data accumulates (weekly cron report).
"""
from __future__ import annotations

AFFINE_A = 2.25
AFFINE_B = 0.50


def corrected_mu(market: str, mu: float) -> float:
    """Independent mean used for probability computation. Strikeouts get the
    affine recentering; other markets pass through (their means are already
    built line-free in batter_feed.py)."""
    if market == "strikeouts":
        return AFFINE_A + AFFINE_B * mu
    return mu
