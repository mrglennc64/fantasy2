"""Mean correction for the strikeout projection: shrink mu toward the line.

Provenance (calibration/fit_mean.py, run 2026-07-08 on 164 FROZEN bet-time
projections, 6/7-6/13 mlb-edge logs + 7/5 entry legs, settled vs StatsAPI):

  mean bias (actual - mu): -0.36 K, and it grows with mu:
    low-mu  (~3.8): +0.35 K   mid-mu (~5.0): -0.45 K   high-mu (~6.3): -0.97 K
  walk-forward, model-chosen side:
    raw     predicted 61.3%  realized 50.4%  gap -10.8 pts  (coin flips!)
    affine  predicted 61.3%  realized 53.7%  gap  -7.5 pts
    anchor  predicted 54.2%  realized 52.9%  gap  -1.3 pts  <- winner
  confident legs (p>=0.65, what the picker stakes): raw realized 55.9% vs
  predicted 73.4% — the exact live failure (7/5: -13.5 pts; 7/7: -25 pts).

The anchor MLE is s = 0.00: on this sample the model's disagreement with the
pick'em line carries NO information — the line alone predicts better. With
s=0 every K leg prices at ~the line (~50-55%), nothing clears breakeven, and
the picker correctly refuses to bet. That is the fix: no demonstrated edge =>
no entries, instead of confidently-wrong entries at -100% ROI.

To EARN s back up: archive_slate.py freezes every day's lambdas; re-run
calibration/fit_mean.py as frozen days accumulate and raise s only when the
walk-forward supports it. Never fit on /v2/slate re-projections (leakage —
see pick6/dispersion.py).
"""
from __future__ import annotations

SHRINK_TO_LINE_S = 0.00


def corrected_mu(market: str, mu: float, line: float | None) -> float:
    """Bet-time mean for scoring a leg. Strikeouts shrink toward the line by
    the fitted s; other markets pass through (their guard is markets.p_cap)."""
    if market == "strikeouts" and line is not None:
        return line + SHRINK_TO_LINE_S * (mu - line)
    return mu
