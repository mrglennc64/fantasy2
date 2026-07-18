"""Leg scoring: numeric predictions for every board row.

For each (player, market, line) the scorer emits real numbers, always:
  predicted   the raw model projection (mean of the market's distribution)
  p_more      P(stat > line) under the market's distribution
  p_less      1 - p_more
  side        which side of the line the model leans ("more"/"less")
  p           the leaned side's probability (confidence)

The probability pipeline is LINE-FREE end to end: an affine-recentered mean
(pick6/projection.py, fitted on frozen projection/actual pairs), the market's
distribution (NB with dispersion fitted on settled data for strikeouts), and
a final probability calibration fitted on the model's own graded history
(pick6/calibrate.py). The published line appears only as the threshold the
probability is about and as a reference column — never inside the engine.

There is no qualification threshold here: every row gets scored and reported.
Rankings order by confidence (distance from 50%); they never suppress output.
"""
from __future__ import annotations

import math

from calibrate import calibrate
from dispersion import DISPERSION_R
from markets import market_side as market_group, p_cap, p_over
from projection import corrected_mu

_EPS = 1e-9


def nb_pmf(k: int, mu: float, r: float = DISPERSION_R) -> float:
    mu = max(mu, _EPS)
    logp = (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
            + r * math.log(r / (r + mu)) + k * math.log(mu / (r + mu)))
    return math.exp(logp)


def p_more(lam: float, line: float, r: float = DISPERSION_R) -> float:
    """P(strikeouts > line) for a half-integer line under NB(mean=lam, size=r).

    More wins on K >= ceil(line); e.g. line 5.5 -> More needs K >= 6.
    """
    need = math.ceil(line)
    return max(0.0, 1.0 - sum(nb_pmf(i, lam, r) for i in range(need)))


def score_leg(leg: dict) -> dict:
    """leg in: name, game, line, lam, market. Out: leg + predicted / p_more /
    p_less / side / p — real numbers for every row, no exceptions."""
    market = leg.get("market", "strikeouts")
    # Two probability tracks, both logged and graded (live A/B on the record):
    #   raw        — straight from the projection, no correction, no ceiling.
    #   calibrated — LINE-FREE pipeline (2026-07-11): affine-recentered mean
    #                (projection.py) -> market distribution -> probability
    #                calibration fitted on our own graded history
    #                (calibrate.py). The line is only the threshold the
    #                probability is about — never the mean or an anchor.
    # The displayed point prediction is always the raw projection.
    pm_raw = p_over(market, leg["lam"], leg["line"])
    mu = corrected_mu(market, leg["lam"], leg.get("mu_source", "unknown"))
    pm = p_over(market, mu, leg["line"])
    side, p = ("more", pm) if pm >= 0.5 else ("less", 1.0 - pm)
    # p_uncal is the probability calibrate() is ABOUT to be applied to. Logging
    # it is what lets refit_calibration.py fit on the same quantity it will be
    # applied to: it previously fitted on p_more_raw (from the UNcorrected mu)
    # while production applied the result here, to the corrected one. beta=0
    # hid that mismatch — a constant ignores its input — and it would have
    # surfaced as systematic miscalibration the first time beta moved.
    p_uncal = p
    cap = p_cap(market)
    if cap is not None and p > cap:
        p = cap                      # un-fitted dispersion: cap the confidence
    p = calibrate(market_group(market), p)
    pm = p if side == "more" else 1.0 - p
    return {**leg, "predicted": leg["lam"], "p_more": pm, "p_less": 1.0 - pm,
            "side": side, "p": p, "p_more_raw": pm_raw, "p_uncal": p_uncal}


def rank_by_confidence(legs: list[dict]) -> list[dict]:
    """All legs scored, ordered by confidence. Nothing is dropped."""
    return sorted((score_leg(l) for l in legs), key=lambda l: l["p"], reverse=True)
