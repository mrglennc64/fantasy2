"""Market registry: per-prop distribution + dispersion so the same machinery
scores strikeouts, hits, total bases, home runs, etc. — not just Ks.

Each market maps a projection (mean mu) to P(Over line). Strikeouts use the
Negative-Binomial with the dispersion fitted in Phase 2. Other markets are
scaffolded with a documented distribution; their dispersion still needs fitting
on real data (calibration/fit_market.py, TODO) and — crucially — a PROJECTION
SOURCE, since strike/mlb-edge only projects strikeouts. Until a market has both
a fitted dispersion and a mu source, it is declared but not production-ready.
"""
from __future__ import annotations

import math

from dispersion import DISPERSION_R

_EPS = 1e-9


def _nb_pmf(k, mu, r):
    mu = max(mu, _EPS)
    return math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                    + r * math.log(r / (r + mu)) + k * math.log(mu / (r + mu)))


def _pois_pmf(k, mu):
    mu = max(mu, _EPS)
    return math.exp(-mu) * mu ** k / math.factorial(k)


# dist: "nb" needs r; "poisson" for rare counts.
# confidence: "calibrated" (dispersion fitted on settled data) vs "baseline"
# (StatsAPI season-rate projection, matchup-neutral, dispersion NOT yet fitted).
# side: "pitcher" or "batter" — drives the dashboard toggle.
MARKETS = {
    "strikeouts":   {"dist": "nb", "r": DISPERSION_R, "ready": True, "side": "pitcher",
                     "confidence": "calibrated", "mu_source": "mlb-edge /v2/slate"},
    "hits":         {"dist": "poisson", "ready": True, "side": "batter",
                     "confidence": "baseline", "mu_source": "StatsAPI season H/AB"},
    "total_bases":  {"dist": "nb", "r": 4.0, "ready": True, "side": "batter",
                     "confidence": "baseline", "mu_source": "StatsAPI season TB/AB"},
    "home_runs":    {"dist": "poisson", "ready": True, "side": "batter",
                     "confidence": "baseline", "mu_source": "StatsAPI season HR/AB"},
    "rbi":          {"dist": "poisson", "ready": True, "side": "batter",
                     "confidence": "baseline", "mu_source": "StatsAPI season RBI/PA"},
    "runs":         {"dist": "poisson", "ready": True, "side": "batter",
                     "confidence": "baseline", "mu_source": "StatsAPI season R/PA"},
}


def is_ready(market: str) -> bool:
    return MARKETS.get(market, {}).get("ready", False)


def market_side(market: str) -> str:
    return MARKETS.get(market, {}).get("side", "pitcher")


def confidence(market: str) -> str:
    return MARKETS.get(market, {}).get("confidence", "baseline")


# Baseline markets have NO fitted dispersion, so their tail probabilities are
# not trustworthy — a .380-SLG regular vs a 0.5 TB line prices at 85%+ on season
# rate alone. Cap what a baseline leg may claim so it can never outrank a
# calibrated leg on manufactured confidence (7/7: both entries anchored on an
# 86% baseline batter leg). Remove per-market once its dispersion is fitted.
BASELINE_P_CAP = 0.70


def p_cap(market: str) -> float | None:
    """Max model P a leg in this market may claim, or None (uncapped)."""
    return BASELINE_P_CAP if confidence(market) == "baseline" else None


def p_over(market: str, mu: float, line: float) -> float:
    """P(stat > line) for a half-integer line, per the market's distribution.

    Over wins on stat >= ceil(line); e.g. line 5.5 -> Over needs >= 6.
    """
    spec = MARKETS.get(market)
    if spec is None:
        raise ValueError(f"unknown market {market!r}")
    need = math.ceil(line)
    if spec["dist"] == "nb":
        cdf = sum(_nb_pmf(i, mu, spec["r"]) for i in range(need))
    else:
        cdf = sum(_pois_pmf(i, mu) for i in range(need))
    return max(0.0, 1.0 - cdf)
