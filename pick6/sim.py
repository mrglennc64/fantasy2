"""Pick6 entry builder + exact outcome-potential matrix (PAPER ONLY).

Adapted from the strike/mlb-edge parlay simulator, with the sportsbook odds path
removed: Pick6 has no vig, so a leg is just P(chosen side beats DK's line) and an
entry is an all-or-nothing power play paying config.entry_multiplier().

Pipeline:
  1  score each board leg: model P(More) and P(Less) from the projection (lambda)
  2  pick the side with the higher model prob; keep legs that clear breakeven+margin
  3  build n-pick entries (combinations), never two pitchers from the same game
  4  rank by EV; size with fractional-Kelly capped to a daily budget
  5  exact P&L distribution over every leg win/loss combo (handles shared legs)

*** NOT BETTING ADVICE. The single-leg K model is uncalibrated / over-projects
until the calibration backtest proves otherwise — treat EV as illustrative. ***
"""
from __future__ import annotations

import math
from itertools import combinations

from config import breakeven_per_leg, entry_ev, entry_multiplier
from dispersion import DISPERSION_R
from markets import p_cap, p_over

# ---- step 1: side probabilities from a calibrated Negative-Binomial ----------
# Phase 2 (calibration/compare.py) showed Poisson under-disperses strikeouts and
# is over-confident in the 60-65% band Pick6 legs live in. NB with the fitted
# dispersion (pick6/dispersion.py) cut the mean reliability gap from 3.2 -> 1.6.

_EPS = 1e-9


def nb_pmf(k: int, mu: float, r: float = DISPERSION_R) -> float:
    mu = max(mu, _EPS)
    logp = (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
            + r * math.log(r / (r + mu)) + k * math.log(mu / (r + mu)))
    return math.exp(logp)


def pois_pmf(k: int, lam: float) -> float:  # kept for reference / backtests
    return math.exp(-lam) * lam ** k / math.factorial(k)


def p_more(lam: float, line: float, r: float = DISPERSION_R) -> float:
    """P(strikeouts > line) for a half-integer line under NB(mean=lam, size=r).

    More wins on K >= ceil(line); e.g. line 5.5 -> More needs K >= 6.
    """
    need = math.ceil(line)
    return max(0.0, 1.0 - sum(nb_pmf(i, lam, r) for i in range(need)))


def score_leg(leg: dict) -> dict:
    """leg in: name, game, line, lam, market, [more_boost]. Chooses side + prob
    using the market's distribution (markets.p_over); defaults to strikeouts."""
    market = leg.get("market", "strikeouts")
    pm = p_over(market, leg["lam"], leg["line"])
    side, p = ("more", pm) if pm >= 0.5 else ("less", 1.0 - pm)
    cap = p_cap(market)
    if cap is not None:
        p = min(p, cap)  # baseline markets: no fitted dispersion, cap the claim
    boost = leg.get("more_boost", 1.0) if side == "more" else 1.0
    return {**leg, "side": side, "p": p, "boost": boost}


# ---- step 2: keep confident legs --------------------------------------------

def _side_available(leg: dict) -> bool:
    """DK sometimes offers only More (or only Less) on a leg. If availability
    flags are present, the chosen side must actually be offered."""
    if leg["side"] == "more":
        return leg.get("more_available", True)
    return leg.get("less_available", True)


def rank_legs(legs: list[dict], n_picks: int, margin: float = 0.05,
              platform: str = "dk_pick6") -> list[dict]:
    """Keep legs whose model prob clears the n-pick breakeven (for this platform)
    by `margin` AND whose model-chosen side is actually offered on the board."""
    be = breakeven_per_leg(n_picks, platform=platform)
    scored = [score_leg(l) for l in legs if _side_available(score_leg(l))]
    keep = [l for l in scored if l["p"] >= be + margin]
    return sorted(keep, key=lambda l: l["p"], reverse=True)


# ---- step 3/4: build + size entries -----------------------------------------

def build_entries(legs: list[dict], n_picks: int, max_entries: int,
                  platform: str = "dk_pick6") -> list[dict]:
    entries = []
    for combo in combinations(legs, n_picks):
        if len({l["game"] for l in combo}) < n_picks:
            continue  # never two picks from the same game in one entry
        boosts = [l["boost"] for l in combo]
        probs = [l["p"] for l in combo]
        m = entry_multiplier(n_picks, boosts, platform)
        p_all = math.prod(probs)
        entries.append({"legs": combo, "p": p_all, "mult": m, "n": n_picks,
                        "platform": platform, "ev": entry_ev(probs, boosts, platform),
                        "kelly": _kelly(p_all, m)})
    entries.sort(key=lambda e: e["ev"], reverse=True)
    return entries[:max_entries]


def _kelly(p: float, mult: float) -> float:
    b = mult - 1.0
    return max(0.0, (p * b - (1 - p)) / b)


def allocate(entries, bankroll, daily_frac, kelly_frac, per_cap_frac):
    raw = [min(kelly_frac * e["kelly"], per_cap_frac) * bankroll for e in entries]
    daily_cap = bankroll * daily_frac
    total = sum(raw)
    scale = daily_cap / total if total > daily_cap and total > 0 else 1.0
    for e, s in zip(entries, raw):
        e["stake"] = round(s * scale, 2)
    return entries, daily_cap, scale


# ---- step 5: exact outcome matrix (shared legs handled) ----------------------

def outcome_matrix(entries):
    leg_ids = sorted({id(l) for e in entries for l in e["legs"]}, key=lambda x: x)
    idx = {lid: i for i, lid in enumerate(leg_ids)}
    leg_p = {}
    for e in entries:
        for l in e["legs"]:
            leg_p[idx[id(l)]] = l["p"]
    L = len(leg_ids)
    dist = []
    for mask in range(1 << L):
        prob = 1.0
        won = [False] * L
        for i in range(L):
            hit = (mask >> i) & 1
            won[i] = bool(hit)
            prob *= leg_p[i] if hit else (1 - leg_p[i])
        pnl, nwon = 0.0, 0
        for e in entries:
            if all(won[idx[id(l)]] for l in e["legs"]):
                pnl += e["stake"] * (e["mult"] - 1)
                nwon += 1
            else:
                pnl -= e["stake"]
        dist.append((prob, pnl, nwon))
    ev = sum(p * pnl for p, pnl, _ in dist)
    var = sum(p * (pnl - ev) ** 2 for p, pnl, _ in dist)
    p_profit = sum(p for p, pnl, _ in dist if pnl > 0)
    staked = sum(e["stake"] for e in entries)
    by_n = {}
    for p, pnl, nwon in dist:
        a = by_n.setdefault(nwon, [0.0, 0.0])
        a[0] += p
        a[1] += p * pnl
    return {"ev": ev, "sd": var ** 0.5, "p_profit": p_profit, "staked": staked,
            "best": max(dist, key=lambda t: t[1]), "worst": min(dist, key=lambda t: t[1]),
            "by_n": by_n}
