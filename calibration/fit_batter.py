"""Fit batter-market dispersions on the frozen prediction log.

    python fit_batter.py

Reads data/predictions_log.csv (graded batter rows: frozen generation-time
mean `predicted`, the published line, the model's side, and the actual).
For each market, fits the Negative-Binomial size r by MLE on (mu, actual)
pairs, then validates WALK-FORWARD by date: fit on earlier days only, score
the held-out day's logged side under (a) the production distribution
(markets.py as deployed) and (b) the re-fit — stated vs realized plus
log-loss. Recommends per-market r only when the held-out comparison
supports it; paste winners into markets.MARKETS with provenance.

Runs fine on the VPS from the hourly cron (no network needed — everything
comes from the log); the cron publishes the output as a static report.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pick6"))
from markets import MARKETS, over_threshold, p_over  # noqa: E402

LOG = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_log.csv")
MIN_TRAIN = 60
_EPS = 1e-9

R_GRID = [0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0, 150.0, 500.0]


def nb_logpmf(k: int, mu: float, r: float) -> float:
    mu = max(mu, _EPS)
    return (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
            + r * math.log(r / (r + mu)) + k * math.log(mu / (r + mu)))


def fit_r(pairs: list[tuple[float, int]]) -> float:
    best_r, best_ll = R_GRID[0], -1e18
    for r in R_GRID:
        ll = sum(nb_logpmf(a, mu, r) for mu, a in pairs)
        if ll > best_ll:
            best_ll, best_r = ll, r
    # one refinement pass around the grid winner
    for r in [best_r * f for f in (0.6, 0.75, 0.9, 1.1, 1.3, 1.6)]:
        ll = sum(nb_logpmf(a, mu, r) for mu, a in pairs)
        if ll > best_ll:
            best_ll, best_r = ll, r
    return best_r


def p_more_nb(mu: float, line: float, r: float) -> float:
    need = over_threshold(line)
    cdf = sum(math.exp(nb_logpmf(i, mu, r)) for i in range(need))
    return max(0.0, min(1.0, 1.0 - cdf))


def side_p(mu: float, line: float, side: str, r: float | None, market: str) -> float:
    """Stated probability of the logged side under candidate r (None =
    production distribution from markets.py)."""
    pm = p_over(market, mu, line) if r is None else p_more_nb(mu, line, r)
    return pm if side == "more" else 1.0 - pm


def main() -> None:
    if not os.path.exists(LOG):
        print(f"no log at {LOG}")
        return
    rows = [r for r in csv.DictReader(open(LOG, encoding="utf-8"))
            if (r.get("market") or "strikeouts") != "strikeouts"
            and r.get("actual") not in (None, "")]
    by_market: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_market[r["market"]].append(r)

    print(f"BATTER DISPERSION FIT — {len(rows)} settled rows in the frozen log\n")
    for market, mrows in sorted(by_market.items()):
        pairs = [(float(r["predicted"]), int(float(r["actual"]))) for r in mrows]
        spec = MARKETS.get(market, {})
        prod = f"NB r={spec.get('r')}" if spec.get("dist") == "nb" else "Poisson"
        r_full = fit_r(pairs)
        bias = sum(a - mu for mu, a in pairs) / len(pairs)
        print(f"{market}: n={len(pairs)}  production={prod}  "
              f"full-sample MLE r={r_full:.2f}  mean bias {bias:+.2f}")

        # walk-forward: production vs re-fit-as-you-go
        by_day = defaultdict(list)
        for r in mrows:
            by_day[r["date"]].append(r)
        dates = sorted(by_day)
        held = {"production": [], "refit": []}
        seen: list[tuple[float, int]] = []
        for d in dates:
            if len(seen) >= MIN_TRAIN:
                r_wf = fit_r(seen)
                for row in by_day[d]:
                    if row.get("result") not in ("1", "0"):
                        continue
                    mu, line = float(row["predicted"]), float(row["line"])
                    won = row["result"] == "1"
                    for tag, rr in (("production", None), ("refit", r_wf)):
                        p = side_p(mu, line, row["side"], rr, market)
                        held[tag].append((p, won))
            seen.extend((float(r["predicted"]), int(float(r["actual"])))
                        for r in by_day[d])
        for tag, legs in held.items():
            if not legs:
                print(f"  {tag:10} (not enough earlier days to train yet)")
                continue
            n = len(legs)
            pred = sum(p for p, _ in legs) / n
            real = sum(1 for _, w in legs if w) / n
            ll = -sum(math.log(max(p if w else 1 - p, _EPS)) for p, w in legs) / n
            print(f"  {tag:10} n={n:<4} stated {pred*100:5.1f}%  realized "
                  f"{real*100:5.1f}%  gap {(real-pred)*100:+5.1f}  log-loss {ll:.4f}")
        print()
    print("=> adopt a re-fit r in markets.MARKETS only where 'refit' beats"
          "\n   'production' on held-out gap AND log-loss. Small n first days —"
          "\n   expect the verdict to firm up as the log grows.")


if __name__ == "__main__":
    main()
