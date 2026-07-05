"""Calibration backtest: does the K model's stated P(side) match reality?

For Pick6 there is no market to bail you out — a leg's stated probability IS the
edge, so it must be *calibrated*: legs the model calls 60% must win ~60% of the
time. This replays settled starts, computes model P(More) at a synthetic line,
and bins predicted-vs-realized to expose over/under-confidence.

Input: live_settled.csv (date,pitcher,expected_ks,actual_ks,started) from
strike/mlb-edge. Uses each start's own expected_ks as lambda and grades against
actual_ks at a line = round(actual to nearest .5 neighbours) sweep.

    python backtest.py [path-to-live_settled.csv]

Reliability output tells you whether to (a) trust the Poisson probs, or (b)
switch to Negative-Binomial / shrink lambda before building Pick6 entries.
"""
from __future__ import annotations

import csv
import math
import sys

DEFAULT = r"C:\strike-data\features\live_settled.csv"


def pois_pmf(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def p_more(lam, line):
    need = math.ceil(line)
    return 1.0 - sum(pois_pmf(i, lam) for i in range(need))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    rows = [r for r in csv.DictReader(open(path)) if r.get("expected_ks")]

    # Build (predicted_prob, hit) pairs by scoring BOTH sides at every half line
    # that sits between plausible outcomes (line = 3.5 .. 8.5).
    samples = []  # (p_side, won)
    signed_err = []  # actual - expected (bias)
    abs_err = []
    for r in rows:
        lam = float(r["expected_ks"]); act = float(r["actual_ks"])
        signed_err.append(act - lam); abs_err.append(abs(act - lam))
        for line in [3.5, 4.5, 5.5, 6.5, 7.5]:
            pm = p_more(lam, line)
            samples.append((pm, act > line))            # More side
            samples.append((1 - pm, act < line))         # Less side

    n = len(rows)
    print(f"n={n} settled starts   MAE={sum(abs_err)/n:.2f} K   "
          f"bias(actual-exp)={sum(signed_err)/n:+.2f} K")
    print("(negative bias => model OVER-projects strikeouts)\n")

    # Reliability table: bin predicted prob, compare to realized win rate.
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
            (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    print(f"  {'pred bucket':>14}{'n':>7}{'predicted':>11}{'realized':>10}{'gap':>8}")
    total_gap_w = 0.0; total_n = 0
    for lo, hi in bins:
        grp = [(p, w) for p, w in samples if lo <= p < hi]
        if not grp:
            continue
        pred = sum(p for p, _ in grp) / len(grp)
        real = sum(1 for _, w in grp if w) / len(grp)
        gap = real - pred
        total_gap_w += abs(gap) * len(grp); total_n += len(grp)
        flag = "  <-- overconfident" if gap < -0.04 else ("  <-- underconfident" if gap > 0.04 else "")
        print(f"  {lo:.2f}-{hi:<8.2f}{len(grp):>7}{pred*100:10.1f}%{real*100:9.1f}%"
              f"{gap*100:+7.1f}%{flag}")
    print(f"\n  weighted mean |gap| = {total_gap_w/total_n*100:.1f} pts "
          f"(want < ~3 pts before trusting Pick6 breakevens)")
    print("\nIf high-prob buckets read 'overconfident', Poisson under-disperses "
          "the K distribution — switch p_more() to Negative-Binomial (fit the\n"
          "overdispersion on this same file) and/or shrink lambda toward the "
          "slate mean, then re-run.")


if __name__ == "__main__":
    main()
