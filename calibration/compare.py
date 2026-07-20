"""Phase 2 calibration: fit NB dispersion on settled data, then compare the
reliability of Poisson vs Negative-Binomial P(side) head-to-head.

    python compare.py [path-to-live_settled.csv]

Prints the fitted r, implied variance inflation, and a side-by-side reliability
table with the weighted mean |gap| for each model. Also sweeps a light lambda
shrink toward the slate mean to see if it helps further.
"""
from __future__ import annotations

import csv
import math
import sys

from nb import fit_dispersion, nb_p_more

DEFAULT = r"C:\strike-data\features\live_settled.csv"
LINES = [3.5, 4.5, 5.5, 6.5, 7.5]
BINS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
        (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]


def pois_p_more(mu, line):
    need = math.floor(line) + 1   # see markets.over_threshold (canonical)
    return 1.0 - sum(math.exp(-mu) * mu ** i / math.factorial(i) for i in range(need))


def samples(rows, p_more, shrink=1.0, grand=0.0):
    out = []
    for mu, act in rows:
        m = grand + shrink * (mu - grand) if shrink != 1.0 else mu
        for line in LINES:
            pm = p_more(m, line)
            out.append((pm, act > line))
            out.append((1 - pm, act < line))
    return out


def reliability(samps):
    table, gap_w, n_tot = [], 0.0, 0
    for lo, hi in BINS:
        grp = [(p, w) for p, w in samps if lo <= p < hi]
        if not grp:
            continue
        pred = sum(p for p, _ in grp) / len(grp)
        real = sum(1 for _, w in grp if w) / len(grp)
        gap = real - pred
        gap_w += abs(gap) * len(grp); n_tot += len(grp)
        table.append((lo, hi, len(grp), pred, real, gap))
    return table, gap_w / n_tot


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    rows = [(float(r["expected_ks"]), float(r["actual_ks"]))
            for r in csv.DictReader(open(path)) if r.get("expected_ks")]
    grand = sum(mu for mu, _ in rows) / len(rows)

    r = fit_dispersion(rows)
    infl = 1 + grand / r  # variance/mean at the average projection
    print(f"n={len(rows)} starts   grand mean lambda={grand:.2f}")
    print(f"fitted NB dispersion r = {r:.1f}  "
          f"(variance is ~{infl:.2f}x the Poisson at avg lambda; r->inf = Poisson)\n")

    def nb(mu, line): return nb_p_more(mu, line, r)

    variants = [
        ("Poisson", samples(rows, pois_p_more)),
        ("NegBinom", samples(rows, nb)),
        ("NB + 0.85 shrink", samples(rows, nb, shrink=0.85, grand=grand)),
    ]

    for name, samps in variants:
        table, mgap = reliability(samps)
        print(f"=== {name} ===   weighted mean |gap| = {mgap*100:.1f} pts")
        print(f"  {'bucket':>13}{'n':>6}{'pred':>8}{'real':>8}{'gap':>8}")
        for lo, hi, k, pred, real, gap in table:
            flag = " *" if abs(gap) > 0.04 else ""
            print(f"  {lo:.2f}-{hi:<7.2f}{k:>6}{pred*100:7.1f}%{real*100:7.1f}%{gap*100:+7.1f}%{flag}")
        print()

    print("Lower weighted |gap| = better calibrated. Wire the winning model's "
          "r into pick6/dispersion.py and switch sim.p_more to NB.")


if __name__ == "__main__":
    main()
