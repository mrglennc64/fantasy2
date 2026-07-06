"""Walk-forward dispersion validation — LINE-INDEPENDENT, so it's valid whether
you bet sportsbook or Pick6. It never touches a betting line: it only asks
whether actual strikeouts scatter around the projection the way the model's
Negative-Binomial assumes.

Method: sort settled starts by date; for each test day, MLE-fit the NB dispersion
r on all EARLIER days only, then on the held-out day compute the mid-PIT
(F(actual-1) + .5*pmf(actual)) of each start. Under a calibrated model the PITs
are uniform, so ~50% land in the central [.25,.75] band and ~80% in [.10,.90].
If far fewer do, actual K is MORE spread than the model thinks -> overconfident
at every line. Compared against the in-sample (fit-and-test-on-all) baseline to
expose overfit.

Data: live_settled.csv (current model, 6/28-7/3) + logged predictions x game
logs (early June — OLDER model version; version-mixing is a caveat, flagged).

    python walk_forward.py
"""
from __future__ import annotations

import csv
import os
import unicodedata
from collections import defaultdict

from nb import fit_dispersion, nb_pmf

SETTLED = r"C:\strike-data\features\live_settled.csv"
MLB = r"C:\Users\carin\OneDrive\Dokument\stike\mlb-edge\data"
PRED = os.path.join(MLB, "exports", "vps", "predictions.csv")
LOGS = os.path.join(MLB, "all_starters_gamelogs_2024_2026.csv")
MIN_TRAIN_STARTS = 40


def norm(n):
    n = unicodedata.normalize("NFKD", n)
    n = "".join(c for c in n if not unicodedata.combining(c))
    return "".join(c for c in n.lower() if c.isalpha() or c == " ").strip()


def load_pairs():
    """[(date, mu, actual, source)] with no line data at all."""
    pairs = []
    for r in csv.DictReader(open(SETTLED, encoding="utf-8")):
        if r.get("expected_ks"):
            pairs.append((r["date"], float(r["expected_ks"]), int(float(r["actual_ks"])), "current"))
    act = {}
    for r in csv.DictReader(open(LOGS, encoding="utf-8")):
        try:
            act[(r["date"], norm(r["pitcher"]))] = int(float(r["K"]))
        except (ValueError, KeyError):
            pass
    seen = set()
    for r in csv.DictReader(open(PRED, encoding="utf-8")):
        if not r.get("expected_ks"):
            continue
        key = (r["date"], norm(r["pitcher"]))
        if key in seen or key not in act:
            continue
        seen.add(key)
        pairs.append((r["date"], float(r["expected_ks"]), act[key], "earlyJune"))
    return pairs


def mid_pit(actual, mu, r):
    below = sum(nb_pmf(i, mu, r) for i in range(actual))     # F(actual-1)
    return below + 0.5 * nb_pmf(actual, mu, r)


def coverage(pits):
    n = len(pits)
    c50 = sum(1 for p in pits if 0.25 <= p <= 0.75) / n
    c80 = sum(1 for p in pits if 0.10 <= p <= 0.90) / n
    return n, c50, c80


def main():
    pairs = load_pairs()
    by_day = defaultdict(list)
    for d, mu, a, src in pairs:
        by_day[d].append((mu, a))
    dates = sorted(by_day)
    print(f"{len(pairs)} settled starts over {len(dates)} days "
          f"({dates[0]} -> {dates[-1]})   [line-independent]\n")

    # walk forward: fit on earlier days, test on the held-out next day
    held_pits, r_traj = [], []
    seen = []
    for i, d in enumerate(dates):
        if len(seen) >= MIN_TRAIN_STARTS:
            r = fit_dispersion(seen)
            for mu, a in by_day[d]:
                held_pits.append(mid_pit(a, mu, r))
            r_traj.append((d, r, len(by_day[d])))
        seen.extend(by_day[d])

    # in-sample baseline (fit and test on everything = overfit)
    r_all = fit_dispersion([(mu, a) for _, mu, a, _ in pairs])
    ins_pits = [mid_pit(a, mu, r_all) for _, mu, a, _ in pairs]

    n_h, h50, h80 = coverage(held_pits)
    n_i, i50, i80 = coverage(ins_pits)
    print("  interval coverage   central-50 (want ~50%)   central-80 (want ~80%)")
    print(f"  IN-SAMPLE  (r={r_all:.1f}, overfit) n={n_i:>4}     {i50*100:5.1f}%"
          f"                {i80*100:5.1f}%")
    print(f"  HELD-OUT   (walk-forward)          n={n_h:>4}     {h50*100:5.1f}%"
          f"                {h80*100:5.1f}%")
    gap = (i50 - h50) * 100
    print(f"\n  overfit gap (in-sample - held-out, central-50): {gap:+.1f} pts")
    if h50 < 0.42:
        print("  => HELD-OUT coverage well below 50%: actual K is more spread than the")
        print("     NB assumes -> model is OVERCONFIDENT out of sample at EVERY line.")
        print("     Fix: lower r (more dispersion). Try re-fitting r on a rolling window,")
        print("     or shrink it until held-out central-50 reaches ~50%.")
    elif h50 >= 0.46:
        print("  => Held-out coverage near nominal: dispersion generalizes on this data.")
    print("\n  walk-forward fitted r by test day:")
    for d, r, n in r_traj:
        print(f"    {d}  r={r:5.1f}  ({n} held-out starts)")
    print("\n  NOTE: line-independent (no betting line used). Says nothing about the")
    print("  Pick6 EDGE — only whether the K-variance is right. Early-June starts use")
    print("  an older model version (confound).")


if __name__ == "__main__":
    main()
