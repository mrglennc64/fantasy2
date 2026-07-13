"""Weekly refit of the probability-calibration constants (pick6/calibrate.py).

    python refit_calibration.py

Fits p' = sigmoid(alpha + beta*logit(p)) on the graded pitcher rows of the
frozen prediction log, walk-forward by date (fit on days < d, score day d),
with beta monotone-constrained (>= 0). Compares held-out Brier/log-loss for
the production constants vs the refit, and prints a recommendation. The
core projection model stays stable; only this calibration layer updates
weekly — paste recommended constants into pick6/calibrate.py with
provenance when the held-out comparison supports them.

Uses raw_p_more (the uncalibrated engine probability) as the input p, so
the fit is not distorted by whatever constants were live when a row was
logged. Line-free: inputs are our stated probabilities and our outcomes.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pick6"))
from calibrate import GROUPS  # noqa: E402

LOG = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_log.csv")
MIN_TRAIN = 60
_EPS = 1e-9


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def fit(obs: list[tuple[float, bool]]) -> tuple[float, float]:
    """(alpha, beta) MLE with beta >= 0 (grid + refine)."""
    best = (0.0, 0.0, 1e18)
    for be in [x * 0.05 for x in range(21)]:
        for al in [x * 0.02 for x in range(-40, 41)]:
            s = 0.0
            for p, w in obs:
                q = min(max(_sig(al + be * _logit(p)), _EPS), 1 - _EPS)
                s -= math.log(q if w else 1 - q)
            if s < best[2]:
                best = (al, be, s)
    return best[0], best[1]


def main() -> None:
    if not os.path.exists(LOG):
        print(f"no log at {LOG}")
        return
    rows = [r for r in csv.DictReader(open(LOG, encoding="utf-8"))
            if (r.get("market") or "strikeouts") == "strikeouts"
            and r.get("result") in ("1", "0") and r.get("raw_p_more")]
    if not rows:
        print("no graded pitcher rows with a raw probability yet.")
        return

    # engine-side p (uncalibrated): the chosen side's probability from raw_p_more
    def side_p(r) -> tuple[float, bool]:
        praw = float(r["raw_p_more"])
        p = praw if praw >= 0.5 else 1 - praw
        won = ((float(r["actual"]) > float(r["line"])) == (praw >= 0.5))
        return p, won

    by_day = defaultdict(list)
    for r in rows:
        by_day[r["date"]].append(side_p(r))
    dates = sorted(by_day)
    n_all = sum(len(v) for v in by_day.values())
    print(f"CALIBRATION REFIT — {n_all} graded pitcher rows over {len(dates)} days")

    a_prod, b_prod = GROUPS["pitcher"]
    held = {"production": [], "refit": []}
    seen: list[tuple[float, bool]] = []
    for d in dates:
        if len(seen) >= MIN_TRAIN:
            a_wf, b_wf = fit(seen)
            for p, w in by_day[d]:
                for tag, (a, b) in (("production", (a_prod, b_prod)),
                                    ("refit", (a_wf, b_wf))):
                    q = max(0.5, _sig(a + b * _logit(p)))
                    held[tag].append((q, w))
        seen.extend(by_day[d])

    for tag, obs in held.items():
        if not obs:
            print(f"  {tag:10} (not enough earlier days to train yet)")
            continue
        n = len(obs)
        stated = sum(q for q, _ in obs) / n
        real = sum(1 for _, w in obs if w) / n
        brier = sum((q - (1.0 if w else 0.0)) ** 2 for q, w in obs) / n
        ll = -sum(math.log(max(q if w else 1 - q, _EPS)) for q, w in obs) / n
        print(f"  {tag:10} n={n:<4} stated {stated*100:5.1f}%  realized "
              f"{real*100:5.1f}%  Brier {brier:.4f}  log-loss {ll:.4f}")

    a_full, b_full = fit([o for v in by_day.values() for o in v])
    print(f"\nfull-sample fit: alpha={a_full:+.3f}  beta={b_full:.2f}"
          f"   (production: alpha={a_prod:+.3f}, beta={b_prod:.2f})")
    print("=> update pick6/calibrate.py only if 'refit' beats 'production' on"
          "\n   held-out Brier AND log-loss. beta>0 means confidence ranking has"
          "\n   started carrying information — spread returns exactly then.")


if __name__ == "__main__":
    main()
