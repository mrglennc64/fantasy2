"""Weekly refit of the probability-calibration constants (pick6/calibrate.py).

    python refit_calibration.py

Fits p' = sigmoid(alpha + beta*logit(p)) on the graded pitcher rows of the
frozen prediction log, walk-forward by date (fit on days < d, score day d),
with beta monotone-constrained (>= 0). Compares held-out Brier/log-loss for
the production constants vs the refit, and prints a recommendation. The
core projection model stays stable; only this calibration layer updates
weekly — paste recommended constants into pick6/calibrate.py with
provenance when the held-out comparison supports them.

INPUT DOMAIN (fixed 2026-07-18). calibrate() is applied in scoring.py to the
probability computed from the CORRECTED mu. This fit must therefore use that
same quantity, logged as model_p_uncal — the chosen side's probability after
the mean correction, before the cap, before calibrate() itself.

It previously fitted on raw_p_more, which comes from the UNCORRECTED mu. Fit
domain and apply domain were different distributions, and beta=0 hid it
completely: a constant output ignores its input, so the mismatch could not
show up in any diagnostic. It would have surfaced as systematic
miscalibration the first time beta moved off zero.

raw_p_more is still logged and still the uncorrected A/B track on the
dashboard — it just isn't what this layer is fitted on. Rows predating
model_p_uncal are skipped rather than silently mixed across the two domains.
Line-free either way: inputs are our stated probabilities and our outcomes.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pick6"))
sys.path.insert(0, os.path.dirname(__file__))
from calibrate import GROUPS  # noqa: E402

import gate            # noqa: E402
import params_io       # noqa: E402

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
    graded = [r for r in csv.DictReader(open(LOG, encoding="utf-8"))
              if (r.get("market") or "strikeouts") == "strikeouts"
              and r.get("result") in ("1", "0")]
    rows = [r for r in graded if r.get("model_p_uncal")]
    if not rows:
        print(f"no graded pitcher rows carry model_p_uncal yet "
              f"({len(graded)} graded rows predate it).")
        print("This layer is fitted on the probability it is APPLIED to. Rows "
              "logged\nbefore 2026-07-18 only have raw_p_more, from the "
              "uncorrected mu — a\ndifferent distribution. Fitting across both "
              "would be the same class of\nerror as pooling two estimators "
              "into one mean correction. Wait for the\nlog to refill.")
        return
    if len(graded) > len(rows):
        print(f"NOTE: {len(graded) - len(rows)} graded rows predate "
              f"model_p_uncal and are excluded\n(pre-2026-07-18 rows only have "
              f"the uncorrected-mu probability).\n")

    # The chosen side's probability from the CORRECTED mu — the exact quantity
    # calibrate() will be applied to in production.
    def side_p(r) -> tuple[float, bool]:
        p = float(r["model_p_uncal"])
        side_more = r["side"] == "more"
        won = ((float(r["actual"]) > float(r["line"])) == side_more)
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

    ok, why = gate.promote(held["production"], held["refit"],
                           n=n_all, days=len(dates))
    if not ok:
        print(f"\n=> KEEP production (gate: {why})")
        return
    print(f"\n=> PROMOTE pitcher: alpha={a_full:+.3f}, beta={b_full:.2f}  ({why})")
    if b_full > 0 and b_prod == 0:
        print("   beta leaves zero for the first time: the confidence ranking "
              "has\n   started carrying information, so spread returns to the "
              "board.")
    if "--write" not in sys.argv:
        print("   (report-only; re-run with --write to apply)")
        return
    path = params_io.update(
        "calibration", {"pitcher": [round(a_full, 3), round(b_full, 2)]},
        {"n": n_all, "days": len(dates), "fitted_on": "model_p_uncal",
         "held_out_brier_production": gate.brier(held["production"]),
         "held_out_brier_refit": gate.brier(held["refit"]),
         "held_out_logloss_production": gate.logloss(held["production"]),
         "held_out_logloss_refit": gate.logloss(held["refit"])},
        written_by="calibration/refit_calibration.py")
    print(f"   wrote {path}")


if __name__ == "__main__":
    main()
