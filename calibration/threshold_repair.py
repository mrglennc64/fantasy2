"""Repair whole-number-line probabilities logged before the 2026-07-20 fix.

    python threshold_repair.py            # report: rows affected, leans flipped
    python threshold_repair.py --self-test

Until 2026-07-20 the Over threshold was ceil(line), which counts the push as a
win on a whole-number line (see markets.over_threshold). Every such row was
logged with P(more) overstated by exactly pmf(line), and some had their side
flipped by it. refit_calibration.py fits on the LOGGED model_p_uncal, so those
rows carry the bias into the fitted calibration — the record keeps teaching the
model the wrong answer long after the serving code is correct.

WHY INVERT RATHER THAN RECOMPUTE
model_p_uncal is a frozen serving-time artifact: the corrected mean's
probability under the params in force THAT DAY. Recomputing it from `predicted`
with today's projection params would silently restate history under a different
model. Instead we recover the serving-time corrected mean by inverting the NB
CDF — P(K >= L) is strictly increasing in mu, so the inversion is unique — and
then re-derive the probability at the correct threshold. Only the threshold
changes; the model that produced the number is preserved.

The served columns (`side`, `model_p`, `result`) are NEVER rewritten. What was
published stays published; this only supplies a corrected probability for
FITTING. A track record you edit is not a track record.

LIMITATION: the inversion uses the CURRENT dispersion r. If r has been re-fitted
since a row was served, the recovered mu absorbs that drift. The correction
(subtracting the push mass) stays close to right, because it is a local
adjustment at the same threshold, but rows served under a very different r are
approximate. Half-integer rows are untouched and exact.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pick6"))
from dispersion import DISPERSION_R                      # noqa: E402
from markets import over_threshold                       # noqa: E402

LOG = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_log.csv")
_EPS = 1e-12


def _nb_pmf(k: int, mu: float, r: float = DISPERSION_R) -> float:
    mu = max(mu, _EPS)
    return math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                    + r * math.log(r / (r + mu)) + k * math.log(mu / (r + mu)))


def _p_at_least(n: int, mu: float, r: float = DISPERSION_R) -> float:
    """P(K >= n) under NB(mu, r). Strictly increasing in mu."""
    return max(0.0, 1.0 - sum(_nb_pmf(i, mu, r) for i in range(n)))


def is_whole(line: float) -> bool:
    """True when the line can push — the only case the old threshold got wrong."""
    return abs(line - round(line)) < 1e-9


def recover_mu(p_at_least: float, n: int, r: float = DISPERSION_R) -> float:
    """Invert P(K >= n) = p for mu by bisection. Unique: monotone in mu."""
    lo, hi = 1e-6, 60.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _p_at_least(n, mid, r) < p_at_least:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def repair_row(side: str, p_uncal: float, line: float,
               r: float = DISPERSION_R) -> tuple[str, float, bool]:
    """(side, p_uncal) as they SHOULD have been. Third value: did the lean flip?

    Half-integer lines are returned unchanged — the old threshold was already
    correct there.
    """
    if not is_whole(line):
        return side, p_uncal, False
    p_more_logged = p_uncal if side == "more" else 1.0 - p_uncal
    # The old code computed P(K >= ceil(line)) == P(K >= line) for a whole line.
    mu = recover_mu(p_more_logged, int(round(line)), r)
    p_more_fixed = _p_at_least(over_threshold(line), mu, r)
    new_side = "more" if p_more_fixed >= 0.5 else "less"
    new_p = p_more_fixed if new_side == "more" else 1.0 - p_more_fixed
    return new_side, new_p, new_side != side


def _self_test() -> int:
    fails = []
    r = DISPERSION_R
    for mu, line in ((5.92, 6.0), (3.48, 3.0), (4.31, 4.0), (7.10, 7.0), (2.20, 2.0)):
        # Reproduce what the buggy code would have logged.
        p_more_wrong = _p_at_least(int(line), mu, r)
        side = "more" if p_more_wrong >= 0.5 else "less"
        p_uncal = p_more_wrong if side == "more" else 1.0 - p_more_wrong
        new_side, new_p, _ = repair_row(side, p_uncal, line, r)
        # Ground truth, computed directly from the true mu.
        want_more = _p_at_least(over_threshold(line), mu, r)
        want_side = "more" if want_more >= 0.5 else "less"
        want_p = want_more if want_side == "more" else 1.0 - want_more
        if new_side != want_side:
            fails.append(f"mu={mu} line={line}: side {new_side} != {want_side}")
        if abs(new_p - want_p) > 1e-6:
            fails.append(f"mu={mu} line={line}: p {new_p:.6f} != {want_p:.6f}")
    # Half-integer rows must pass through untouched.
    for line in (5.5, 6.5):
        s, p, flipped = repair_row("more", 0.61, line)
        if (s, p, flipped) != ("more", 0.61, False):
            fails.append(f"half line {line} was modified: {(s, p, flipped)}")
    for f in fails:
        print("  -", f)
    print("FAIL" if fails else "ok — inversion recovers the true probability")
    return 1 if fails else 0


def main() -> None:
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    if not os.path.exists(LOG):
        print(f"no log at {LOG}")
        print("The record is owned by the host that runs the cron (see .gitignore).")
        print("Run this there; it only reads unless you ask otherwise.")
        return
    rows = [r for r in csv.DictReader(open(LOG, encoding="utf-8"))
            if (r.get("market") or "strikeouts") == "strikeouts"]
    graded = [r for r in rows if r.get("result") in ("1", "0")]
    affected = [r for r in graded
                if r.get("model_p_uncal") and r.get("line") and is_whole(float(r["line"]))]

    print(f"{len(rows)} logged, {len(graded)} graded, "
          f"{len(affected)} graded on a whole-number line")
    if not affected:
        print("nothing to repair.")
        return

    flips = Counter()
    hit_before = hit_after = 0
    for r in affected:
        side, p, flipped = repair_row(r["side"], float(r["model_p_uncal"]),
                                      float(r["line"]))
        actual, line = float(r["actual"]), float(r["line"])
        hit_before += int(r["result"] == "1")
        hit_after += int((actual > line) == (side == "more"))
        flips[flipped] += 1

    n = len(affected)
    print(f"leans that flip under the correct threshold: {flips[True]}/{n}")
    print(f"hit rate on those rows   as served: {hit_before}/{n} = {hit_before/n*100:.1f}%")
    print(f"                      as corrected: {hit_after}/{n} = {hit_after/n*100:.1f}%")
    print("\nServed columns are unchanged and will not be rewritten — the record "
          "of what\nwas published stays intact. Point refit_calibration.py at "
          "repair_row() so the\nFIT stops inheriting the push mass.")


if __name__ == "__main__":
    main()
