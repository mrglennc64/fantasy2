"""Self-checks for the Over threshold and the probabilities built on it.

    python test_markets.py            # exits non-zero on failure

No test framework in this repo, so this is a plain script — runnable from the
cron or by hand. It guards the 2026-07-20 defect: ceil() was used to derive the
Over threshold, which is right for half-integer lines and WRONG for whole
numbers (ceil(6.0) == 6, but 6 K against a 6.0 line is a push, not an Over).
That overstated P(more) by exactly pmf(line) and flipped published leans.
"""
from __future__ import annotations

import math

from dispersion import DISPERSION_R
from markets import over_threshold, p_over
from scoring import nb_pmf, p_more

FAILURES: list[str] = []


def check(label: str, got, want, tol: float = 0.0) -> None:
    ok = abs(got - want) <= tol if tol else got == want
    if not ok:
        FAILURES.append(f"{label}: got {got!r}, want {want!r}")


# ---- threshold ---------------------------------------------------------------
# Half-integer: unchanged from the old ceil() behaviour.
check("half 5.5", over_threshold(5.5), 6)
check("half 0.5", over_threshold(0.5), 1)
check("half 8.5", over_threshold(8.5), 9)
# Whole number: the regression. ceil() would give the line itself.
check("whole 6.0", over_threshold(6.0), 7)
check("whole 3.0", over_threshold(3.0), 4)
check("whole 0.0", over_threshold(0.0), 1)

# ---- P(over) is a strict inequality ------------------------------------------
# For a whole line L, P(over) must EXCLUDE the push mass at exactly L.
for mu, line in ((5.92, 6.0), (3.48, 3.0), (4.31, 4.0)):
    strict = sum(nb_pmf(k, mu, DISPERSION_R) for k in range(int(line) + 1, 60))
    check(f"strict mu={mu} line={line}", p_over("strikeouts", mu, line), strict, tol=1e-6)
    # The old ceil() answer must differ by exactly the push mass — proving the
    # bug was real and that this test would have caught it.
    buggy = sum(nb_pmf(k, mu, DISPERSION_R) for k in range(int(line), 60))
    check(f"push gap mu={mu} line={line}",
          buggy - p_over("strikeouts", mu, line),
          nb_pmf(int(line), mu, DISPERSION_R), tol=1e-6)

# ---- the three leans this defect flipped on the 2026-07-19 board -------------
# Each was published MORE and is LESS under a correct strict inequality.
for mu, line in ((5.92, 6.0), (3.48, 3.0), (4.31, 4.0)):
    check(f"lean mu={mu} line={line} is LESS", p_over("strikeouts", mu, line) < 0.5, True)

# ---- the two entry points agree ----------------------------------------------
for mu, line in ((5.92, 6.0), (6.72, 6.5), (4.85, 5.5), (3.48, 3.0)):
    check(f"p_more==p_over mu={mu} line={line}",
          p_more(mu, line), p_over("strikeouts", mu, line), tol=1e-12)

# ---- monotonicity: a higher line can never be easier to beat -----------------
prev = 1.1
for line in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0):
    p = p_over("strikeouts", 5.0, line)
    check(f"monotone at {line}", p <= prev, True)
    prev = p

# ---- Poisson markets use the same threshold ----------------------------------
check("poisson whole 1.0", p_over("home_runs", 0.4, 1.0),
      1.0 - math.exp(-0.4) * (1 + 0.4), tol=1e-9)


if __name__ == "__main__":
    import sys
    if FAILURES:
        print(f"FAIL ({len(FAILURES)})")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("ok — threshold, strictness, lean flips, agreement, monotonicity")
