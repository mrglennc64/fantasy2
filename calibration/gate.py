"""The shared promotion bar: may a refit replace what is live?

One implementation, because "beats production" has to mean the same thing for
the mean correction and the calibration layer. A gate that each script defines
for itself drifts, and the whole point of automating promotion is that nobody
is reading the reports closely enough to catch that.

The margin is RELATIVE and non-zero on purpose. A gate of "refit > production"
promotes on pure noise roughly half the time it is run, so the constants would
churn every week and the provenance trail would stop meaning anything. 1% of
Brier is small enough to let a real improvement through and large enough that
week-to-week sampling wobble does not.

Both metrics must improve. Brier rewards being right; log-loss punishes being
confidently wrong. A change that trades one for the other is not an
improvement in a system whose failure mode is overconfidence.
"""
from __future__ import annotations

import math

_EPS = 1e-9

MIN_N = 150
MIN_DAYS = 5
MIN_REL_GAIN = 0.01


def brier(obs: list[tuple[float, bool]]) -> float:
    return sum((p - (1.0 if w else 0.0)) ** 2 for p, w in obs) / len(obs)


def logloss(obs: list[tuple[float, bool]]) -> float:
    return sum(-math.log(max(p if w else 1 - p, _EPS)) for p, w in obs) / len(obs)


def promote(held_prod: list[tuple[float, bool]],
            held_refit: list[tuple[float, bool]],
            n: int, days: int,
            min_n: int = MIN_N, min_days: int = MIN_DAYS) -> tuple[bool, str]:
    """(promote?, reason). The reason is printed either way — a refusal should
    read as a decision with a cause, not as missing output."""
    if n < min_n:
        return False, f"n={n} < {min_n} graded rows"
    if days < min_days:
        return False, f"only {days} walk-forward day(s) < {min_days}"
    if not held_prod or not held_refit:
        return False, "no held-out rows scored"

    b_p, b_r = brier(held_prod), brier(held_refit)
    l_p, l_r = logloss(held_prod), logloss(held_refit)
    if b_r > b_p * (1 - MIN_REL_GAIN):
        return False, f"Brier {b_r:.4f} vs production {b_p:.4f} (< {MIN_REL_GAIN:.0%} gain)"
    if l_r > l_p * (1 - MIN_REL_GAIN):
        return False, f"log-loss {l_r:.4f} vs production {l_p:.4f} (< {MIN_REL_GAIN:.0%} gain)"
    return True, (f"Brier {b_r:.4f} < {b_p:.4f}, log-loss {l_r:.4f} < {l_p:.4f} "
                  f"(n={n}, {days} days)")
