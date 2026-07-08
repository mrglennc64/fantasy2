"""Archive today's slate projections FROZEN, for future dispersion re-fits.

    python archive_slate.py 2026-07-08

Writes data/slates/<date>.csv (pitcher, expected_ks) once per date and never
overwrites — the whole point is a bet-time snapshot. The /v2/slate endpoint
re-projects past dates with current season stats (outcome leakage: fitting
dispersion on it collapses r toward Poisson), so re-fits must use these frozen
files. Runs from the hourly cron; poll-safe (skips once captured).
"""
from __future__ import annotations

import csv
import os
import sys

from feed import slate_lambdas

SLATES = os.path.join(os.path.dirname(__file__), "..", "data", "slates")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: archive_slate.py <date>")
        return
    date = sys.argv[1]
    path = os.path.join(SLATES, f"{date}.csv")
    if os.path.exists(path):
        print(f"slate {date} already archived — skipping (frozen means frozen).")
        return
    lams = slate_lambdas(date)
    if not lams:
        print(f"slate {date}: no rows yet — retry next poll.")
        return
    os.makedirs(SLATES, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pitcher", "expected_ks"])
        for name, lam in sorted(lams.items()):
            w.writerow([name, f"{lam:.3f}"])
    print(f"archived {len(lams)} frozen slate lambdas -> {path}")


if __name__ == "__main__":
    main()
