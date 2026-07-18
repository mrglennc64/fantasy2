"""Crash-safe CSV rewrite.

data/predictions_log.csv is the ONLY copy of the graded accuracy record — it is
host-owned and gitignored, and it is the training data for every refit in
calibration/. grade.py rewrites it in place on every cron run (every 30 min,
15:00-02:00 UTC), and a plain open(path, "w") truncates before it writes: a
crash, an OOM kill, or a full disk between those two moments loses the entire
history.

write_rows() removes that window. The temp file lives in the SAME directory as
the target, because os.replace is only atomic within a filesystem — /tmp is
routinely a different mount, which would silently downgrade this to a copy.
"""
from __future__ import annotations

import csv
import os


def write_rows(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows to path atomically: full write + fsync, then rename over.

    extrasaction="ignore" so a key the current schema doesn't know about can
    never raise mid-rewrite — the cron must survive a row written by a newer
    version of the code.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
