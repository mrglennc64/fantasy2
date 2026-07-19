"""The prediction-log schema, in one place.

grade.py and log_predictions.py both write data/predictions_log.csv and each
used to carry its own copy of the FIELDS literal. Two copies of a DictWriter
fieldname list is a latent crash: the moment they drift, whichever script owns
the extra key raises ValueError mid-write on the hourly cron.

SCHEMA v2 (2026-07-18) appends three columns at the END of v1. Appending is
safe because every reader uses csv.DictReader (name-keyed, not positional), and
end-append keeps the file readable by eye and by any external tooling that
already knows the first fourteen columns.

  mu_source       WHICH estimator produced `predicted` (see feed.py SRC_*).
                  Keys the per-source mean correction in projection.py. Without
                  it, fit_mean.py pools two different estimators into one fit —
                  which is exactly how the mlb-edge affine ended up being
                  applied to owned-kmodel projections after 2026-07-13.
  mu_version      WHICH fitted params, within that source. Deliberately separate
                  from mu_source: a kmodel retrain must be able to invalidate a
                  kmodel-specific correction without the coarse source key
                  growing a new value every time we retrain.
  model_p_uncal   The chosen side's probability from the CORRECTED mu, before
                  the cap and before calibrate(). This is the quantity
                  calibrate() is applied to, so it is the quantity the
                  calibration must be fitted on (see refit_calibration.py).
                  raw_p_more stays what it always was — the uncorrected A/B
                  track — rather than being repurposed.
"""
from __future__ import annotations

import csv
import os

import atomicio

SCHEMA_VERSION = 3

_V1 = ["date", "player", "game", "market", "platform", "side", "line",
       "predicted", "model_p", "raw_p_more", "rw_proj", "rw_agree",
       "actual", "result"]

_V2 = _V1 + ["mu_source", "mu_version", "model_p_uncal"]

# v3 (2026-07-19): the owned kmodel became the served projection and mlb-edge
# became a benchmark. bench_proj is what the upstream said for the SAME start,
# recorded so the head-to-head is answerable from the record rather than from
# an opinion — the same standing rw_proj has. Blank when upstream had no
# projection, or when upstream itself served the row (nothing to compare).
FIELDS = _V2 + ["bench_proj", "bench_source"]

# Last date the mlb-edge upstream served projections. From 2026-07-13 the feed
# chain could fall through to the owned kmodel (commit 40ab36d), so rows from
# that date onward are of genuinely UNKNOWN source and must not be guessed.
LEGACY_SOURCE_CUTOVER = "2026-07-13"

SRC_UNKNOWN = "unknown"
SRC_LEGACY = "mlbedge_slate"


def legacy_source(date: str) -> str:
    """Source to assume for a pre-schema-v2 row that has no mu_source.

    Before the cutover every projection came from mlb-edge, so that label is a
    fact, not a guess. On or after it, the row could be either estimator and we
    say so: "unknown" maps to the identity correction and is excluded from
    fitting. Labelling those rows kmodel to grow the sample would poison the
    very fit they'd be feeding.
    """
    return SRC_LEGACY if date < LEGACY_SOURCE_CUTOVER else SRC_UNKNOWN


def backfill(rows: list[dict]) -> list[dict]:
    """Fill later-schema columns on rows read from an older file. In place."""
    for r in rows:
        if not r.get("mu_source"):
            r["mu_source"] = legacy_source(r.get("date", ""))
        for c in ("mu_version", "model_p_uncal", "bench_proj", "bench_source"):
            r.setdefault(c, "")
    return rows


def ensure_schema(path: str) -> None:
    """Migrate a v1 log to v2 on disk, atomically. No-op if already current.

    log_predictions.py APPENDS, so it cannot be the thing that discovers a
    stale header — appending 17-field rows under a 14-column header produces a
    file that is silently misaligned rather than one that fails loudly. Both
    writers call this first.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == FIELDS:
            return
        rows = list(reader)
    atomicio.write_rows(path, FIELDS, backfill(rows))
    print(f"Migrated {os.path.basename(path)} to schema v{SCHEMA_VERSION} "
          f"({len(rows)} rows)")
