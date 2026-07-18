"""Write learned constants to data/params.json, atomically and reversibly.

Read the counterpart in pick6/params.py for why the file exists and why it is
gitignored while data/params.pinned.json is tracked.

Two properties this module has to guarantee:

  SECTION ISOLATION. fit_mean.py and refit_calibration.py both write, and the
  weekly cron may run them back to back. Each does a read-modify-write of only
  its own section, so neither can clobber the other's promotion by writing a
  whole document built from a stale read.

  REVERSIBILITY. Once constants stop living in git, the commit history stops
  being the audit trail. Every write also drops an immutable snapshot in
  data/params_history/ — never overwritten, so rolling back a bad promotion is
  a file copy, and "what was live on the 14th, and why" stays answerable.
"""
from __future__ import annotations

import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pick6"))
from params import PARAMS, SCHEMA_VERSION  # noqa: E402

HISTORY = os.path.join(os.path.dirname(__file__), "..", "data", "params_history")


def _read() -> dict:
    try:
        with open(PARAMS, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and d.get("schema_version") == SCHEMA_VERSION:
            return d
    except Exception:
        pass
    return {"schema_version": SCHEMA_VERSION}


def update(name: str, value, provenance: dict, written_by: str) -> str:
    """Merge one section into data/params.json. Returns the path written."""
    doc = _read()
    doc[name] = value
    doc.setdefault("provenance", {})[name] = provenance
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc["written_at"] = stamp
    doc["written_by"] = written_by

    os.makedirs(os.path.dirname(PARAMS) or ".", exist_ok=True)
    blob = json.dumps(doc, indent=2, sort_keys=True)
    # Temp file in the SAME directory: os.replace is only atomic within one
    # filesystem, and a reader in the hourly cron must never see a partial doc.
    tmp = PARAMS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, PARAMS)

    os.makedirs(HISTORY, exist_ok=True)
    snap = os.path.join(HISTORY, f"{stamp.replace(':', '')}-{name}.json")
    with open(snap, "w", encoding="utf-8") as f:
        f.write(blob)
    return PARAMS
