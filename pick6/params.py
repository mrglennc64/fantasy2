"""Learned constants, loaded at import with the checked-in .py values as the
floor.

WHY THIS EXISTS. Every fitted constant used to be printed by the weekly cron
and hand-pasted into source. That is why a stale correction survived a model
swap for five days: the mlb-edge affine kept being applied after the feed had
fallen through to the owned kmodel, and closing that gap required a human to
notice, read a report, and edit a file. The loop has to close without one.

WHY TWO FILES. deploy/cron_daily.sh runs `git reset --hard origin/main` every
half hour. Anything tracked is reverted within the hour, so the refit-written
file MUST be gitignored and host-owned — the same reason predictions_log.csv
is. And the human override MUST be tracked, precisely so it survives that
reset:

    data/params.json          gitignored, written by calibration/*.py refits
    data/params.pinned.json   tracked, hand-edited, always wins

DEFENSE IN DEPTH. Three independent layers, because this file sits in the
serving path of an unattended cron and a bad read must degrade rather than
break:

  file   — unparseable, wrong schema, or missing -> {} -> .py defaults
  value  — each consumer range-checks its OWN keys and falls back per key, so
           one absurd number can't take a whole section down with it
  operator — FANTASY_PARAMS_OFF=1 ignores both files entirely

The .py constants stay literal in their own modules. Reading projection.py
still tells you what runs when no learned file is present, which is the state
every fresh checkout and every test run is in.
"""
from __future__ import annotations

import json
import os

SCHEMA_VERSION = 1

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
PARAMS = os.path.join(_DATA, "params.json")
PINNED = os.path.join(_DATA, "params.pinned.json")


def _load(path: str) -> dict:
    """Parse a params file, or {} for ANY problem.

    Deliberately silent and deliberately broad. This runs inside the hourly
    cron; a half-written file during a concurrent refit, a truncated disk
    write, or a schema from a future version must all mean "use the defaults",
    never a traceback that costs the day's board.
    """
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return {}
    if not isinstance(d, dict) or d.get("schema_version") != SCHEMA_VERSION:
        return {}
    return d


def _loaded() -> dict:
    if os.environ.get("FANTASY_PARAMS_OFF"):
        return {}
    # Pinned wins per top-level section — that is what makes a pin durable
    # against a refit that would otherwise promote over it.
    return {**_load(PARAMS), **_load(PINNED)}


_LEARNED = _loaded()


def section(name: str) -> dict:
    """The learned dict for `name`, or {} if absent/unusable."""
    v = _LEARNED.get(name)
    return v if isinstance(v, dict) else {}


def scalar(name: str, default: float, lo: float, hi: float) -> float:
    """A learned top-level number, range-checked against the .py default."""
    v = _LEARNED.get(name)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi:
        return float(v)
    return default


def pairs(name: str, defaults: dict, lo_a: float, hi_a: float,
          lo_b: float, hi_b: float) -> dict:
    """Merge a learned {key: [a, b]} section over `defaults`, per key.

    A key whose value is malformed or out of range is dropped individually and
    keeps its default — one bad entry must not discard the others, since the
    sections here hold several independent estimators' fits.
    """
    out = dict(defaults)
    for k, v in section(name).items():
        try:
            a, b = float(v[0]), float(v[1])
        except Exception:
            continue
        if lo_a <= a <= hi_a and lo_b <= b <= hi_b:
            out[k] = (a, b)
    return out


def provenance() -> dict:
    return section("provenance")


def describe() -> str:
    """One line for the daily diagnostic: what is actually live, and from where."""
    if os.environ.get("FANTASY_PARAMS_OFF"):
        return "params: DISABLED (FANTASY_PARAMS_OFF) — serving .py defaults"
    have = [n for n, p in (("learned", PARAMS), ("pinned", PINNED))
            if _load(p)]
    if not have:
        bad = [n for n, p in (("learned", PARAMS), ("pinned", PINNED))
               if os.path.exists(p)]
        if bad:
            return (f"params: {'+'.join(bad)} file present but UNUSABLE "
                    f"(corrupt or wrong schema) — serving .py defaults")
        return "params: none on disk — serving .py defaults"
    return (f"params: {'+'.join(have)} loaded (schema v{SCHEMA_VERSION}), "
            f"written {_LEARNED.get('written_at', '?')} "
            f"by {_LEARNED.get('written_by', '?')}")
