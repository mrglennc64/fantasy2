"""Freeze the feature vector each projection was actually computed from.

    data/features/<date>.csv   one row per (date, pitcher), written once

WHAT THIS IS FOR — and what it is not. It is NOT a training corpus:
calibration/train_kmodel.py already rebuilds these same eight features from
gamelogs for ~11,800 starts, which is two orders of magnitude more data than
the board will ever produce. Nobody should expect this file to move MAE.

What it uniquely records is what the SERVING path saw, pre-game, which the
gamelog rebuild cannot reconstruct:

  1. Feature drift. kmodel.project_detail() silently substitutes the training
     mean for any feature it can't build. A pitcher whose opponent lookup
     failed scores as a league-average matchup and looks like a perfectly
     normal projection — the substitution is invisible in the output and shows
     up only as unexplained residual. `imputed` names those features per row,
     so a StatsAPI shape change surfaces as a column of misses rather than as
     a model that mysteriously got worse.
  2. Opponent K% as known BEFORE first pitch, not backfilled afterward.
  3. Residual analysis conditioned on features, per source.

WHY A SIDECAR rather than columns on predictions_log.csv. The log is one row
per (player, market, line, platform) — a pitcher recurs across platforms and
lines — while features are per (date, pitcher). Inlining would duplicate a
dozen floats several times per pitcher. It would also widen the file that
grade.py rewrites in place every half hour and that build_site.py parses on
every rebuild. Write-once-per-date matches archive_slate.py, which already
proved the pattern.
"""
from __future__ import annotations

import csv
import os

from feed import norm
from kmodel_params import FEATURES

DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")

FIELDS = (["date", "player", "player_key", "pitcher_id", "mu", "mu_source",
           "mu_version"] + list(FEATURES)
          + ["venue", "team", "opp", "imputed"])


def path_for(date: str) -> str:
    return os.path.join(DIR, f"{date}.csv")


def rows_from(date: str, legs: list[dict]) -> list[dict]:
    """One row per distinct pitcher on the board.

    Rows are emitted for non-kmodel sources too, with blank feature cells.
    "This start existed and we projected it, features unknown" is a more useful
    record than an absent row when auditing coverage later — a gap should mean
    'we never saw this start', not 'the upstream served it'.
    """
    out, seen = [], set()
    for l in legs:
        if l.get("market", "strikeouts") != "strikeouts":
            continue
        key = norm(l["name"])
        if key in seen:
            continue
        seen.add(key)
        d = l.get("_kfeat") or {}
        feats = d.get("features") or {}
        row = {"date": date, "player": l["name"], "player_key": key,
               "pitcher_id": d.get("pid", ""), "mu": f"{l['predicted']:.4f}",
               "mu_source": l.get("mu_source", "unknown"),
               "mu_version": l.get("mu_version", ""),
               "venue": d.get("venue") or "", "team": d.get("team") or "",
               "opp": d.get("opp") or "",
               "imputed": "|".join(d.get("imputed") or [])}
        for f in FEATURES:
            v = feats.get(f)
            row[f] = "" if v is None else f"{float(v):.6f}"
        out.append(row)
    return out


def write(date: str, legs: list[dict]) -> str | None:
    """Append players not yet in the day's snapshot; never rewrite one.

    Was refuse-if-file-exists; per-row logging (log_predictions.py, 2026-07-20)
    made that wrong — rows now log across several cron ticks as lineups post,
    and only the first tick's features would have been frozen. The guarantee
    that matters is per ROW and it is kept: a player's feature row is written
    once, at the moment his prediction logs, and never touched again.
    """
    p = path_for(date)
    have: set[str] = set()
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            have = {r["player_key"] for r in csv.DictReader(f)}
    rows = [r for r in rows_from(date, legs) if r["player_key"] not in have]
    if not rows:
        return None
    os.makedirs(DIR, exist_ok=True)
    new = not os.path.exists(p)
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
    return p
