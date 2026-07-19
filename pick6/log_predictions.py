"""Record a day's predictions to data/predictions_log.csv (one row per leg).

    python log_predictions.py 2026-07-08

Logs EVERY scored board row — the full numeric output, not a subset. Idempotent:
refuses to double-log a date already present. Grade later with grade.py once the
games settle; the accumulating log is the forward accuracy record (hit rate +
calibration on data the model had never seen).

Also freezes the scored board to data/boards/<date>_scored.json at log time:
compute_board() re-runs with live inputs on every hourly rebuild, so without a
snapshot the dashboard's numbers would drift after the day's predictions were
recorded (seen 7/7: the 02:30 UTC rebuild showed different rows than were
logged). build_site.py renders from the snapshot whenever one exists.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import sys

import log_features
from log_schema import FIELDS, ensure_schema
from pick6_today import compute_board

LOG = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_log.csv")
BOARDS = os.path.join(os.path.dirname(__file__), "..", "data", "boards")


def snapshot_path(date: str) -> str:
    return os.path.join(BOARDS, f"{date}_scored.json")


def write_snapshot(date: str, res: dict) -> str:
    """Freeze everything the dashboard renders."""
    snap = {"date": date,
            "frozen_at": datetime.datetime.now(datetime.timezone.utc)
                         .strftime("%b %d %H:%M UTC"),
            "legs": [{k: v for k, v in l.items() if not k.startswith("_")}
                     for l in res["legs"]],
            "unmatched": res.get("unmatched", [])}
    os.makedirs(BOARDS, exist_ok=True)
    path = snapshot_path(date)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f)
    return path


def already_logged(date: str) -> bool:
    if not os.path.exists(LOG):
        return False
    with open(LOG, encoding="utf-8") as f:
        return any(r["date"] == date for r in csv.DictReader(f))


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-07-08"
    # Before appending: a stale v1 header on disk would take 17-field rows
    # silently misaligned rather than failing, so migrate first.
    ensure_schema(LOG)
    if already_logged(date):
        print(f"{date} already logged in {LOG} — skipping (delete its rows to re-log).")
        return

    res = compute_board(date)
    legs = res["legs"]
    if not legs:
        print(f"{date}: no scored rows to log (board not captured or slate not live).")
        return

    rows = []
    for l in legs:
        rw = l.get("rw_proj")
        rows.append({
            "date": date, "player": l["name"], "game": l["game"],
            "market": l.get("market", "strikeouts"),
            "platform": l.get("platform", ""), "side": l["side"],
            "line": l["line"], "predicted": f"{l['predicted']:.3f}",
            "model_p": f"{l['p']:.4f}",
            "raw_p_more": f"{l.get('p_more_raw', ''):.4f}" if l.get('p_more_raw') is not None else "",
            "rw_proj": "" if rw is None else f"{rw:.1f}",
            "rw_agree": {True: "1", False: "0", None: ""}[l.get("rw_agree")],
            "actual": "", "result": "",
            "mu_source": l.get("mu_source", "unknown"),
            "mu_version": l.get("mu_version", ""),
            "model_p_uncal": (f"{l['p_uncal']:.4f}"
                              if l.get("p_uncal") is not None else ""),
            "bench_proj": (f"{l['bench_proj']:.3f}"
                           if l.get("bench_proj") is not None else ""),
            "bench_source": l.get("bench_source", ""),
        })

    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"Logged {len(rows)} predictions for {date} -> {LOG}")
    print(f"Froze scored board -> {write_snapshot(date, res)}")
    feat = log_features.write(date, legs)
    if feat:
        print(f"Froze serving features -> {feat}")


if __name__ == "__main__":
    main()
