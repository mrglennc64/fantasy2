"""Record a day's predictions to data/predictions_log.csv (one row per leg).

    python log_predictions.py 2026-07-08

Logs every scored board row — but per ROW, when that row's information is
complete, not per date (changed 2026-07-20). A kmodel pitcher row is held until
the opposing manager posts a confirmed lineup (the model then projects from the
actual nine batters — information that did not exist when the reference line
was set) or until 45 minutes before first pitch, whichever comes first. Rows
from upstream sources log immediately: their mu is static, waiting adds
nothing. The cron re-runs this every 30 minutes, so held rows drain as lineups
post through the afternoon.

A logged row is append-only and never rewritten — the record stays a record.
`lineup_used` in the schema marks which variable each row was served with, so
the with-lineup vs without-lineup split is answerable from the log itself.

The dashboard snapshot mirrors the same rule: logged rows are frozen at their
logged values forever; unlogged rows render live, marked provisional, until
they log. (The original all-at-once freeze — 0cd4832 — existed to stop the
02:30 rebuild re-picking recorded rows. Per-row freezing keeps that guarantee
row by row.)
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

HOLD_UNTIL_MIN = 45     # log without a lineup this close to first pitch


def snapshot_path(date: str) -> str:
    return os.path.join(BOARDS, f"{date}_scored.json")


def _leg_key(name, game) -> str:
    return f"{name}|{game}"


def write_snapshot(date: str, res: dict, logged: set[str]) -> str:
    """Frozen where logged, live where pending.

    A leg already in the previous snapshot as frozen keeps those exact values —
    the numbers that were logged — no matter what this rebuild computed. Legs
    logged THIS run freeze at current values (they are what just got logged).
    Everything else renders live with provisional=True.
    """
    prev: dict[str, dict] = {}
    if os.path.exists(snapshot_path(date)):
        try:
            with open(snapshot_path(date), encoding="utf-8") as f:
                old = json.load(f)
            prev = {_leg_key(l.get("name"), l.get("game")): l
                    for l in old.get("legs", []) if l.get("frozen")}
        except Exception:
            prev = {}

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%b %d %H:%M UTC")
    legs_out = []
    for l in res["legs"]:
        clean = {k: v for k, v in l.items() if not k.startswith("_")}
        key = _leg_key(l["name"], l["game"])
        if key in prev:
            legs_out.append(prev[key])          # logged earlier: stays as logged
        elif key in logged:
            legs_out.append({**clean, "frozen": True, "frozen_at": now})
        else:
            legs_out.append({**clean, "provisional": True})
    snap = {"date": date, "frozen_at": now, "legs": legs_out,
            "unmatched": res.get("unmatched", [])}
    os.makedirs(BOARDS, exist_ok=True)
    path = snapshot_path(date)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f)
    return path


def logged_keys(date: str) -> set[str]:
    if not os.path.exists(LOG):
        return set()
    with open(LOG, encoding="utf-8") as f:
        return {_leg_key(r["player"], r["game"])
                for r in csv.DictReader(f) if r["date"] == date}


def _ready(l: dict, now_utc: datetime.datetime) -> bool:
    """May this leg be logged yet?

    Only a kmodel row without its lineup is worth holding: its mu upgrades the
    moment the lineup posts. Everything else is as informed now as it will
    ever be. The T-45min fallback exists so a lineup that never posts (rain,
    API gap) delays a row, never loses it. An unparseable game time logs
    immediately — a missing timestamp must not hold a row hostage.
    """
    if l.get("mu_source") != "kmodel":
        return True
    if l.get("lineup_used"):
        return True
    gt = l.get("_game_utc") or ""
    try:
        start = datetime.datetime.fromisoformat(gt.replace("Z", "+00:00"))
    except ValueError:
        return True
    return (start - now_utc).total_seconds() <= HOLD_UNTIL_MIN * 60


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-07-08"
    # Before appending: a stale header on disk would take longer rows silently
    # misaligned rather than failing, so migrate first.
    ensure_schema(LOG)

    res = compute_board(date)
    legs = res["legs"]
    if not legs:
        print(f"{date}: no scored rows to log (board not captured or slate not live).")
        return

    done = logged_keys(date)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    rows, held, logged_now = [], [], set()
    for l in legs:
        key = _leg_key(l["name"], l["game"])
        if key in done:
            continue
        if not _ready(l, now_utc):
            held.append(l["name"])
            continue
        rw = l.get("rw_proj")
        logged_now.add(key)
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
            "lineup_used": {True: "1", False: "0", None: ""}[l.get("lineup_used")],
        })

    if rows:
        new = not os.path.exists(LOG)
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)
        with_lu = sum(1 for r in rows if r["lineup_used"] == "1")
        print(f"Logged {len(rows)} predictions for {date} "
              f"({with_lu} with a confirmed lineup) -> {LOG}")
    if held:
        print(f"Holding {len(held)} kmodel rows for lineups: "
              f"{', '.join(held[:6])}{'...' if len(held) > 6 else ''}")
    if not rows and not held:
        print(f"{date}: all {len(legs)} rows already logged.")

    print(f"Snapshot ({len(done | logged_now)} frozen / "
          f"{len(legs) - len(done | logged_now)} provisional) -> "
          f"{write_snapshot(date, res, done | logged_now)}")
    if rows:
        feat = log_features.write(date, [l for l in legs
                                         if _leg_key(l['name'], l['game'])
                                         in logged_now])
        if feat:
            print(f"Froze serving features -> {feat}")


if __name__ == "__main__":
    main()
