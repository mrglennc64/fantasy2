"""Log a day's paper Pick6 entries to data/pick6_entries.csv (one row per leg).

    python log_entries.py 2026-07-05

Idempotent: refuses to double-log a date already present. Grade later with
grade.py once the games settle. This is the forward-validation record — the NB
dispersion was fit in-sample on 147 starts, so out-of-sample leg hit rates here
are what actually prove (or break) the calibration before real money.

Also freezes the scored board to data/boards/<date>_scored.json at log time:
compute_entries() re-runs with live inputs on every hourly rebuild, so without
a snapshot the dashboard would keep re-picking after the entries were bet
(7/7: the 02:30 UTC rebuild showed entries that were never logged).
build_site.py renders from the snapshot whenever one exists.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import sys

from pick6_today import compute_entries
from sim import score_leg

LOG = os.path.join(os.path.dirname(__file__), "..", "data", "pick6_entries.csv")
BOARDS = os.path.join(os.path.dirname(__file__), "..", "data", "boards")
FIELDS = ["date", "entry_id", "platform", "n_picks", "mult", "stake", "leg_idx",
          "pitcher", "game", "market", "side", "line", "lam", "model_p", "boost",
          "rw_proj", "rw_agree", "actual_ks", "leg_won"]


def snapshot_path(date: str) -> str:
    return os.path.join(BOARDS, f"{date}_scored.json")


def write_snapshot(date: str, res: dict) -> str:
    """Freeze everything the dashboard renders (legs incl. side/p, entries)."""
    legs = []
    for l in res["legs"]:
        s = score_leg(l)
        legs.append({k: v for k, v in l.items() if not k.startswith("_")}
                    | {"side": s["side"], "p": s["p"]})
    entries = [{**e, "legs": [dict(l) for l in e["legs"]]} for e in res["entries"]]
    snap = {"date": date,
            "frozen_at": datetime.datetime.now(datetime.timezone.utc)
                         .strftime("%b %d %H:%M UTC"),
            "legs": legs, "entries": entries,
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
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-07-05"
    if already_logged(date):
        print(f"{date} already logged in {LOG} — skipping (delete its rows to re-log).")
        return

    res = compute_entries(date)
    entries = res["entries"]
    if not entries:
        print(f"{date}: no playable entry to log.")
        return

    rows = []
    for i, e in enumerate(entries, 1):
        eid = f"{date}-{i}"
        for j, l in enumerate(e["legs"]):
            rw = l.get("rw_proj")
            rows.append({
                "date": date, "entry_id": eid, "platform": e.get("platform", ""),
                "n_picks": e["n"],
                "mult": f"{e['mult']:.3f}", "stake": f"{e['stake']:.2f}",
                "leg_idx": j, "pitcher": l["name"], "game": l["game"],
                "market": l.get("market", "strikeouts"),
                "side": l["side"], "line": l["line"], "lam": f"{l['lam']:.3f}",
                "model_p": f"{l['p']:.4f}", "boost": l["boost"],
                "rw_proj": "" if rw is None else f"{rw:.1f}",
                "rw_agree": {True: "1", False: "0", None: ""}[l.get("rw_agree")],
                "actual_ks": "", "leg_won": "",
            })

    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"Logged {len(entries)} entr{'y' if len(entries)==1 else 'ies'} "
          f"({len(rows)} legs) for {date} -> {LOG}")
    print(f"Froze scored board -> {write_snapshot(date, res)}")


if __name__ == "__main__":
    main()
