"""Log a day's paper Pick6 entries to data/pick6_entries.csv (one row per leg).

    python log_entries.py 2026-07-05

Idempotent: refuses to double-log a date already present. Grade later with
grade.py once the games settle. This is the forward-validation record — the NB
dispersion was fit in-sample on 147 starts, so out-of-sample leg hit rates here
are what actually prove (or break) the calibration before real money.
"""
from __future__ import annotations

import csv
import os
import sys

from pick6_today import compute_entries

LOG = os.path.join(os.path.dirname(__file__), "..", "data", "pick6_entries.csv")
FIELDS = ["date", "entry_id", "platform", "n_picks", "mult", "stake", "leg_idx",
          "pitcher", "game", "market", "side", "line", "lam", "model_p", "boost",
          "rw_proj", "rw_agree", "actual_ks", "leg_won"]


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


if __name__ == "__main__":
    main()
