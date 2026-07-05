"""Grade logged Pick6 entries against real MLB strikeout results.

    python grade.py

Reads data/pick6_entries.csv, fills actual_ks + leg_won for any ungraded leg
whose game is Final (MLB StatsAPI boxscores), rewrites the file, then prints:
  - entry-level P&L / ROI (a power play pays only if ALL its legs hit)
  - leg-level reliability (predicted model_p vs realized) — the OUT-OF-SAMPLE
    check on the NB calibration; if these buckets drift, re-fit dispersion.
"""
from __future__ import annotations

import csv
import json
import math
import os
import unicodedata
import urllib.request

LOG = os.path.join(os.path.dirname(__file__), "..", "data", "pick6_entries.csv")
FIELDS = ["date", "entry_id", "platform", "n_picks", "mult", "stake", "leg_idx",
          "pitcher", "game", "market", "side", "line", "lam", "model_p", "boost",
          "rw_proj", "rw_agree", "actual_ks", "leg_won"]


def norm(name: str) -> str:
    nk = unicodedata.normalize("NFKD", name)
    nk = "".join(c for c in nk if not unicodedata.combining(c))
    return "".join(c for c in nk.lower() if c.isalpha() or c == " ").strip()


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


# market -> (boxscore stat group, field)
_STAT = {
    "strikeouts":  ("pitching", "strikeOuts"),
    "hits":        ("batting", "hits"),
    "total_bases": ("batting", "totalBases"),
    "home_runs":   ("batting", "homeRuns"),
    "rbi":         ("batting", "rbi"),
    "runs":        ("batting", "runs"),
}


def final_stats(date: str) -> dict[str, dict[str, int]]:
    """norm(player) -> {market: actual} for FINAL games on date (pitching + batting)."""
    sched = _get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}")
    out: dict[str, dict[str, int]] = {}
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            try:
                box = _get(f"https://statsapi.mlb.com/api/v1/game/{g['gamePk']}/boxscore")
            except Exception:
                continue
            for side in ("home", "away"):
                for pdata in box["teams"][side]["players"].values():
                    stats = pdata.get("stats", {})
                    rec = {}
                    for market, (grp, field) in _STAT.items():
                        v = stats.get(grp, {}).get(field)
                        if v is not None:
                            rec[market] = int(v)
                    if rec:
                        out[norm(pdata["person"]["fullName"])] = rec
    return out


def leg_won(side: str, line: float, actual: int) -> bool:
    return actual > line if side == "more" else actual < line


def main() -> None:
    if not os.path.exists(LOG):
        print(f"No log at {LOG} — run log_entries.py first.")
        return
    rows = list(csv.DictReader(open(LOG, encoding="utf-8")))

    pending_dates = sorted({r["date"] for r in rows if r["leg_won"] == ""})
    results = {d: final_stats(d) for d in pending_dates}
    graded_now = 0
    for r in rows:
        if r["leg_won"] != "":
            continue
        market = r.get("market", "strikeouts")
        actual = results.get(r["date"], {}).get(norm(r["pitcher"]), {}).get(market)
        if actual is None:
            continue  # game not Final yet (or player didn't play) — leave pending
        won = leg_won(r["side"], float(r["line"]), actual)
        r["actual_ks"] = actual   # column holds the market's actual value
        r["leg_won"] = "1" if won else "0"
        graded_now += 1

    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Graded {graded_now} new legs.\n")

    # ---- entry-level P&L (entry graded only when ALL its legs are graded) ----
    entries: dict[str, list] = {}
    for r in rows:
        entries.setdefault(r["entry_id"], []).append(r)
    staked = pnl = won_ct = graded_ct = 0.0
    print("ENTRY RESULTS")
    for eid, legs in sorted(entries.items()):
        if any(l["leg_won"] == "" for l in legs):
            continue
        graded_ct += 1
        stake = float(legs[0]["stake"]); mult = float(legs[0]["mult"])
        won = all(l["leg_won"] == "1" for l in legs)
        p = stake * (mult - 1) if won else -stake
        staked += stake; pnl += p; won_ct += won
        tag = "WON " if won else "lost"
        names = " + ".join(f"{l['pitcher'].split()[-1]} {l['side'][0].upper()}{l['line']}"
                           for l in legs)
        print(f"  {eid}  {tag}  {names}   P&L ${p:+.2f}")
    if graded_ct:
        roi = pnl / staked * 100 if staked else 0
        print(f"\n  {int(won_ct)}/{int(graded_ct)} entries won   "
              f"staked ${staked:.2f}   net ${pnl:+.2f}   ROI {roi:+.1f}%")
    else:
        print("  (no fully-graded entries yet)")

    # ---- leg-level calibration (out-of-sample NB check) ----------------------
    graded_legs = [(float(l["model_p"]), l["leg_won"] == "1")
                   for l in rows if l["leg_won"] != ""]
    if graded_legs:
        n = len(graded_legs)
        pred = sum(p for p, _ in graded_legs) / n
        real = sum(1 for _, w in graded_legs if w) / n
        print(f"\nLEG CALIBRATION (out-of-sample)  n={n}  "
              f"predicted {pred*100:.1f}%  realized {real*100:.1f}%  "
              f"gap {(real-pred)*100:+.1f} pts")
        print("  (grows each day; when n is large, drift here => re-fit dispersion)")


if __name__ == "__main__":
    main()
