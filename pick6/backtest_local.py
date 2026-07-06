"""Local backtest — no API. Joins the model's LOGGED projections+lines
(mlb-edge exports/vps/predictions.csv: real expected_ks + real sportsbook line)
to actual strikeouts (all_starters_gamelogs_2024_2026.csv: K per start), then
runs the same gated + disjoint pipeline + Pick6-line sensitivity sweep as
backtest_pick6.py — but from disk, so it's instant and uses REAL logged lines.

    python backtest_local.py

Same caveats: sportsbook line is a Pick6 proxy; the sweep shows sensitivity.
Two opposing starters share a game via `venue` (no game_pk in the log).
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

from backtest_pick6 import line_sources, run
from config import MIN_PICKS
from feed import norm

MLB = r"C:\Users\carin\OneDrive\Dokument\stike\mlb-edge\data"
PRED = os.path.join(MLB, "exports", "vps", "predictions.csv")
LOGS = os.path.join(MLB, "all_starters_gamelogs_2024_2026.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
GATE = 0.08


def load_actuals() -> dict:
    act = {}
    for r in csv.DictReader(open(LOGS, encoding="utf-8")):
        try:
            act[(r["date"], norm(r["pitcher"]))] = int(float(r["K"]))
        except (ValueError, KeyError):
            pass
    return act


def load_proj_lines() -> dict:
    """(date, norm) -> leg. One row per pitcher-date; prefer the draftkings line."""
    best = {}
    for r in csv.DictReader(open(PRED, encoding="utf-8")):
        if not r.get("expected_ks") or not r.get("line"):
            continue
        key = (r["date"], norm(r["pitcher"]))
        if key not in best or r.get("bookmaker") == "draftkings":
            best[key] = {"name": r["pitcher"], "market": "strikeouts",
                         "game": r.get("venue") or r["pitcher"],
                         "line": float(r["line"]), "lam": float(r["expected_ks"])}
    return best


def main():
    act = load_actuals()
    proj = load_proj_lines()
    day_legs = defaultdict(list)
    matched = 0
    for (date, nm), leg in proj.items():
        a = act.get((date, nm))
        if a is None:
            continue
        leg = dict(leg); leg["actual"] = a
        day_legs[date].append(leg); matched += 1
    day_legs = {d: ls for d, ls in day_legs.items() if len(ls) >= MIN_PICKS}

    dates = sorted(day_legs)
    print(f"LOCAL BACKTEST (real logged lines)   {dates[0]} -> {dates[-1]}")
    print(f"  {len(day_legs)} days, {matched} pitcher-starts joined "
          "(predictions.csv x gamelogs) · gated 0.08 · disjoint · flat 1u\n")
    print(f"  {'line source':22}{'entries':>8}{'won':>5}{'win%':>7}{'ROI':>8}   leg-calib pred->real")

    fields = ["line_source", "date", "n", "won", "pnl", "legs"]
    all_rows = []
    for tag, fn in line_sources():
        rows, ls = run(day_legs, fn, GATE)
        for r in rows:
            all_rows.append({"line_source": tag, **r})
        if not rows:
            print(f"  {tag:22}{0:>8}"); continue
        won = sum(r["won"] for r in rows); pnl = sum(r["pnl"] for r in rows)
        pred = sum(p for p, _ in ls) / len(ls); real = sum(1 for _, w in ls if w) / len(ls)
        print(f"  {tag:22}{len(rows):>8}{won:>5}{won/len(rows)*100:>6.0f}%"
              f"{pnl/len(rows)*100:>+7.0f}%   {pred*100:.1f}% -> {real*100:.1f}% (n={len(ls)})")

    out_path = os.path.join(OUT, f"backtest_local_{dates[0]}_{dates[-1]}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\n  CSV -> {out_path}")


if __name__ == "__main__":
    main()
