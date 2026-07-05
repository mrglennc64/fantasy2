"""Backtest the Pick6 pitcher-strikeout strategy over a date range and write a CSV.

For each day: pull the strike/mlb-edge slate (projected K = lambda, and the book
line as a Pick6-line proxy), pull real strikeout actuals from MLB StatsAPI, run
the SAME pipeline as the live picker (calibrated NB scoring -> breakeven gate ->
step-down power-play build -> correlation-adjusted sizing), grade each entry, and
record it. Flat 1-unit stake per entry so ROI is stake-agnostic.

    python backtest_pick6.py 2026-06-05 2026-07-04

CAVEATS (read these):
  * Book line is a PROXY for the DK Pick6 line (real Pick6 boards weren't stored
    historically). Pick6 lines are usually softer, so real edge could differ.
  * No RotoWire gate (no historical projections) -> this is the UN-gated strategy.
  * The NB dispersion was fit on 6/28-7/3, so those days are IN-SAMPLE.
  * Only days with usable book lines + final games are covered; gaps are reported.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import urllib.request
from datetime import date as _date, timedelta

from config import MIN_PICKS, breakeven_per_leg
from correlation import joint_p_all, same_side
from grade import final_stats, leg_won
from sim import build_entries, rank_legs

N_PICKS, MAX_ENTRIES, MARGIN, STAKE = 3, 4, 0.05, 1.0
OUT = os.path.join(os.path.dirname(__file__), "..", "data")


def _get(u):
    with urllib.request.urlopen(u, timeout=90) as r:
        return json.load(r)


def slate_legs(d: str) -> list[dict]:
    try:
        s = _get(f"https://strike.perfecthold.online/api/v2/slate?date={d}")
    except Exception:
        return []
    legs = []
    for r in s.get("rows", []) or []:
        if r.get("expected_ks") is None or r.get("line") is None:
            continue
        legs.append({"name": r["pitcher"], "market": "strikeouts",
                     "game": r.get("game_pk") or r.get("opponent") or r["pitcher"],
                     "line": float(r["line"]), "lam": float(r["expected_ks"])})
    return legs


def daterange(a: str, b: str):
    d0 = _date.fromisoformat(a); d1 = _date.fromisoformat(b)
    while d0 <= d1:
        yield d0.isoformat()
        d0 += timedelta(days=1)


def pick(legs):
    """Same selection as the live picker: breakeven gate + step-down build."""
    n = N_PICKS
    while n >= MIN_PICKS:
        cand = rank_legs(legs, n, MARGIN)
        entries = build_entries(cand, n, MAX_ENTRIES) if len(cand) >= n else []
        if entries:
            return n, entries
        n -= 1
    return n, []


def main():
    end = sys.argv[2] if len(sys.argv) > 2 else (_date.today() - timedelta(days=1)).isoformat()
    start = sys.argv[1] if len(sys.argv) > 1 else (_date.fromisoformat(end) - timedelta(days=29)).isoformat()

    out_path = os.path.join(OUT, f"backtest_pick6_{start}_{end}.csv")
    fields = ["date", "n_picks", "legs", "model_p", "corr_p", "mult", "stake",
              "won", "pnl", "detail"]
    entry_rows, leg_samples = [], []
    days_used, days_skipped = [], []

    for d in daterange(start, end):
        legs = slate_legs(d)
        actuals = final_stats(d) if legs else {}
        # keep only legs whose game is final (we have an actual K)
        from feed import norm
        for l in legs:
            l["actual"] = actuals.get(norm(l["name"]), {}).get("strikeouts")
        legs = [l for l in legs if l["actual"] is not None]
        if len(legs) < MIN_PICKS:
            days_skipped.append(d)
            continue
        days_used.append(d)
        n, entries = pick(legs)
        for e in entries:
            e_won = all(leg_won(l["side"], l["line"], l["actual"]) for l in e["legs"])
            pnl = STAKE * (e["mult"] - 1) if e_won else -STAKE
            corr_p = joint_p_all(e["legs"])
            detail = " + ".join(
                f"{l['name'].split()[-1]} {l['side'][0].upper()}{l['line']}"
                f"={l['actual']}{'W' if leg_won(l['side'],l['line'],l['actual']) else 'L'}"
                for l in e["legs"])
            entry_rows.append({
                "date": d, "n_picks": n,
                "legs": " + ".join(f"{l['name'].split()[-1]} {l['side'][0].upper()}{l['line']}" for l in e["legs"]),
                "model_p": f"{e['p']:.4f}", "corr_p": f"{corr_p:.4f}",
                "mult": f"{e['mult']:.1f}", "stake": f"{STAKE:.2f}",
                "won": int(e_won), "pnl": f"{pnl:+.2f}", "detail": detail})
            for l in e["legs"]:
                leg_samples.append((__import__("sim").score_leg(l)["p"],
                                    leg_won(l["side"], l["line"], l["actual"])))

    os.makedirs(OUT, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(entry_rows)

    # summary
    staked = sum(STAKE for _ in entry_rows)
    pnl = sum(float(r["pnl"]) for r in entry_rows)
    won = sum(int(r["won"]) for r in entry_rows)
    print(f"BACKTEST {start} -> {end}")
    print(f"  days with usable data: {len(days_used)} / {len(days_used)+len(days_skipped)} "
          f"(skipped {len(days_skipped)} — no book lines / no finals)")
    print(f"  entries: {len(entry_rows)}   won: {won}   "
          f"win rate: {won/len(entry_rows)*100:.1f}%" if entry_rows else "  entries: 0")
    if entry_rows:
        print(f"  staked: {staked:.0f}u   net: {pnl:+.2f}u   ROI: {pnl/staked*100:+.1f}%")
        n = len(leg_samples)
        pred = sum(p for p, _ in leg_samples) / n
        real = sum(1 for _, w in leg_samples if w) / n
        print(f"  leg calibration: predicted {pred*100:.1f}%  realized {real*100:.1f}%  "
              f"gap {(real-pred)*100:+.1f} pts  (n={n})")
    print(f"  CSV -> {out_path}")


if __name__ == "__main__":
    main()
