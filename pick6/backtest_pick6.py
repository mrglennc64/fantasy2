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

import math

from config import MIN_PICKS, entry_multiplier
from correlation import joint_p_all
from feed import norm
from grade import final_stats, leg_won
from sim import build_entries, rank_legs, score_leg

N_PICKS, MAX_ENTRIES, MARGIN, STAKE = 3, 4, 0.05, 1.0
OUT = os.path.join(os.path.dirname(__file__), "..", "data")


def _make_entry(legs):
    boosts = [l["boost"] for l in legs]
    mult = entry_multiplier(len(legs), boosts)
    p = math.prod(l["p"] for l in legs)
    return {"legs": legs, "p": p, "mult": mult, "ev": p * mult - 1}


def build_disjoint(legs, n, max_entries):
    """Entries that never REUSE a leg (no double-counted, correlated overlap)."""
    ranked = rank_legs(legs, n, MARGIN)
    used, entries = set(), []
    while len(entries) < max_entries:
        chosen, games = [], set()
        for l in ranked:
            if id(l) in used or l["game"] in games:
                continue
            chosen.append(l); games.add(l["game"])
            if len(chosen) == n:
                break
        if len(chosen) < n:
            break
        for l in chosen:
            used.add(id(l))
        entries.append(_make_entry(chosen))
    return entries


def select(legs, mode):
    """Return (n_picks, entries) for a selection MODE, stepping 3->2 picks.
      overlap  = live default (up to 4 entries, may share legs)
      disjoint = up to 4 entries, each leg used at most once
      best     = the single top entry per day (fully independent day-bets)
    """
    n = N_PICKS
    while n >= MIN_PICKS:
        if mode == "disjoint":
            e = build_disjoint(legs, n, MAX_ENTRIES)
        else:
            cand = rank_legs(legs, n, MARGIN)
            e = build_entries(cand, n, MAX_ENTRIES) if len(cand) >= n else []
            if mode == "best":
                e = e[:1]
        if e:
            return n, e
        n -= 1
    return n, []


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


MODES = ["overlap", "disjoint", "best"]


def main():
    end = sys.argv[2] if len(sys.argv) > 2 else (_date.today() - timedelta(days=1)).isoformat()
    start = sys.argv[1] if len(sys.argv) > 1 else (_date.fromisoformat(end) - timedelta(days=29)).isoformat()

    # Fetch each day ONCE (slate + actuals are the slow part); modes reuse it.
    day_legs, days_skipped = {}, []
    for d in daterange(start, end):
        legs = slate_legs(d)
        actuals = final_stats(d) if legs else {}
        for l in legs:
            l.update(score_leg(l))                       # attach side/p/boost
            l["actual"] = actuals.get(norm(l["name"]), {}).get("strikeouts")
        legs = [l for l in legs if l["actual"] is not None]
        if len(legs) < MIN_PICKS:
            days_skipped.append(d)
        else:
            day_legs[d] = legs

    fields = ["mode", "date", "n_picks", "legs", "model_p", "corr_p", "mult",
              "stake", "won", "pnl", "detail"]
    all_rows, summary = [], {}
    for mode in MODES:
        rows, legsamp = [], []
        for d, legs in day_legs.items():
            n, entries = select(legs, mode)
            for e in entries:
                e_won = all(leg_won(l["side"], l["line"], l["actual"]) for l in e["legs"])
                pnl = STAKE * (e["mult"] - 1) if e_won else -STAKE
                detail = " + ".join(
                    f"{l['name'].split()[-1]} {l['side'][0].upper()}{l['line']}"
                    f"={l['actual']}{'W' if leg_won(l['side'],l['line'],l['actual']) else 'L'}"
                    for l in e["legs"])
                rows.append({
                    "mode": mode, "date": d, "n_picks": n,
                    "legs": " + ".join(f"{l['name'].split()[-1]} {l['side'][0].upper()}{l['line']}" for l in e["legs"]),
                    "model_p": f"{e['p']:.4f}", "corr_p": f"{joint_p_all(e['legs']):.4f}",
                    "mult": f"{e['mult']:.1f}", "stake": f"{STAKE:.2f}",
                    "won": int(e_won), "pnl": f"{pnl:+.2f}", "detail": detail})
                for l in e["legs"]:
                    legsamp.append((l["p"], leg_won(l["side"], l["line"], l["actual"])))
        all_rows += rows
        staked = len(rows) * STAKE
        pnl = sum(float(r["pnl"]) for r in rows)
        won = sum(int(r["won"]) for r in rows)
        summary[mode] = (len(rows), won, staked, pnl, legsamp)

    out_path = os.path.join(OUT, f"backtest_pick6_{start}_{end}.csv")
    os.makedirs(OUT, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)

    print(f"BACKTEST {start} -> {end}   days usable {len(day_legs)}/{len(day_legs)+len(days_skipped)}")
    print(f"  {'mode':9}{'entries':>8}{'won':>6}{'win%':>7}{'ROI':>9}   leg-calib (pred->real)")
    for mode in MODES:
        cnt, won, staked, pnl, ls = summary[mode]
        if not cnt:
            print(f"  {mode:9}{'0':>8}"); continue
        pred = sum(p for p, _ in ls) / len(ls); real = sum(1 for _, w in ls if w) / len(ls)
        print(f"  {mode:9}{cnt:>8}{won:>6}{won/cnt*100:>6.0f}%{pnl/staked*100:>+8.0f}%"
              f"   {pred*100:.1f}% -> {real*100:.1f}% (n={len(ls)})")
    print("\n  overlap  = live default (entries may share legs; correlated, inflates sample)")
    print("  disjoint = each leg used once (de-correlated, the honest strategy number)")
    print("  best     = 1 entry/day (fully independent daily bets)")
    print(f"  CSV -> {out_path}")


if __name__ == "__main__":
    main()
