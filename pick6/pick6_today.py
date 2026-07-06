"""Score TODAY's DraftKings Pick6 pitcher-strikeout board with the mlb-edge model.

Joins the live strike/mlb-edge slate (its projected strikeouts = lambda per
pitcher) to the captured DK Pick6 board (data/pick6_board_<date>.csv), scores
More/Less against DK's OWN line (not the sportsbook line — DK's soft number is
the edge), and builds paper power-play entries.

    python pick6_today.py 2026-07-05

*** PAPER ONLY — model is uncalibrated / over-projects. Run the calibration
backtest (calibration/backtest.py) before trusting any breakeven. ***
"""
from __future__ import annotations

import csv
import os
import sys

from collections import Counter

from batter_feed import project as batter_project
from config import DEFAULT_PLATFORM, ENTRY_SIZES, breakeven_per_leg
from correlation import corr_outcome_matrix, joint_p_all, same_side
from crosscheck import annotate, gate
from feed import lambdas_for
from sim import allocate, build_entries, outcome_matrix, rank_legs, score_leg

BANKROLL, MAX_ENTRIES, PER_PS = 1000.0, 6, 3   # PER_PS: entries kept per platform×size
MAX_PER_PLAYER = 2                              # cluster cap: max entries anchored on one player
DAILY_FRAC, KELLY_FRAC, PER_CAP, MARGIN = 0.05, 0.25, 0.02, 0.05
REQUIRE_AGREE = True  # Phase 3: drop legs RotoWire disagrees with
PLATFORM_ABBR = {"dk_pick6": "DK", "prizepicks": "PP", "underdog": "UD",
                 "sleeper": "SL", "betr": "BR", "parlayplay": "Pr"}


DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MKT_ABBR = {"strikeouts": "K", "hits": "H", "total_bases": "TB",
            "home_runs": "HR", "rbi": "RBI", "runs": "R"}
_SUFFIX = {"Jr.", "Sr.", "II", "III", "IV"}


def leg_label(l: dict) -> str:
    """Short unambiguous leg label: 'Witt Jr. H O0.5' (keeps Jr./Sr. + market)."""
    parts = l["name"].split()
    surname = " ".join(parts[-2:]) if len(parts) > 1 and parts[-1] in _SUFFIX else parts[-1]
    mkt = MKT_ABBR.get(l["market"], l["market"])
    return f"{surname} {mkt} {l['side'][0].upper()}{l['line']}"


def _read_board_file(path: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rows.append({
            "name": (r.get("player") or r.get("pitcher") or "").strip(),
            "game": r["game"], "line": float(r["line"]),
            "market": r.get("market", "strikeouts"),
            "platform": (r.get("platform") or DEFAULT_PLATFORM).strip(),
            "slot": int(r["slot"]) if r.get("slot") else None,
            "more_boost": float(r["more_boost"]),
            "more_available": r["more_available"] == "True",
            "less_available": r["less_available"] == "True",
        })
    return rows


def load_board(date: str) -> list[dict]:
    """Pitcher board plus an optional batter board (captured from DK's separate
    tabs): pick6_board_<date>.csv  +  pick6_board_<date>_batters.csv. Returns []
    if no board was captured for the date (so the cron/dashboard don't crash)."""
    rows = []
    for suffix in ("", "_batters"):
        p = os.path.join(DATA, f"pick6_board_{date}{suffix}.csv")
        if os.path.exists(p):
            rows += _read_board_file(p)
    return rows


def _project(b: dict, date: str, slate: dict) -> float | None:
    """Route a board row to its projection source by market."""
    if b["market"] == "strikeouts":
        return slate.get(b["name"])
    season = int(date[:4])
    return batter_project(b["name"], b["market"], season, b.get("slot"))


def compute_entries(date: str) -> dict:
    """Build the day's paper entries. Returns a dict consumed by both the CLI
    display and the entry logger (log_entries.py). No printing here.
    """
    board = load_board(date)
    # strikeout λ come from the mlb-edge slate (batch); batter λ from StatsAPI.
    slate = lambdas_for([b for b in board if b["market"] == "strikeouts"], date)

    legs, unmatched = [], []
    for b in board:
        L = _project(b, date, slate)
        if L is None:
            unmatched.append(b["name"])
            continue
        legs.append({"name": b["name"], "game": b["game"], "line": b["line"],
                     "market": b["market"], "platform": b["platform"], "lam": L,
                     "more_boost": b["more_boost"],
                     "more_available": b["more_available"],
                     "less_available": b["less_available"]})

    # Phase 3: attach a model side, then RotoWire second-opinion agreement, and
    # gate out legs RotoWire explicitly disagrees with.
    for l in legs:
        l["side"] = score_leg(l)["side"]
    annotate(legs)
    gated = gate(legs, REQUIRE_AGREE)

    # Build entries for every PLATFORM x SIZE (2..5): each entry lives on one
    # platform (can't mix apps), sized 2-5. Legs are line-shopped implicitly —
    # a softer line on any platform scores higher and surfaces.
    raw = []
    for plat in sorted({l["platform"] for l in gated}):
        plegs = [l for l in gated if l["platform"] == plat]
        for n in ENTRY_SIZES:
            cand = rank_legs(plegs, n, MARGIN, plat)
            if len(cand) >= n:
                raw += build_entries(cand, n, PER_PS, plat)

    # Phase 4: correlation-adjust, then rank by corrected EV.
    for e in raw:
        e["corr_p"] = joint_p_all(e["legs"])
        e["corr_ev"] = e["corr_p"] * e["mult"] - 1.0
        e["same_side"] = same_side(e["legs"])
        b = e["mult"] - 1.0
        e["kelly"] = max(0.0, (e["corr_p"] * b - (1 - e["corr_p"])) / b)
    raw.sort(key=lambda e: e["corr_ev"], reverse=True)

    # Best entry at each size 2..5 (line-shopped to the best platform), with a
    # per-player cluster cap so no single player anchors every entry — on 7/5 one
    # leg (Weathers) sat in all 4 sizes, so his miss went 0/4. Iterate sizes
    # SMALLEST-first (shorter sets hit more often) so each size gets a fair pick
    # of the scarce strong legs; cap => one miss sinks <= MAX_PER_PLAYER entries.
    # On a thin slate this yields fewer than 4 entries — honestly, it should.
    used, entries = Counter(), []
    for n in ENTRY_SIZES:
        for e in raw:                       # raw is EV-desc
            if e["n"] != n or e["corr_ev"] <= 0:
                continue
            names = [l["name"] for l in e["legs"]]
            if any(used[nm] >= MAX_PER_PLAYER for nm in names):
                continue
            entries.append(e)
            for nm in names:
                used[nm] += 1
            break

    daily_cap = scale = None
    if entries:
        entries, daily_cap, scale = allocate(
            entries, BANKROLL, DAILY_FRAC, KELLY_FRAC, PER_CAP)
    return {"date": date, "board": board, "legs": legs, "unmatched": unmatched,
            "entries": entries, "daily_cap": daily_cap, "scale": scale}


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-07-05"
    res = compute_entries(date)
    legs, entries = res["legs"], res["entries"]
    board, unmatched = res["board"], res["unmatched"]
    platforms = sorted({b["platform"] for b in board})

    print(f"Pick'em edge [{'/'.join(PLATFORM_ABBR.get(p, p) for p in platforms)}]  {date}"
          f"   board {len(board)} legs, model matched {len(legs)}"
          + (f"  (unmatched: {', '.join(unmatched)})" if unmatched else ""))
    if not legs:
        print("No model<->board matches (is the slate live / names aligned?).")
        return

    kept_names = {x["name"] for e in entries for x in e["legs"]}
    print(f"\nLEG SCORES (kept legs clear the size/platform breakeven + "
          f"{MARGIN*100:.0f}% margin; RotoWire must agree)")
    print(f"  {'player':22}{'prop':>5}{'line':>6}{'lambda':>8}{'pick':>7}{'modelP':>8}"
          f"{'RW':>6}{'play':>6}")
    for l in sorted(legs, key=lambda x: -score_leg(x)["p"]):
        s = score_leg(l)
        rw = l.get("rw_proj")
        rwtxt = (f"{rw:.1f}" if rw is not None else "-") + \
            {True: "ok", False: "!!", None: ""}[l.get("rw_agree")]
        play = "yes" if l["name"] in kept_names else ""
        print(f"  {l['name'][:22]:22}{MKT_ABBR.get(l['market'], l['market']):>5}"
              f"{l['line']:6.1f}{l['lam']:8.2f}{s['side'].upper():>7}{s['p']*100:7.1f}%"
              f"{rwtxt:>6}{play:>6}")

    if not entries:
        print("\nNo playable entry today: no 2-5 pick combo clears breakeven+margin "
              "on any platform across distinct games. Stop.")
        return

    daily_cap, scale = res["daily_cap"], res["scale"]
    apps = "/".join(PLATFORM_ABBR.get(p, p) for p in platforms)
    print(f"\nPICK'EM ENTRIES  (best 2/3/4/5-pick, line-shopped across {apps}, "
          f"daily cap ${daily_cap:.0f}"
          f"{', scaled '+format(scale,'.2f')+'x' if scale<1 else ''})")
    print("  (P_cor = day-correlation-adjusted win prob, used for sizing)")
    print(f"  {'#':>2} {'app':>3}{'pk':>3}  {'legs':40}{'P_cor':>7}{'mult':>6}{'EV':>7}{'stake':>8}")
    for i, e in enumerate(entries, 1):
        names = " + ".join(leg_label(l) for l in e["legs"])
        conc = " *same-side" if e["same_side"] else ""
        print(f"  {i:>2} {PLATFORM_ABBR.get(e['platform'], e['platform']):>3}{e['n']:>3}  "
              f"{names:40}{e['corr_p']*100:6.1f}%{e['mult']:5.1f}x"
              f"{e['corr_ev']*100:+6.0f}%{e['stake']:8.2f}{conc}")

    om_i = outcome_matrix(entries)
    om = corr_outcome_matrix(entries)
    print(f"\nOUTCOME MATRIX  (staked ${om['staked']:.2f})   [independent -> correlated]")
    print(f"  expected P&L ${om_i['ev']:+.2f} -> ${om['ev']:+.2f}   "
          f"st.dev ${om_i['sd']:.2f} -> ${om['sd']:.2f}   "
          f"P(profit) {om_i['p_profit']*100:.1f}% -> {om['p_profit']*100:.1f}%")
    print(f"  best ${om['best'][1]:+.2f}  worst ${om['worst'][1]:+.2f}")
    if any(e["same_side"] for e in entries):
        print("  * same-side entries sweep together on an extreme K day — higher "
              "win prob, fatter tail (that's the correlation at work).")
    print("\nPAPER ONLY — validate calibration before staking real money.")


if __name__ == "__main__":
    main()
