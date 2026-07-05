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
import sys

from config import MIN_PICKS, breakeven_per_leg
from feed import lambdas_for
from sim import allocate, build_entries, outcome_matrix, rank_legs

BANKROLL, N_PICKS, MAX_ENTRIES = 1000.0, 3, 4
DAILY_FRAC, KELLY_FRAC, PER_CAP, MARGIN = 0.05, 0.25, 0.02, 0.05


def load_board(date: str) -> list[dict]:
    path = f"../data/pick6_board_{date}.csv"
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rows.append({
            "pitcher": r["pitcher"], "game": r["game"], "line": float(r["line"]),
            "more_boost": float(r["more_boost"]),
            "more_available": r["more_available"] == "True",
            "less_available": r["less_available"] == "True",
        })
    return rows


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-07-05"
    board = load_board(date)
    lam = lambdas_for(board, date)  # full-slate feed + per-pitcher fallback

    legs, unmatched = [], []
    for b in board:
        L = lam.get(b["pitcher"])
        if L is None:
            unmatched.append(b["pitcher"])
            continue
        legs.append({"name": b["pitcher"], "game": b["game"], "line": b["line"],
                     "lam": L, "more_boost": b["more_boost"],
                     "more_available": b["more_available"],
                     "less_available": b["less_available"]})

    print(f"DK Pick6 strikeouts  {date}   board {len(board)} legs, "
          f"model matched {len(legs)}"
          + (f"  (unmatched: {', '.join(unmatched)})" if unmatched else ""))
    if not legs:
        print("No model<->board matches (is the slate live / names aligned?).")
        return

    ranked = rank_legs(legs, N_PICKS, MARGIN)
    be = breakeven_per_leg(N_PICKS)
    print(f"\nLEG SCORES (need P >= breakeven {be*100:.1f}% + {MARGIN*100:.0f}%margin)")
    print(f"  {'pitcher':16}{'DKline':>7}{'lambda':>8}{'pick':>7}{'modelP':>8}{'keep':>6}")
    for l in (rank_legs(legs, N_PICKS, -1.0)):  # show all, flag kept
        kept = "yes" if l in ranked else ""
        print(f"  {l['name']:16}{l['line']:7.1f}{l['lam']:8.2f}"
              f"{l['side'].upper():>7}{l['p']*100:7.1f}%{kept:>6}")

    # Step down from N_PICKS to MIN_PICKS until a valid entry set exists
    # (legs must clear breakeven for that pick count AND span distinct games).
    n = N_PICKS
    entries: list = []
    while n >= MIN_PICKS:
        cand = rank_legs(legs, n, MARGIN)
        entries = build_entries(cand, n, MAX_ENTRIES) if len(cand) >= n else []
        if entries:
            break
        n -= 1
    if not entries:
        print(f"\nNo playable entry: fewer than {MIN_PICKS} independent legs clear "
              "breakeven+margin across distinct games today. Stop.")
        return
    if n < N_PICKS:
        print(f"\n(Only enough edge for a {n}-pick — stepped down from {N_PICKS}.)")

    entries, daily_cap, scale = allocate(entries, BANKROLL, DAILY_FRAC, KELLY_FRAC, PER_CAP)

    print(f"\nPOWER-PLAY ENTRIES  ({n}-pick, <= {MAX_ENTRIES}/day, "
          f"daily cap ${daily_cap:.0f}{', scaled '+format(scale,'.2f')+'x' if scale<1 else ''})")
    print(f"  {'#':>2} {'legs':40}{'P(win)':>8}{'mult':>7}{'EV':>7}{'stake':>8}")
    for i, e in enumerate(entries, 1):
        names = " + ".join(f"{l['name'].split()[-1]} {l['side'][0].upper()}{l['line']}"
                           for l in e["legs"])
        print(f"  {i:>2} {names:40}{e['p']*100:7.1f}%{e['mult']:6.1f}x"
              f"{e['ev']*100:+6.0f}%{e['stake']:8.2f}")

    om = outcome_matrix(entries)
    print(f"\nOUTCOME MATRIX  (staked ${om['staked']:.2f})")
    print(f"  expected P&L ${om['ev']:+.2f}  st.dev ${om['sd']:.2f}  "
          f"P(profit) {om['p_profit']*100:.1f}%")
    print(f"  best ${om['best'][1]:+.2f}  worst ${om['worst'][1]:+.2f}")
    print("\nPAPER ONLY — validate calibration before staking real money.")


if __name__ == "__main__":
    main()
