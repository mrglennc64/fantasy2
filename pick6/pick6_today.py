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
import json
import sys
import urllib.request

from config import breakeven_per_leg
from sim import allocate, build_entries, outcome_matrix, rank_legs

SLATE = "https://strike.perfecthold.online/api/v2/slate"
BANKROLL, N_PICKS, MAX_ENTRIES = 1000.0, 3, 4
DAILY_FRAC, KELLY_FRAC, PER_CAP, MARGIN = 0.05, 0.25, 0.02, 0.05


def norm(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalpha() or c == " ").strip()


def load_board(date: str) -> list[dict]:
    path = f"../data/pick6_board_{date}.csv"
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rows.append({
            "name": r["pitcher"], "game": r["game"], "line": float(r["line"]),
            "more_boost": float(r["more_boost"]),
            "more_available": r["more_available"] == "True",
            "less_available": r["less_available"] == "True",
        })
    return rows


def load_model_lambdas() -> dict[str, float]:
    """Pitcher -> projected strikeouts (lambda) from the live slate."""
    d = json.load(urllib.request.urlopen(SLATE, timeout=60))
    lam = {}
    for r in d.get("card", []) or []:
        exp = r.get("expected_ks")
        if exp is not None:
            lam[norm(r["pitcher"])] = float(exp)
    # some builds also expose a full projections list; merge if present
    for r in d.get("projections", []) or []:
        exp = r.get("expected_ks")
        if exp is not None:
            lam.setdefault(norm(r["pitcher"]), float(exp))
    return lam


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-07-05"
    board = load_board(date)
    lam = load_model_lambdas()

    legs = []
    for b in board:
        L = lam.get(norm(b["name"]))
        if L is None:
            continue  # model has no projection for this pitcher today
        legs.append({"name": b["name"], "game": b["game"], "line": b["line"],
                     "lam": L, "more_boost": b["more_boost"]})

    print(f"DK Pick6 strikeouts  {date}   board {len(board)} legs, "
          f"model matched {len(legs)}")
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

    if len(ranked) < N_PICKS:
        print(f"\nOnly {len(ranked)} legs clear breakeven+margin — need {N_PICKS}. Stop.")
        return

    entries = build_entries(ranked, N_PICKS, MAX_ENTRIES)
    entries, daily_cap, scale = allocate(entries, BANKROLL, DAILY_FRAC, KELLY_FRAC, PER_CAP)

    print(f"\nPOWER-PLAY ENTRIES  ({N_PICKS}-pick, <= {MAX_ENTRIES}/day, "
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
