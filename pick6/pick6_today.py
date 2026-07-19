"""Score today's captured prop board — full numeric output for every row.

Joins the strike/mlb-edge slate (projected strikeouts per pitcher) and the
StatsAPI batter baseline (batter_feed.py) to the captured board
(data/boards/<date>.csv), then emits a prediction for EVERY matched row:
predicted value, P(more)/P(less) vs the platform's published line, and the
model's lean with its confidence. Nothing is suppressed or filtered — rows are
simply ordered by confidence, and RotoWire's independent projection is shown
alongside as a second opinion.

    python pick6_today.py 2026-07-08
"""
from __future__ import annotations

import csv
import os
import sys

from batter_feed import project as batter_project
from crosscheck import annotate
from feed import lambdas_for
from scoring import score_leg

TOP_N = 6  # rows highlighted as the day's highest-confidence predictions

PLATFORM_ABBR = {"dk_pick6": "DK", "prizepicks": "PP", "underdog": "UD",
                 "sleeper": "SL", "betr": "BR", "parlayplay": "Pr"}
MKT_ABBR = {"strikeouts": "K", "hits": "H", "total_bases": "TB",
            "home_runs": "HR", "rbi": "RBI", "runs": "R"}
_SUFFIX = {"Jr.", "Sr.", "II", "III", "IV"}
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def leg_label(l: dict) -> str:
    """Short unambiguous label: 'Witt Jr. H M0.5' (keeps Jr./Sr. + market)."""
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
            "platform": (r.get("platform") or "dk_pick6").strip(),
            "slot": int(r["slot"]) if r.get("slot") else None,
        })
    return rows


def load_board(date: str) -> list[dict]:
    """Load the day's board from data/boards/<date>.csv (+ _batters), where the
    capture helper writes it; falls back to the legacy pick6_board_<date>.csv.
    Returns [] if nothing captured (so the cron/dashboard don't crash)."""
    rows = []
    for suffix in ("", "_batters"):
        for p in (os.path.join(DATA, "boards", f"{date}{suffix}.csv"),
                  os.path.join(DATA, f"pick6_board_{date}{suffix}.csv")):
            if os.path.exists(p):
                rows += _read_board_file(p)
                break  # prefer boards/ when present
    return rows


def _project(b: dict, date: str, slate: dict) -> dict | None:
    """Route a board row to its projection source by market.

    Returns a record ({"mu", "source", "version", "detail"}) so the source
    travels with the number all the way to scoring and the log — the whole
    point being that no downstream step ever has to guess where a mu came from.
    """
    if b["market"] == "strikeouts":
        return slate.get(b["name"])
    season = int(date[:4])
    mu = batter_project(b["name"], b["market"], season, b.get("slot"), date=date)
    if mu is None:
        return None
    return {"mu": mu, "source": "statsapi_baseline", "version": "",
            "detail": None, "bench_mu": None, "bench_source": ""}


def compute_board(date: str) -> dict:
    """Score the whole board. Returns a dict consumed by the CLI display, the
    prediction logger (log_predictions.py) and the dashboard. No printing here.
    """
    # PITCHERS ONLY (2026-07-13): batter markets are no longer scored,
    # logged, or displayed. Their board lines still get captured/archived as
    # frozen data; re-enable by removing this filter.
    board = [b for b in load_board(date) if b["market"] == "strikeouts"]
    slate = lambdas_for(board, date)

    legs, unmatched = [], []
    for b in board:
        rec = _project(b, date, slate)
        if rec is None:
            unmatched.append(b["name"])
            continue
        # _kfeat is underscore-prefixed on purpose: write_snapshot() already
        # strips _-keys, so the feature blob stays out of the frozen board JSON
        # without a new exclusion list to keep in sync.
        legs.append(score_leg({"name": b["name"], "game": b["game"],
                               "line": b["line"], "market": b["market"],
                               "platform": b["platform"], "lam": rec["mu"],
                               "mu_source": rec["source"],
                               "mu_version": rec["version"],
                               "bench_proj": rec.get("bench_mu"),
                               "bench_source": rec.get("bench_source", ""),
                               "_kfeat": rec["detail"]}))
    annotate(legs)  # RotoWire second opinion — displayed, never a filter
    legs.sort(key=lambda l: -l["p"])
    return {"date": date, "board": board, "legs": legs, "unmatched": unmatched}


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-07-08"
    res = compute_board(date)
    legs, board, unmatched = res["legs"], res["board"], res["unmatched"]
    platforms = sorted({b["platform"] for b in board})

    print(f"Prop projections [{'/'.join(PLATFORM_ABBR.get(p, p) for p in platforms)}]"
          f"  {date}   board {len(board)} rows, model matched {len(legs)}"
          + (f"  (unmatched: {', '.join(unmatched)})" if unmatched else ""))
    if not legs:
        print("No model<->board matches (is the slate live / names aligned?).")
        return

    print(f"\nPREDICTIONS (every matched row, ordered by confidence)")
    print(f"  {'player':22}{'prop':>5}{'line':>6}{'predicted':>10}{'P(more)':>9}"
          f"{'lean':>6}{'P':>8}{'RW':>7}")
    for l in legs:
        rw = l.get("rw_proj")
        rwtxt = (f"{rw:.1f}" if rw is not None else "-") + \
            {True: "=", False: "!", None: ""}[l.get("rw_agree")]
        print(f"  {l['name'][:22]:22}{MKT_ABBR.get(l['market'], l['market']):>5}"
              f"{l['line']:6.1f}{l['predicted']:10.2f}{l['p_more']*100:8.1f}%"
              f"{l['side'].upper()[:4]:>6}{l['p']*100:7.1f}%{rwtxt:>7}")

    print(f"\nTOP {min(TOP_N, len(legs))} BY CONFIDENCE")
    for l in legs[:TOP_N]:
        print(f"  {leg_label(l):26} predicted {l['predicted']:.2f} vs line "
              f"{l['line']}  P={l['p']*100:.1f}%")
    print("\n(RW = RotoWire independent projection: '=' same lean, '!' opposite"
          " lean, '-' none free for this market.)")


if __name__ == "__main__":
    main()
