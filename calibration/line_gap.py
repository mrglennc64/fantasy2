"""Does the pick'em line disagree with the sportsbook line, and who is right?

    python calibration/line_gap.py [--pickem PATH] [--book PATH]

THE QUESTION, AND WHY IT IS NOT THE ONE THE REST OF THIS DIRECTORY ASKS.

Everything else here asks "is our projection better than the line". The answer,
over 164 frozen starts, was s = 0.00 — it is not, and pick6/projection.py now
anchors to the line because of it. Adding features has not moved that, and a
per-game decomposition of the 2026-07-19 slate says why it may never: innings
pitched barely tracks the miss (r = +0.18) while single-game K/9 ranged 1.8 to
16.6. The quantity we would need to forecast is mostly noise.

So this asks something with no projection in it at all: **when the two published
lines disagree, does the sportsbook's side land more often?**

Pick'em platforms post one number per player and move it slowly. Sportsbooks
reprice continuously. If the book has absorbed information the pick'em board has
not, the book's side should win more than half the disagreements — and that is a
statement about two published numbers, not about our model. It is the cheapest
remaining question in this repo, because both feeds are already being recorded.

WHAT A POSITIVE RESULT DOES AND DOES NOT ESTABLISH.

The book side of the join is the CLOSING line; the pick'em side was captured
earlier in the day. That asymmetry is the whole point (it is what "stale" means),
but it also makes a positive result NECESSARY, NOT SUFFICIENT. It shows the gap
contains information. It cannot show the gap was visible early enough to act on,
because the closing number did not exist yet at pick'em-capture time. Confirming
that needs the book line captured at the same timestamp as the board — which,
as of 2026-07-20, nothing schedules: strike's two capture timers both fire the
same `close` unit, and all 700 rows of line_history.csv are tagged close.

Read a positive here as "build the same-timestamp capture and re-run". A
negative closes the question outright, which is the cheaper and more likely
outcome and is worth just as much.

DIRECTION. gap = pickem_line - book_line.
  gap < 0  pick'em line is LOW  -> book projects more Ks -> the book side is MORE
  gap > 0  pick'em line is HIGH -> book projects fewer Ks -> the book side is LESS
Whole-number pick'em lines are exact-hit ties when the actual lands on them;
those are excluded rather than counted as either outcome.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_PICKEM = os.path.join(HERE, "..", "data", "predictions_log.csv")
DEF_BOOK = os.path.join(HERE, "..", "data", "line_history.csv")

# Minimum graded disagreements before this script will state an accuracy.
# Same spirit as calibration/gate.py: a rate printed without a bar beside it
# gets quoted back later as though it were a finding.
MIN_N = 60


def norm(name: str) -> str:
    """Accent- and case-insensitive key. The two feeds spell names differently
    ("German Marquez" vs "Germán Márquez"), and an unnormalised join silently
    drops exactly the rows it fails to match instead of reporting them."""
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().replace(".", "").replace("'", "").split())


def load_pickem(path: str) -> list[dict]:
    """Graded pitcher-strikeout rows: the pick'em line and what actually happened."""
    if not os.path.exists(path):
        return []
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if (r.get("market") or "strikeouts") != "strikeouts":
            continue
        if r.get("actual") in ("", None) or not r.get("line"):
            continue
        try:
            out.append({"date": r["date"], "key": norm(r.get("player") or r.get("pitcher", "")),
                        "name": r.get("player") or r.get("pitcher", ""),
                        "pickem": float(r["line"]), "actual": float(r["actual"])})
        except ValueError:
            continue
    return out


def load_book(path: str) -> dict[tuple[str, str], float]:
    """(date, pitcher) -> closing book line. Later captures win, so a day with
    several snapshots resolves to the last one before first pitch."""
    if not os.path.exists(path):
        return {}
    best: dict[tuple[str, str], tuple[str, float]] = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("tag") not in (None, "", "close"):
            continue
        try:
            line = float(r["line"])
        except (ValueError, KeyError):
            continue
        k = (r["date"], norm(r["pitcher"]))
        stamp = r.get("captured_at", "")
        if k not in best or stamp >= best[k][0]:
            best[k] = (stamp, line)
    return {k: v[1] for k, v in best.items()}


def grade(gap: float, pickem: float, actual: float) -> bool | None:
    """Did the book's side land? None = exact tie on a whole-number line."""
    if gap == 0:
        return None
    if actual == pickem:
        return None
    return (actual > pickem) if gap < 0 else (actual < pickem)


def required_accuracy() -> tuple[str, float]:
    """The easiest per-selection accuracy any configured multi-pick entry needs.

    Pick'em is not a two-way market, so 50% is not the reference. An entry of L
    selections paying M times the entry needs each selection to land with
    probability M**(-1/L) for the entry to return what it cost. This reports the
    most FORGIVING option in pick6/config.py, so nothing is judged against a
    threshold it would not have to meet.

    Independence is assumed, which is mildly optimistic: pick6/correlation.py
    measures a same-day factor of tau ~ 0.08, and positively correlated
    selections make a same-day entry slightly easier than this figure implies.
    Erring strict is the right direction for a threshold.
    """
    try:
        sys.path.insert(0, os.path.join(HERE, "..", "pick6"))
        from config import PLATFORMS
    except Exception:
        return ("2-selection @ 3.0x", 3.0 ** -0.5)
    best = None
    for plat, tiers in PLATFORMS.items():
        for legs, mult in tiers.items():
            p = mult ** (-1.0 / legs)
            if best is None or p < best[1]:
                best = (f"{plat} {legs}-selection @ {mult}x", p)
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickem", default=DEF_PICKEM)
    ap.add_argument("--book", default=DEF_BOOK)
    args = ap.parse_args(argv)

    pk = load_pickem(args.pickem)
    book = load_book(args.book)
    if not pk or not book:
        print(f"need both feeds: {len(pk)} pick'em rows, {len(book)} book lines.")
        return 1

    rows, unmatched = [], 0
    for r in pk:
        b = book.get((r["date"], r["key"]))
        if b is None:
            unmatched += 1
            continue
        r = dict(r, book=b, gap=round(r["pickem"] - b, 2))
        r["won"] = grade(r["gap"], r["pickem"], r["actual"])
        rows.append(r)

    dates = sorted({r["date"] for r in rows})
    print(f"joined {len(rows)} pitcher-days over {len(dates)} dates "
          f"({dates[0]}..{dates[-1]}); {unmatched} pick'em rows had no book line")

    agree = [r for r in rows if r["gap"] == 0]
    disagree = [r for r in rows if r["gap"] != 0]
    print(f"\n  lines AGREE     {len(agree):>4}  ({len(agree)/len(rows)*100:.0f}%)")
    print(f"  lines DISAGREE  {len(disagree):>4}  ({len(disagree)/len(rows)*100:.0f}%)"
          "   <- the only rows that can carry information")
    if not disagree:
        print("\nThe two markets never disagreed. Nothing to measure and nothing"
              "\nto build — the pick'em board is not stale against the book.")
        return 0

    graded = [r for r in disagree if r["won"] is not None]
    ties = len(disagree) - len(graded)
    print("\nBOOK-SIDE ACCURACY (does the side the book leans actually land?)")
    if graded:
        w = sum(1 for r in graded if r["won"])
        print(f"  {w}-{len(graded)-w} = {w/len(graded)*100:.1f}%   ({ties} exact ties excluded)")

    # If the effect is real it should GROW with the size of the disagreement.
    # A flat profile across buckets is the signature of noise, not of staleness.
    print("\n  BY GAP SIZE (a bigger disagreement should be more informative)")
    buckets = defaultdict(list)
    for r in graded:
        g = abs(r["gap"])
        tag = "0.5" if g <= 0.5 else ("1.0" if g <= 1.0 else "1.5+")
        buckets[tag].append(r)
    for tag in ("0.5", "1.0", "1.5+"):
        rs = buckets.get(tag, [])
        if not rs:
            continue
        w = sum(1 for r in rs if r["won"])
        print(f"    |gap| {tag:<4} n={len(rs):>3}   {w}-{len(rs)-w} = {w/len(rs)*100:5.1f}%")

    # If only one direction works, that is usually a systematic difference in how
    # the two feeds round or set lines, not information about the games.
    print("\n  BY DIRECTION")
    for tag, sel in (("pick'em LOW  (book says MORE)", lambda r: r["gap"] < 0),
                     ("pick'em HIGH (book says LESS)", lambda r: r["gap"] > 0)):
        rs = [r for r in graded if sel(r)]
        if not rs:
            continue
        w = sum(1 for r in rs if r["won"])
        print(f"    {tag}  n={len(rs):>3}   {w}-{len(rs)-w} = {w/len(rs)*100:5.1f}%")

    # Two thresholds, kept separate, because they answer different questions and
    # conflating them is how a 54% gets called a discovery:
    #   50%       does the disagreement carry information at all?
    #   required  is it enough to matter at the published multipliers?
    print("\n" + "=" * 70)
    if len(graded) < MIN_N:
        print(f"NOT ENOUGH DATA: {len(graded)} graded disagreements < {MIN_N}.")
        print("No rate above should be quoted as a finding yet. The join is capped"
              "\nby how many dates carry BOTH feeds — it grows by keeping"
              "\nline_history.csv and predictions_log.csv running on the same days.")
        return 0
    w = sum(1 for r in graded if r["won"])
    rate = w / len(graded)
    se = (rate * (1 - rate) / len(graded)) ** 0.5
    lo, hi = rate - 1.96 * se, rate + 1.96 * se
    print(f"n={len(graded)}  book-side accuracy {rate*100:.1f}%  "
          f"95% CI [{lo*100:.1f}%, {hi*100:.1f}%]")

    label, need = required_accuracy()
    print(f"\n  vs 50%   (informative?)  "
          + ("CLEARS — the disagreement predicts the outcome"
             if lo > 0.5 else "includes 50% — no demonstrated information"))
    print(f"  vs {need*100:.1f}% (sufficient?)   "
          + (f"CLEARS {label}"
             if lo > need else f"includes {need*100:.1f}% — short of {label}"))
    if 0.5 < rate < need:
        print("\n  NOTE: the point estimate beats 50% but falls short of the"
              "\n  multiplier threshold. Even if it holds, it is a true statement"
              "\n  about the two markets that is still not worth acting on — a"
              "\n  reason to keep measuring, not a result.")
    if lo > need:
        print("\n  THE NEXT STEP IS ANOTHER MEASUREMENT, NOT AN ACTION: this used"
              "\n  the CLOSING book line, so it cannot show the disagreement was"
              "\n  visible in time. Build the same-timestamp capture and re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
