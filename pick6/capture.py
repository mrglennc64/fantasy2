"""Capture a pick'em board fast from pasted lines (PrizePicks/DK/UD/etc.).

Automated scraping of these apps is bot-blocked (PrizePicks 403s servers and
fingerprints headless browsers), so the reliable path is a paste helper. You
paste "Player  line  [more|less|both]" rows off a screenshot; this looks up each
pitcher's game from the live strike slate (so you don't type matchups) and writes
the board CSV the picker reads.

    python capture.py <date> <platform> <market>   < lines.txt
    # or heredoc:
    python capture.py 2026-07-06 prizepicks strikeouts <<'EOF'
    Cristopher Sanchez 6.5
    Griffin Jax 5.5 less
    Kevin Gausman 5.5 both
    EOF

Each input line: PLAYER NAME  LINE  [more|less|both]  (side defaults to both).
Strikeout rows get game auto-filled from the slate; batter markets leave game
blank unless you append it (…  line  side  TEAM@TEAM).
"""
from __future__ import annotations

import csv
import os
import sys

import json
import urllib.request

from feed import norm

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
BATTER = {"hits", "total_bases", "home_runs", "rbi", "runs"}


def slate_games(date: str) -> dict[str, str]:
    """norm(pitcher) -> 'AWAY@HOME' from the strike slate (for auto game fill)."""
    try:
        s = json.load(urllib.request.urlopen(
            f"https://strike.perfecthold.online/api/v2/slate?date={date}", timeout=60))
    except Exception:
        return {}
    out = {}
    for r in s.get("rows", []) or []:
        g = r.get("game_pk") or r.get("opponent") or ""
        out[norm(r["pitcher"])] = str(g)
    return out


def parse(line: str):
    """'Cristopher Sanchez 6.5 less' -> (name, 6.5, 'less', game_or_None)."""
    toks = line.split()
    if len(toks) < 2:
        return None
    game = None
    if "@" in toks[-1]:
        game = toks.pop()
    side = "both"
    if toks[-1].lower() in ("more", "less", "both"):
        side = toks.pop().lower()
    try:
        val = float(toks[-1])
    except ValueError:
        return None
    name = " ".join(toks[:-1])
    return name, val, side, game


def main():
    if len(sys.argv) < 4:
        print(__doc__); return
    date, platform, market = sys.argv[1], sys.argv[2], sys.argv[3]
    is_batter = market in BATTER
    games = {} if is_batter else slate_games(date)

    rows = []
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        p = parse(raw)
        if not p:
            print(f"  ? skipped: {raw!r}")
            continue
        name, val, side, game = p
        game = game or games.get(norm(name), "")
        rows.append({
            "player": name, "team": "", "game": game, "market": market,
            "platform": platform, "line": val, "slot": "", "more_boost": 1.0,
            "more_available": str(side in ("more", "both")),
            "less_available": str(side in ("less", "both")), "notes": "captured"})

    if not rows:
        print("no valid rows parsed"); return
    suffix = "_batters" if is_batter else ""
    path = os.path.join(DATA, f"pick6_board_{date}{suffix}.csv")
    hdr = ["date", "player", "team", "game", "market", "platform", "line", "slot",
           "more_boost", "more_available", "less_available", "notes"]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({"date": date, **r})
    miss = [r["player"] for r in rows if not r["game"]]
    print(f"wrote {len(rows)} {platform} {market} rows -> {path}"
          + (f"   (no game found for: {', '.join(miss)})" if miss else ""))


if __name__ == "__main__":
    main()
