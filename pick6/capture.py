"""Capture a pick'em board from RAW pasted text — no API, no keys.

Select-all + copy the DK Pick6 (or PrizePicks/Underdog) board and paste it in.
This parses the messy text into player cards — name, matchup, line, More/Less
availability, and any x-boost — and writes data/boards/<date>.csv (+ _batters
for batter markets), which the picker and the daily cron read automatically.

    python capture.py <date> <platform> <market>   < board.txt
    # e.g. copy the DK Pitcher-Strikeouts tab, then:
    python capture.py 2026-07-06 dk_pick6 strikeouts <<'EOF'
    <paste the whole board here>
    EOF

Parsing is best-effort: it prints what it found and flags anything odd — always
eyeball the CSV. Falls back to a simple "Player  line  [more|less|both]" format
if the paste has no position lines. Games are taken from the matchup text (so
two opposing starters share a game); pitcher games missing from the text are
auto-filled from the live slate.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import urllib.request

from feed import norm

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
BOARDS = os.path.join(DATA, "boards")
BATTER = {"hits", "total_bases", "home_runs", "rbi", "runs", "hits_runs_rbis"}

POS = re.compile(r"^(SP|RP|P|C|1B|2B|3B|SS|LF|CF|RF|DH|OF|IF|UT)\b", re.I)
MATCHUP = re.compile(r"\b([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\b")
HALFLINE = re.compile(r"(?<!\d)(\d{1,2}\.5)(?!\d)")
BOOST = re.compile(r"\b(\d\.\d)\s*x\b", re.I)
NOISE = re.compile(r"strikeouts|thrown|total bases|hits|home runs|runs|rbis?|"
                   r"more|less|^\d{1,2}:\d\d|am|pm|sun|mon|tue|wed|thu|fri|sat", re.I)


def slate_rows(date: str) -> list[tuple[str, str]]:
    """[(full_name, game)] from the live slate — for name + game resolution."""
    try:
        s = json.load(urllib.request.urlopen(
            f"https://strike.perfecthold.online/api/v2/slate?date={date}", timeout=60))
    except Exception:
        return []
    return [(r["pitcher"], str(r.get("game_pk") or r.get("opponent") or ""))
            for r in s.get("rows", []) or []]


def resolve(name: str, slate: list[tuple[str, str]]) -> tuple[str, str]:
    """DK abbreviates first names ('J. Ryan'); map to the slate's full name + game
    by last name (+ first initial to disambiguate). Returns (full_or_orig, game)."""
    key = norm(name)
    for full, g in slate:
        if norm(full) == key:
            return full, g
    parts = name.replace(".", " ").split()
    if len(parts) >= 2:
        last, fi = norm(parts[-1]), parts[0][:1].lower()
        cands = [(full, g) for full, g in slate if norm(full).split()[-1:] == [last]]
        if len(cands) == 1:
            return cands[0]
        for full, g in cands:
            if norm(full)[:1] == fi:
                return full, g
    return name, ""


def parse_raw(lines: list[str]) -> list[dict]:
    """DK/PP card layout: a position line follows the player name; the card's
    attributes (matchup, line, more/less, boost) sit between it and the next."""
    pos_idx = [i for i, l in enumerate(lines) if POS.match(l)]
    cards = []
    for k, pi in enumerate(pos_idx):
        name = lines[pi - 1] if pi > 0 else ""
        end = pos_idx[k + 1] - 1 if k + 1 < len(pos_idx) else len(lines)
        span = " \n ".join(lines[pi:end])
        line = HALFLINE.search(span)
        if not name or not line:
            continue
        mu = MATCHUP.search(span)
        game = f"{mu.group(1)}@{mu.group(2)}" if mu else ""
        has_more = bool(re.search(r"\bmore\b", span, re.I))
        has_less = bool(re.search(r"\bless\b", span, re.I))
        boost = BOOST.search(span)
        cards.append({"name": name.strip(), "line": float(line.group(1)),
                      "game": game, "more": has_more or not has_less,
                      "less": has_less, "boost": float(boost.group(1)) if boost else 1.0})
    return cards


def parse_simple(lines: list[str]) -> list[dict]:
    """Fallback: 'Player Name  5.5  [more|less|both]  [TEAM@TEAM]'."""
    out = []
    for l in lines:
        toks = l.split()
        if len(toks) < 2:
            continue
        game = toks.pop() if (toks and "@" in toks[-1]) else ""
        side = toks.pop().lower() if toks[-1].lower() in ("more", "less", "both") else "both"
        try:
            val = float(toks[-1])
        except ValueError:
            continue
        out.append({"name": " ".join(toks[:-1]), "line": val, "game": game,
                    "more": side in ("more", "both"), "less": side in ("less", "both"),
                    "boost": 1.0})
    return out


def main():
    if len(sys.argv) < 4:
        print(__doc__); return
    date, platform, market = sys.argv[1], sys.argv[2], sys.argv[3]
    is_batter = market in BATTER
    lines = [l.strip() for l in sys.stdin if l.strip()]

    cards = parse_raw(lines)
    if not cards:                      # no position lines -> try the simple format
        cards = parse_simple(lines)
    if not cards:
        print("could not parse any player cards — paste the board text or use "
              "'Player  line  [more|less|both]' rows."); return

    slate = [] if is_batter else slate_rows(date)
    os.makedirs(BOARDS, exist_ok=True)
    suffix = "_batters" if is_batter else ""
    path = os.path.join(BOARDS, f"{date}{suffix}.csv")
    hdr = ["date", "player", "team", "game", "market", "platform", "line", "slot",
           "more_boost", "more_available", "less_available", "notes"]
    exists = os.path.exists(path)
    n, miss = 0, []
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if not exists:
            w.writeheader()
        for c in cards:
            full, sgame = resolve(c["name"], slate) if slate else (c["name"], "")
            c["name"] = full
            game = c["game"] or sgame
            if not game:
                miss.append(c["name"])
            w.writerow({"date": date, "player": c["name"], "team": "", "game": game,
                        "market": market, "platform": platform, "line": c["line"],
                        "slot": "", "more_boost": c["boost"],
                        "more_available": str(c["more"]), "less_available": str(c["less"]),
                        "notes": "captured"})
            n += 1
    print(f"parsed {n} {platform} {market} cards -> {path}")
    for c in cards[:min(n, 20)]:
        b = f" {c['boost']}x" if c["boost"] != 1.0 else ""
        s = "More/Less" if c["more"] and c["less"] else ("More" if c["more"] else "Less")
        print(f"  {c['name']:22} {c['line']:>4} {c['game']:8} {s}{b}")
    if miss:
        print(f"\n  no game matched (verify): {', '.join(miss)}")


if __name__ == "__main__":
    main()
