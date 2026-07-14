"""Scrape the REAL PrizePicks board via Firecrawl -> data/boards/<date>.csv.

Firecrawl's proxies get past the Cloudflare/bot wall that blocks servers, so we
fetch PrizePicks' own JSON:API (clean structured data, not fragile DOM). We take
the STANDARD line per player (skipping demon/goblin alternates), map stat types
to our markets, resolve pitcher names+games from the slate, and write the board
the scorer/cron read. These are the platform's real published lines, so the
daily loop grades predictions against exactly what was published.

*** YOU supply the key (firecrawl.dev free tier). Set FIRECRAWL_API_KEY=fc-... ***

    $env:FIRECRAWL_API_KEY="fc-..."   # PowerShell
    python scrape_firecrawl.py 2026-07-06 prizepicks
"""
from __future__ import annotations

import csv
import json
import os
import sys
import urllib.request

from capture import BOARDS, resolve, slate_rows

STAT_MAP = {
    "pitcher strikeouts": "strikeouts", "hits": "hits", "total bases": "total_bases",
    "home runs": "home_runs", "hits+runs+rbis": "hits_runs_rbis",
    "runs": "runs", "rbis": "rbi",
}
BATTER = {"hits", "total_bases", "home_runs", "rbi", "runs", "hits_runs_rbis"}


def fc_scrape(url: str, key: str) -> str:
    # maxAge=0 forces a LIVE fetch — Firecrawl v2 otherwise serves a cached copy
    # for up to 2 days, which made every scrape today return yesterday's board.
    body = json.dumps({"url": url, "formats": ["rawHtml"], "maxAge": 0}).encode()
    req = urllib.request.Request(
        "https://api.firecrawl.dev/v2/scrape", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    data = d.get("data") or {}
    return data.get("rawHtml") or data.get("markdown") or ""


def probables_rows(date: str) -> list[tuple[str, str]]:
    """[(probable pitcher, gamePk)] straight from MLB StatsAPI — the
    freshness/resolution source that cannot die with the model service
    (7/13: empty upstream slate made the guard refuse a perfectly fresh
    board all day)."""
    try:
        with urllib.request.urlopen(
                "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date="
                + date + "&hydrate=probablePitcher", timeout=45) as r:
            s = json.load(r)
    except Exception:
        return []
    out = []
    for d in s.get("dates", []):
        for g in d.get("games", []):
            for side in ("home", "away"):
                pp = g["teams"][side].get("probablePitcher") or {}
                if pp.get("fullName"):
                    out.append((pp["fullName"], str(g["gamePk"])))
    return out


def mlb_league_id(key: str) -> str:
    j = json.loads(fc_scrape("https://api.prizepicks.com/leagues", key))
    for d in j.get("data", []):
        if (d.get("attributes", {}).get("name") or "").upper() == "MLB":
            return d["id"]
    return "2"


def parse_standard(payload: str) -> list[dict]:
    """One STANDARD line per (player, market) from the JSON:API projections."""
    j = json.loads(payload)
    names = {(i["type"], i["id"]): i["attributes"].get("name")
             for i in j.get("included", []) if i["type"] in ("new_player", "player")}
    best = {}
    for p in j.get("data", []):
        a = p.get("attributes", {})
        if a.get("odds_type") not in (None, "standard"):
            continue  # skip demon/goblin alternate lines
        market = STAT_MAP.get(str(a.get("stat_type", "")).strip().lower())
        if market is None or a.get("line_score") is None:
            continue
        pl = (p.get("relationships", {}).get("new_player") or {}).get("data")
        name = names.get((pl["type"], pl["id"])) if pl else a.get("description", "")
        if not name:
            continue
        key = (name, market)
        # prefer an explicit standard; else keep the first seen
        if key not in best or a.get("odds_type") == "standard":
            best[key] = {"player": name, "market": market, "line": float(a["line_score"])}
    return list(best.values())


def main():
    if len(sys.argv) < 3:
        print(__doc__); return
    date, platform = sys.argv[1], sys.argv[2]
    key = os.environ.get("FIRECRAWL_API_KEY")
    if not key:
        print("set FIRECRAWL_API_KEY (free tier at firecrawl.dev)"); return

    print("finding MLB league via Firecrawl ...")
    lid = mlb_league_id(key)
    print(f"MLB league_id={lid}; fetching projections ...")
    try:
        payload = fc_scrape(
            f"https://api.prizepicks.com/projections?league_id={lid}&per_page=1000", key)
        rows = parse_standard(payload)
    except Exception as e:
        print(f"FAILED: {type(e).__name__} {getattr(e,'code','')}"); return
    if not rows:
        print("no standard lines parsed (board empty or API shape changed)."); return

    by_market = {}
    for r in rows:
        by_market.setdefault(r["market"], []).append(r)
    print(f"parsed {len(rows)} standard lines: "
          + ", ".join(f"{m} {len(v)}" for m, v in sorted(by_market.items())))

    # FRESHNESS GUARD: PrizePicks serves yesterday's board until it posts today's
    # (e.g. overnight US time). If the scraped pitchers aren't among today's
    # actual probable starters, it's stale — refuse to write so a good card is
    # never overwritten. Probables come from MLB StatsAPI (independent of the
    # model service); the upstream slate is only a secondary fallback.
    pslate = probables_rows(date) or slate_rows(date)
    pitchers = by_market.get("strikeouts", [])
    matched = sum(1 for r in pitchers if resolve(r["player"], pslate)[1]) if pitchers else 0
    if not pitchers or matched < max(2, len(pitchers) * 0.5):
        print(f"STALE BOARD: only {matched}/{len(pitchers)} scraped pitchers are on "
              f"today's ({date}) slate — PrizePicks hasn't posted today's board yet. "
              "Not writing (kept any existing card).")
        return

    hdr = ["date", "player", "team", "game", "market", "platform", "line", "slot",
           "more_boost", "more_available", "less_available", "notes"]
    os.makedirs(BOARDS, exist_ok=True)
    for market, group in by_market.items():
        is_batter = market in BATTER
        slate = [] if is_batter else pslate
        path = os.path.join(BOARDS, f"{date}{'_batters' if is_batter else ''}.csv")
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            if not exists:
                w.writeheader()
            for r in group:
                full, game = resolve(r["player"], slate) if slate else (r["player"], "")
                w.writerow({"date": date, "player": full, "team": "", "game": game,
                            "market": market, "platform": platform, "line": r["line"],
                            "slot": "", "more_boost": 1.0, "more_available": "True",
                            "less_available": "True", "notes": "firecrawl"})
        print(f"  {market:14} {len(group):>3} -> {path}")


if __name__ == "__main__":
    main()
