"""Scrape the PrizePicks board into a Fantasy board CSV — via an unlocker proxy.

PrizePicks bot-blocks direct/server requests (403) and fingerprints headless
browsers. A residential UNLOCKER proxy (Bright Data Web Unlocker, or any
equivalent) gets past it. This script sends the PrizePicks JSON:API through that
proxy, parses projections -> board rows, and writes data/pick6_board_<date>.csv
(platform=prizepicks). Markets it understands: strikeouts, hits, total_bases,
home_runs, rbi, runs (others skipped).

*** YOU supply the proxy credentials — I can't sign up or hold your key. Set: ***
    BRIGHTDATA_PROXY = http://brd-customer-<id>-zone-<zone>:<pass>@brd.superproxy.io:22225
(any http(s) proxy URL works — Web Unlocker, ScraperAPI proxy mode, etc.)
If unset, it tries a DIRECT request (expect 403) so you can see it needs a proxy.

    BRIGHTDATA_PROXY="http://user:pass@host:port" python scrape_prizepicks.py 2026-07-06
    python scrape_prizepicks.py 2026-07-06 --dry     # print counts, don't write

Respect PrizePicks' ToS and rate limits; scrape at a human cadence.
"""
from __future__ import annotations

import csv
import json
import os
import ssl
import sys
import urllib.request

from feed import norm

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
BASE = "https://api.prizepicks.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Accept": "application/json",
    "Origin": "https://app.prizepicks.com",
    "Referer": "https://app.prizepicks.com/",
}
# PrizePicks stat_type label -> our market
STAT_MAP = {
    "strikeouts": "strikeouts", "pitcher strikeouts": "strikeouts",
    "hits": "hits", "total bases": "total_bases", "home runs": "home_runs",
    "hits+runs+rbis": "hits_runs_rbis", "runs": "runs", "rbis": "rbi", "rbi": "rbi",
}
BATTER = {"hits", "total_bases", "home_runs", "rbi", "runs", "hits_runs_rbis"}


def _opener():
    proxy = os.environ.get("BRIGHTDATA_PROXY") or os.environ.get("SCRAPER_PROXY")
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE  # unlocker proxies MITM TLS
        handlers.append(urllib.request.HTTPSHandler(context=ctx))
    return urllib.request.build_opener(*handlers), bool(proxy)


def get(url, opener):
    req = urllib.request.Request(url, headers=HEADERS)
    with opener.open(req, timeout=60) as r:
        return json.load(r)


def mlb_league_id(opener):
    d = get(f"{BASE}/leagues", opener)
    for lg in d.get("data", []):
        if re_mlb(lg.get("attributes", {}).get("name", "")):
            return lg["id"]
    return None


def re_mlb(name):
    n = (name or "").lower()
    return "mlb" in n or "baseball" in n


def parse(payload):
    """JSON:API projections -> [ {player, market, line} ]."""
    inc = {(i["type"], i["id"]): i for i in payload.get("included", [])}
    stat_names = {k: v["attributes"].get("name", "")
                  for k, v in inc.items() if k[0] == "stat_type"}
    rows = []
    for p in payload.get("data", []):
        a = p.get("attributes", {})
        rel = p.get("relationships", {})
        # stat type: attribute or relationship
        stat = a.get("stat_type") or ""
        st_rel = (rel.get("stat_type") or {}).get("data")
        if not stat and st_rel:
            stat = stat_names.get(("stat_type", st_rel["id"]), "")
        market = STAT_MAP.get(str(stat).strip().lower())
        if market is None or a.get("line_score") is None:
            continue
        pl = (rel.get("new_player") or rel.get("player") or {}).get("data")
        name = ""
        if pl:
            pi = inc.get((pl["type"], pl["id"]))
            name = pi["attributes"].get("name", "") if pi else ""
        name = name or a.get("description", "")
        if name:
            rows.append({"player": name, "market": market, "line": float(a["line_score"])})
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    date = sys.argv[1]
    dry = "--dry" in sys.argv
    opener, via_proxy = _opener()
    print(f"fetching PrizePicks {'via proxy' if via_proxy else 'DIRECT (expect 403)'} ...")
    try:
        lid = mlb_league_id(opener)
        if not lid:
            print("could not find MLB league id"); return
        payload = get(f"{BASE}/projections?league_id={lid}&per_page=500&single_stat=true", opener)
    except Exception as e:
        print(f"FAILED: {type(e).__name__} {getattr(e,'code','')} — "
              "set BRIGHTDATA_PROXY to a working unlocker and retry.")
        return
    rows = parse(payload)
    if not rows:
        print("no rows parsed (API shape changed or empty board)."); return

    pitchers = [r for r in rows if r["market"] == "strikeouts"]
    batters = [r for r in rows if r["market"] in BATTER]
    print(f"parsed {len(rows)}: {len(pitchers)} pitcher-K, {len(batters)} batter props")
    if dry:
        for r in rows[:12]:
            print(f"  {r['player']:22} {r['market']:12} {r['line']}")
        return

    hdr = ["date", "player", "team", "game", "market", "platform", "line", "slot",
           "more_boost", "more_available", "less_available", "notes"]
    for suffix, group in (("", pitchers), ("_batters", batters)):
        if not group:
            continue
        path = os.path.join(DATA, f"pick6_board_{date}{suffix}.csv")
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            if not exists:
                w.writeheader()
            for r in group:
                w.writerow({"date": date, "player": r["player"], "team": "", "game": "",
                            "market": r["market"], "platform": "prizepicks",
                            "line": r["line"], "slot": "", "more_boost": 1.0,
                            "more_available": "True", "less_available": "True",
                            "notes": "prizepicks scrape"})
        print(f"  wrote {len(group)} -> {path}")
    print("NOTE: game/team blank (fill via slate for pitchers, or leave for batters).")


if __name__ == "__main__":
    main()
