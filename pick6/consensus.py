"""Freeze FantasyPros daily consensus projections -> data/slates/<date>_consensus.csv

    python consensus.py 2026-07-08

Why: fit_mean.py (2026-07-08, 164 frozen starts) showed our expected_ks has NO
edge over the pick'em line (anchor s=0.00) — the picker now correctly refuses
K entries. The only way to earn betting rights back is an INDEPENDENT
projection source measured the same honest way. FantasyPros daily projections
are an expert consensus, refreshed per-slate:
  daily-pitchers.php: IP, K, ...          daily-hitters.php: H, 2B, 3B, HR, R, RBI
TB is derived: H + 2B + 2*3B + 3*HR (2B/3B columns count EXTRA bases over the
single already in H).

RULES (learned the hard way):
  - FROZEN means frozen: written once per date, never overwritten (bet-time
    snapshot for later fits; poll-safe under the hourly cron).
  - DATE GUARD: the page shows whatever slate FP considers current; if its
    header date isn't ours, refuse to write (same lesson as the stale
    PrizePicks board).
  - NOT a betting signal yet. Nothing reads this at pick time. Once ~40+
    starts have settled, calibration/fit_mean.py grows an 'fp-anchor'
    candidate (mu' = line + t*(fp_k - line)); consensus earns t the same
    walk-forward way expected_ks lost s.

Fetch: Firecrawl when FIRECRAWL_API_KEY is set (renders JS -> full table);
plain HTTP otherwise (static HTML carries only the top ~10 rows per page —
partial but unbiased-in-selection enough to start; the cron on the VPS has
the key and gets the full slate).
"""
from __future__ import annotations

import csv
import os
import re
import sys
import urllib.request
from datetime import date as _date

FP_PITCHERS = "https://www.fantasypros.com/mlb/projections/daily-pitchers.php"
FP_HITTERS = "https://www.fantasypros.com/mlb/projections/daily-hitters.php"
SLATES = os.path.join(os.path.dirname(__file__), "..", "data", "slates")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def fetch(url: str) -> str:
    key = os.environ.get("FIRECRAWL_API_KEY")
    if key:
        try:
            from scrape_firecrawl import fc_scrape
            html = fc_scrape(url, key)
            if html:
                return html
        except Exception as e:
            print(f"  firecrawl failed ({type(e).__name__}) — plain fetch fallback")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="ignore")


def parse_table(html: str) -> tuple[list[str], list[list[str]]]:
    """(headers, rows) of the FP projections <table id="data">."""
    m = re.search(r'<table[^>]*id=.data.[^>]*>(.*?)</table>', html, re.S)
    if not m:
        return [], []
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), re.S)
    out = []
    for tr in trs:
        cells = [re.sub(r"<[^>]+>", "", c).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if cells:
            out.append(cells)
    if not out:
        return [], []
    return out[0], out[1:]


def page_date_ok(html: str, date: str) -> bool:
    """FP header says e.g. 'Daily Projections: Wed, Jul 8th'."""
    m = re.search(r"Daily Projections?:?\s*\w+,\s*(\w+) (\d+)", html)
    if not m:
        return False
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    d = _date.fromisoformat(date)
    return m.group(1)[:3] == months[d.month - 1] and int(m.group(2)) == d.day


def clean_name(cell: str) -> str:
    """'Dylan Cease (TOR - SP)' -> 'Dylan Cease'."""
    return re.sub(r"\s*\(.*$", "", cell).strip()


def _f(row: list[str], idx: dict[str, int], col: str) -> float | None:
    i = idx.get(col)
    if i is None or i >= len(row):
        return None
    try:
        return float(row[i])
    except ValueError:
        return None


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: consensus.py <date>")
        return
    date = sys.argv[1]
    path = os.path.join(SLATES, f"{date}_consensus.csv")
    if os.path.exists(path):
        print(f"consensus {date} already archived — skipping (frozen means frozen).")
        return

    recs = []
    for role, url, cols in (("pitcher", FP_PITCHERS, ["K", "IP"]),
                            ("hitter", FP_HITTERS, ["H", "2B", "3B", "HR", "R", "RBI"])):
        html = fetch(url)
        if not page_date_ok(html, date):
            print(f"{role}s page is not showing {date} — not archiving (date guard).")
            return
        hdr, rows = parse_table(html)
        idx = {h: i for i, h in enumerate(hdr)}
        if "Player" not in idx:
            print(f"{role}s table shape changed — no Player column; aborting.")
            return
        n = 0
        for row in rows:
            name = clean_name(row[idx["Player"]])
            if not name:
                continue
            vals = {c: _f(row, idx, c) for c in cols}
            if all(v is None for v in vals.values()):
                continue
            tb = None
            if role == "hitter" and None not in (vals.get("H"), vals.get("HR")):
                tb = (vals["H"] + (vals.get("2B") or 0)
                      + 2 * (vals.get("3B") or 0) + 3 * vals["HR"])
            recs.append({"role": role, "player": name,
                         "fp_k": vals.get("K"), "fp_ip": vals.get("IP"),
                         "fp_h": vals.get("H"), "fp_hr": vals.get("HR"),
                         "fp_r": vals.get("R"), "fp_rbi": vals.get("RBI"),
                         "fp_tb": tb})
            n += 1
        print(f"  {role}s: {n} rows parsed")

    if not recs:
        print("nothing parsed — not writing.")
        return
    os.makedirs(SLATES, exist_ok=True)
    fields = ["role", "player", "fp_k", "fp_ip", "fp_h", "fp_hr", "fp_r", "fp_rbi", "fp_tb"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in recs:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"archived {len(recs)} frozen consensus projections -> {path}")


if __name__ == "__main__":
    main()
