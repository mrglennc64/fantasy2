"""Generate a self-contained static dashboard for fantasy.perfecthold.online.

Renders the day's full numeric prediction board and the accuracy record
(hit rate + out-of-sample calibration) into a single inline-styled index.html
that nginx can serve statically. Run at build/deploy time (it hits the live
slate once):

    python build_site.py 2026-07-08 ../web/dist/index.html

No serve-time dependencies — pure static output. Every matched board row is
shown with real numbers; nothing is suppressed.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pick6"))
from pick6_today import PLATFORM_ABBR, TOP_N, compute_board  # noqa: E402
from markets import market_side  # noqa: E402
from projection import corrected_mu  # noqa: E402

MKT_ABBR = {"strikeouts": "K", "hits": "H", "total_bases": "TB",
            "home_runs": "HR", "rbi": "RBI", "runs": "R"}

PRED_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_log.csv")


def _rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else []


def latest_results():
    """(date, rows) for the most recent date with graded rows — the per-row
    'what happened vs what we said' table the record is built from."""
    rows = [r for r in _rows(PRED_LOG) if r.get("result") in ("1", "0", "X")]
    if not rows:
        return None, []
    rdate = max(r["date"] for r in rows)
    day = [r for r in rows if r["date"] == rdate]
    day.sort(key=lambda r: (0 if (r.get("market") or "strikeouts") == "strikeouts" else 1,
                            -float(r["model_p"])))
    return rdate, day


def accuracy_record():
    rows = [r for r in _rows(PRED_LOG)
            if (r.get("market") or "strikeouts") == "strikeouts"]
    graded = [r for r in rows if r.get("result") in ("1", "0")]

    def _cal(rs):
        n = len(rs)
        return (n, sum(float(r["model_p"]) for r in rs) / n,
                sum(1 for r in rs if r["result"] == "1") / n) if rs else None

    cal = _cal(graded)

    # raw track: the source model's own probabilities on the same rows
    raw_legs = []
    for r in graded:
        try:
            praw = float(r["raw_p_more"])
            actual, line = float(r["actual"]), float(r["line"])
        except (KeyError, TypeError, ValueError):
            continue
        if actual == line:
            continue
        raw_legs.append((max(praw, 1 - praw),
                         (actual > line) == (praw >= 0.5)))
    raw = None
    if raw_legs:
        n = len(raw_legs)
        raw = (n, sum(p for p, _ in raw_legs) / n,
               sum(1 for _, w in raw_legs if w) / n)
    return {"logged": len(rows), "graded": len(graded),
            "hit": (cal[2] * 100 if cal else 0.0),
            "cal": cal, "raw": raw}


CSS = """
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#8b949e;--pos:#3fb950;--neg:#f85149;--acc:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 2px}.sub{color:var(--mut);font-size:13px;margin-bottom:24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:0 0 18px}
h2{font-size:14px;text-transform:uppercase;letter-spacing:.5px;color:var(--mut);margin:0 0 12px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:12px}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--pos)}.neg{color:var(--neg)}.pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:12px;background:#21262d}
.kpi{display:flex;gap:24px;flex-wrap:wrap}.kpi div{min-width:90px}.kpi .v{font-size:22px;font-weight:700}.kpi .l{color:var(--mut);font-size:12px}
.status{border-radius:10px;padding:12px 16px;margin:0 0 18px;font-size:14px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.status small{display:block;font-weight:400;font-size:12px;color:var(--mut);margin-top:2px}
.toggle button{background:transparent;border:1px solid var(--line);color:var(--mut);padding:4px 12px;border-radius:6px;cursor:pointer;font-size:13px;margin-left:4px}
.toggle button.on{background:var(--acc);border-color:var(--acc);color:#fff}
@media(prefers-color-scheme:light){:root{--bg:#f6f8fa;--card:#fff;--line:#d0d7de;--fg:#1f2328;--mut:#636c76}}
"""


def tier(p):
    return ("A+" if p >= 0.62 else "A" if p >= 0.59 else "B" if p >= 0.56
            else "C" if p >= 0.53 else "D")


def _p(l): return l["p"]
def _side(l): return l["side"]


def next_slate_date(today):
    """First date from `today` with regular-season games (StatsAPI), looking
    ahead a week. None on lookup failure -> caller keeps the generic text."""
    import json as _json
    import urllib.request as _rq
    from datetime import date as _d, timedelta as _td
    try:
        d = _d.fromisoformat(today)
        for i in range(0, 8):
            ds = (d + _td(days=i)).isoformat()
            s = _json.load(_rq.urlopen(
                "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=" + ds,
                timeout=30))
            games = [g for dd in s.get("dates", []) for g in dd.get("games", [])
                     if g.get("gameType") == "R"]
            if games:
                return ds
    except Exception:
        pass
    return None


def render(date, res, tr, today=None, gen="", frozen=None, next_slate=None):
    today = today or date
    legs = sorted((l for l in res["legs"]
                   if l.get("market", "strikeouts") == "strikeouts"),
                  key=lambda l: -_p(l))

    top_rows = ""
    for l in legs[:TOP_N]:
        mkt = MKT_ABBR.get(l["market"], l["market"])
        app = PLATFORM_ABBR.get(l.get("platform", ""), l.get("platform", ""))
        pred = l.get('predicted', l['lam'])
        corr = corrected_mu(l["market"], pred)
        top_rows += (f"<tr><td>{l['name']}</td><td><span class=pill>{mkt}</span></td>"
                     f"<td class='n'>{pred:.2f}</td><td class='n'>{corr:.2f}</td>"
                     f"<td>{_side(l).upper()}</td><td class='n'>{_p(l)*100:.1f}%</td>"
                     f"<td><span class=pill>{tier(_p(l))}</span></td>"
                     f"<td class='n'>{l['line']}</td><td class='n'>{corr - l['line']:+.2f}</td></tr>")
    if not top_rows:
        top_rows = "<tr><td colspan=9 style='color:var(--mut)'>No scored rows yet for this date.</td></tr>"

    leg_rows = ""
    for l in legs:
        rw = l.get("rw_proj")
        rwp = f"{rw:.1f}" if rw is not None else "—"
        agree = {True: "<span class=pos>=</span>", False: "<span class=neg>≠</span>",
                 None: "<span style='color:var(--mut)'>·</span>"}[l.get("rw_agree")]
        grp = market_side(l["market"])
        mkt = MKT_ABBR.get(l["market"], l["market"])
        pm = l.get("p_more")
        pm_txt = f"{pm*100:.1f}%" if pm is not None else "—"
        praw = l.get("p_more_raw")
        praw_txt = f"{praw*100:.1f}%" if praw is not None else "—"
        pred = l.get('predicted', l['lam'])
        corr = corrected_mu(l["market"], pred)
        leg_rows += (f"<tr data-side='{grp}'><td>{l['name']}</td>"
                     f"<td><span class=pill>{mkt}</span></td><td>{l.get('game','')}</td>"
                     f"<td class='n'>{pred:.2f}</td><td class='n'>{corr:.2f}</td>"
                     f"<td class='n'>{pm_txt}</td>"
                     f"<td>{_side(l).upper()}</td><td class='n'>{_p(l)*100:.1f}%</td>"
                     f"<td><span class=pill>{tier(_p(l))}</span></td>"
                     f"<td class='n'>{praw_txt}</td>"
                     f"<td class='n'>{l['line']}</td><td class='n'>{corr - l['line']:+.2f}</td>"
                     f"<td class='n'>{rwp}</td><td>{agree}</td></tr>")

    def _cal_txt(c):
        return (f"stated {c[1]*100:.1f}% vs realized {c[2]*100:.1f}% "
                f"(gap {(c[2]-c[1])*100:+.1f} pts, n={c[0]})") if c else None

    cal_html = (_cal_txt(tr["cal"]) if tr.get("cal")
                else "no graded predictions yet")
    if tr.get("raw"):
        cal_html += ("<br>Raw track (source model, no anchor/ceiling): "
                     + _cal_txt(tr["raw"]))

    if date == today:
        stamp = (f"numbers frozen {frozen} (stable across rebuilds)" if frozen
                 else f"generated {gen}")
        status = (f'<div class=status>Predictions for {today}.'
                  f'<small>{stamp} · refreshed hourly · graded automatically after games go final</small></div>')
    elif next_slate and next_slate != today:
        status = (f'<div class=status>No MLB slate for {today} — next games '
                  f'{next_slate}. Showing {date}, the latest scored slate.'
                  f'<small>checked {gen} · predictions resume automatically with the next slate</small></div>')
    else:
        status = (f'<div class=status>Reference lines for {today} not captured yet — '
                  f'showing {date}, the latest scored slate.'
                  f'<small>checked {gen} · {today} appears automatically once its lines are published</small></div>')

    hitcls = "pos" if tr["cal"] and tr["cal"][2] >= tr["cal"][1] else "neg"

    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Fantasy — MLB prop projections</title><style>{CSS}</style></head><body><div class=wrap>
<h1>Fantasy · MLB prop projections</h1>
<div class=sub>Independent pitcher-strikeout projections (affine-corrected mean, Negative-Binomial, own-history probability calibration) · market lines shown as reference only · slate {date}</div>
{status}

<div class="card"><h2>Accuracy record</h2><div class=kpi>
<div><div class="v {hitcls}">{tr['hit']:.1f}%</div><div class=l>realized hit rate</div></div>
<div><div class="v">{tr['graded']}</div><div class=l>graded</div></div>
<div><div class="v">{tr['logged']}</div><div class=l>logged</div></div>
</div><div style="margin-top:10px;color:var(--mut);font-size:13px">Out-of-sample calibration: {cal_html}</div></div>

<div class=card><h2>Highest-confidence predictions today</h2><table>
<tr><th>player</th><th>prop</th><th class=n>predicted</th><th class=n>corrected</th><th>lean</th><th class=n>P</th><th>tier</th><th class=n>ref line</th><th class=n>deviation</th></tr>
{top_rows}</table></div>

<div class=card><h2>Full board — every pitcher scored</h2>
<table id=legtbl style="margin-top:12px">
<tr><th>player</th><th>prop</th><th>game</th><th class=n>predicted</th><th class=n>corrected</th><th class=n>P(more)</th><th>lean</th><th class=n>P</th><th>tier</th><th class=n>raw P(more)</th><th class=n>ref line</th><th class=n>deviation</th><th class=n>RW proj</th><th>RW</th></tr>
{leg_rows}</table><div style="margin-top:8px;color:var(--mut);font-size:12px">RW = RotoWire independent projection: = same lean · ≠ opposite lean. Probabilities are computed line-free: affine-corrected mean, fitted dispersion, and a calibration layer fitted on this system's own graded history. The reference line is shown for comparison only; the lean and deviation both follow the corrected mean, so they always agree.</div></div>

<div class="card" style="color:var(--mut);font-size:13px">Statistical projections with quantified uncertainty, graded daily against official MLB results above. Point predictions are raw model output; probabilities reflect only what settled data has supported.</div>
</div></body></html>"""


def main():
    import datetime
    today = sys.argv[1] if len(sys.argv) > 1 else "2026-07-08"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "dist", "index.html")

    # Show today's slate when its board is captured; otherwise the most recent
    # captured slate, clearly labelled with both dates.
    boards = os.path.join(os.path.dirname(__file__), "..", "data", "boards")
    if os.path.exists(os.path.join(boards, f"{today}.csv")):
        render_date = today
    else:
        avail = sorted(b[:-4] for b in (os.listdir(boards) if os.path.isdir(boards) else [])
                       if b.endswith(".csv") and not b.endswith("_batters.csv"))
        render_date = avail[-1] if avail else today

    gen = datetime.datetime.now(datetime.timezone.utc).strftime("%b %d %H:%M UTC")
    # Render from the frozen snapshot when one exists for this date — a live
    # recompute would silently drift from the logged numbers (see 7/7).
    snap = os.path.join(boards, f"{render_date}_scored.json")
    frozen = None
    if os.path.exists(snap):
        import json
        res = json.load(open(snap, encoding="utf-8"))
        frozen = res.get("frozen_at")
        # legacy snapshots: rows may lack predicted/p_more — derive them
        for l in res.get("legs", []):
            l.setdefault("predicted", l.get("lam"))
            if "p_more" not in l and "p" in l and "side" in l:
                l["p_more"] = l["p"] if l["side"] == "more" else 1.0 - l["p"]
    else:
        res = compute_board(render_date)
    tr = accuracy_record()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(render_date, res, tr, today=today, gen=gen, frozen=frozen,
                       next_slate=(next_slate_date(today) if render_date != today else None)))
    print(f"wrote {out}  showing {render_date} ({len(res['legs'])} scored rows)"
          + (f"  [frozen {frozen}]" if frozen else "  [live scoring]"))


if __name__ == "__main__":
    main()
