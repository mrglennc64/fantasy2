"""Generate a self-contained static dashboard for fantasy.perfecthold.online.

Renders today's Pick6 entries + leg scores and the paper track record (ROI +
out-of-sample calibration) into a single inline-styled index.html that nginx can
serve statically. Run at build/deploy time (it hits the live slate once):

    python build_site.py 2026-07-05 ../web/dist/index.html

No serve-time dependencies — pure static output.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pick6"))
from pick6_today import PLATFORM_ABBR, compute_entries  # noqa: E402
from markets import market_side  # noqa: E402

MKT_ABBR = {"strikeouts": "K", "hits": "H", "total_bases": "TB",
            "home_runs": "HR", "rbi": "RBI", "runs": "R"}

ENTRIES_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "pick6_entries.csv")


def _rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else []


def track_record():
    rows = _rows(ENTRIES_LOG)
    entries = {}
    for r in rows:
        entries.setdefault(r["entry_id"], []).append(r)
    staked = pnl = won = graded = 0.0
    for legs in entries.values():
        if any(l["leg_won"] == "" for l in legs):
            continue
        graded += 1
        stake, mult = float(legs[0]["stake"]), float(legs[0]["mult"])
        w = all(l["leg_won"] == "1" for l in legs)
        pnl += stake * (mult - 1) if w else -stake
        staked += stake
        won += w
    graded_legs = [((r.get("market") or "strikeouts"), float(r["model_p"]),
                    r["leg_won"] == "1")
                   for r in rows if r["leg_won"] in ("1", "0")]

    def _cal(legs):
        n = len(legs)
        return (n, sum(p for p, _ in legs) / n,
                sum(1 for _, w in legs if w) / n) if legs else None

    cal = _cal([(p, w) for m, p, w in graded_legs])
    cal_k = _cal([(p, w) for m, p, w in graded_legs if m == "strikeouts"])
    cal_bat = _cal([(p, w) for m, p, w in graded_legs if m != "strikeouts"])
    return {"staked": staked, "pnl": pnl, "won": won, "graded": graded,
            "roi": (pnl / staked * 100 if staked else 0), "pending": len(entries),
            "cal": cal, "cal_k": cal_k, "cal_bat": cal_bat}


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
.warn{background:#2d2212;border-color:#5c4813;color:#e3b341;font-size:13px}
.banner{border-radius:10px;padding:12px 16px;margin:0 0 18px;font-size:15px;font-weight:600;border:1px solid}
.banner.live{background:#122117;border-color:#238636;color:#3fb950}
.banner.wait{background:#2d2212;border-color:#9e6a03;color:#e3b341}
.banner small{display:block;font-weight:400;font-size:12px;opacity:.85;margin-top:2px}
.toggle button{background:transparent;border:1px solid var(--line);color:var(--mut);padding:4px 12px;border-radius:6px;cursor:pointer;font-size:13px;margin-left:4px}
.toggle button.on{background:var(--acc);border-color:var(--acc);color:#fff}
@media(prefers-color-scheme:light){:root{--bg:#f6f8fa;--card:#fff;--line:#d0d7de;--fg:#1f2328;--mut:#636c76}}
"""


def render(date, res, tr, status="live", today=None, gen="", frozen=None):
    today = today or date
    legs = sorted(res["legs"], key=lambda l: -_p(l))
    entry_rows = ""
    for i, e in enumerate(res["entries"], 1):
        names = " + ".join(f"{l['name'].split()[-1]} {l['side'][0].upper()}{l['line']}" for l in e["legs"])
        if e.get("same_side"):
            names += " <span class=pill>same-side</span>"
        ev = e.get("corr_ev", e["ev"]); pw = e.get("corr_p", e["p"])
        cls = "pos" if ev > 0 else "neg"
        app = PLATFORM_ABBR.get(e.get("platform", ""), e.get("platform", ""))
        entry_rows += (f"<tr><td>{i}</td><td><span class=pill>{app}</span> "
                       f"<span class=pill>{e.get('n','')}-pick</span> {names}</td>"
                       f"<td class='n'>{pw*100:.1f}%</td><td class='n'>{e['mult']:.1f}×</td>"
                       f"<td class='n {cls}'>{ev*100:+.0f}%</td><td class='n'>${e['stake']:.2f}</td></tr>")
    if not entry_rows:
        entry_rows = "<tr><td colspan=6 style='color:var(--mut)'>No entry clears breakeven today.</td></tr>"

    leg_rows = ""
    for l in legs:
        keep = "✓" if _kept(l, res) else ""
        rw = l.get("rw_proj")
        rwp = f"{rw:.1f}" if rw is not None else "—"
        agree = {True: "<span class=pos>✓</span>", False: "<span class=neg>✗</span>",
                 None: "<span style='color:var(--mut)'>·</span>"}[l.get("rw_agree")]
        grp = market_side(l["market"])
        mkt = MKT_ABBR.get(l["market"], l["market"])
        leg_rows += (f"<tr data-side='{grp}'><td>{l['name']}</td>"
                     f"<td><span class=pill>{mkt}</span></td><td>{l.get('game','')}</td>"
                     f"<td class='n'>{l['line']}</td><td class='n'>{l['lam']:.2f}</td>"
                     f"<td>{_side(l).upper()}</td><td class='n'>{_p(l)*100:.1f}%</td>"
                     f"<td class='n'>{rwp}</td><td>{agree}</td>"
                     f"<td style='color:var(--pos)'>{keep}</td></tr>")

    def _cal_txt(c):
        return (f"predicted {c[1]*100:.1f}% vs realized {c[2]*100:.1f}% "
                f"(gap {(c[2]-c[1])*100:+.1f} pts, n={c[0]})") if c else None

    parts = [t for t in (
        _cal_txt(tr["cal"]),
        tr.get("cal_k") and "pitchers: " + _cal_txt(tr["cal_k"]),
        tr.get("cal_bat") and "batters: " + _cal_txt(tr["cal_bat"])) if t]
    cal_html = " · ".join(parts) if parts else "no graded legs yet"
    roicls = "pos" if tr["pnl"] >= 0 else "neg"

    if status == "live":
        lock = (f'picks locked {frozen} (won\'t change on rebuilds)' if frozen
                else f'updated {gen}')
        banner = (f'<div class="banner live">🟢 LIVE — today\'s board ({today}) is up. '
                  f'These are today\'s picks.<small>{lock} · bet before first pitch</small></div>')
    else:
        banner = (f'<div class="banner wait">🟠 WAITING — today\'s ({today}) board isn\'t '
                  f'posted yet. Below is the most recent card ({date}), not today\'s.'
                  f'<small>checked {gen} · today\'s picks appear here automatically once the board goes live</small></div>')

    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Fantasy — pick'em edge</title><style>{CSS}</style></head><body><div class=wrap>
<h1>Fantasy · pick'em edge</h1>
<div class=sub>Pitcher strikeouts (calibrated) + batter props (StatsAPI matchup-adjusted baseline) · showing slate {date} · PAPER ONLY</div>
{banner}

<div class="card"><h2>Paper track record</h2><div class=kpi>
<div><div class="v {roicls}">{tr['roi']:+.1f}%</div><div class=l>ROI</div></div>
<div><div class="v">{int(tr['won'])}/{int(tr['graded'])}</div><div class=l>entries won</div></div>
<div><div class="v {roicls}">${tr['pnl']:+.2f}</div><div class=l>net P&amp;L</div></div>
<div><div class="v">{tr['pending']}</div><div class=l>entries logged</div></div>
</div><div style="margin-top:10px;color:var(--mut);font-size:13px">Out-of-sample leg calibration: {cal_html}</div></div>

<div class=card><h2>Today's power-play entries</h2><table>
<tr><th>#</th><th>legs</th><th class=n>P(win)</th><th class=n>mult</th><th class=n>EV</th><th class=n>stake</th></tr>
{entry_rows}</table></div>

<div class=card><div style="display:flex;justify-content:space-between;align-items:center">
<h2 style="margin:0">All board legs scored</h2>
<div class=toggle><button id=tb-pitcher class=on onclick="flt('pitcher')">Pitchers</button><button id=tb-batter onclick="flt('batter')">Batters</button><button id=tb-all onclick="flt('all')">All</button></div></div>
<table id=legtbl style="margin-top:12px">
<tr><th>player</th><th>prop</th><th>game</th><th class=n>DK line</th><th class=n>λ</th><th>pick</th><th class=n>model P</th><th class=n>RW proj</th><th>RW</th><th>play</th></tr>
{leg_rows}</table><div style="margin-top:8px;color:var(--mut);font-size:12px">RW = RotoWire second opinion: ✓ agrees · ✗ disagrees (gated out) · · no free projection. Batter props use a StatsAPI season-rate baseline adjusted for the opposing starter + platoon split; no fitted dispersion yet, so batter model P is capped at 70%.</div></div>
<script>
function flt(g){{document.querySelectorAll('#legtbl tr[data-side]').forEach(function(r){{r.style.display=(g==='all'||r.dataset.side===g)?'':'none';}});
['pitcher','batter','all'].forEach(function(k){{document.getElementById('tb-'+k).className=(k===g)?'on':'';}});}}
flt('pitcher');
</script>

<div class="card warn">⚠️ Paper only. The model's dispersion was fit in-sample; entries are staked
hypothetically at quarter-Kelly. Nothing here is betting advice. Verify every DK line and multiplier before acting.</div>
</div></body></html>"""


# score legs the same way the picker does, for display. Snapshot legs already
# carry frozen side/p (from log time) — those win, so the page can't re-pick.
from sim import score_leg  # noqa: E402


def _s(l):
    if "_scored" not in l:
        l["_scored"] = score_leg(l)
    return l["_scored"]


def _p(l): return l["p"] if "p" in l else _s(l)["p"]
def _side(l): return l["side"] if "side" in l else _s(l)["side"]


def _kept(l, res):
    return any(l["name"] == x["name"] for e in res["entries"] for x in e["legs"])


def main():
    import datetime
    today = sys.argv[1] if len(sys.argv) > 1 else "2026-07-05"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(__file__), "dist", "index.html")

    # LIVE if today's board exists; otherwise WAITING — render the most recent
    # captured board so the page always shows a real card, clearly labelled.
    boards = os.path.join(os.path.dirname(__file__), "..", "data", "boards")
    if os.path.exists(os.path.join(boards, f"{today}.csv")):
        render_date, status = today, "live"
    else:
        avail = sorted(b[:-4] for b in (os.listdir(boards) if os.path.isdir(boards) else [])
                       if b.endswith(".csv") and not b.endswith("_batters.csv"))
        render_date, status = (avail[-1] if avail else today), "waiting"

    gen = datetime.datetime.now(datetime.timezone.utc).strftime("%b %d %H:%M UTC")
    # Render from the frozen snapshot when entries were logged for this date —
    # a live recompute would silently re-pick with drifted inputs (see 7/7).
    snap = os.path.join(boards, f"{render_date}_scored.json")
    frozen = None
    if os.path.exists(snap):
        import json
        res = json.load(open(snap, encoding="utf-8"))
        frozen = res.get("frozen_at")
    else:
        res = compute_entries(render_date)
    tr = track_record()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(render_date, res, tr, status=status, today=today, gen=gen,
                       frozen=frozen))
    print(f"wrote {out}  [{status}] showing {render_date} "
          f"({len(res['legs'])} legs, {len(res['entries'])} entries)"
          + (f"  [frozen {frozen}]" if frozen else "  [live recompute]"))


if __name__ == "__main__":
    main()
