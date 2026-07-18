"""Grade logged predictions against real MLB results.

    python grade.py

Reads data/predictions_log.csv, fills actual + result for any ungraded row
whose game is Final (MLB StatsAPI boxscores), rewrites the file, then prints
the accuracy record:
  - hit rate: how often the model's lean was on the correct side of the line
  - calibration: stated probability vs realized frequency, split pitcher vs
    batter — the OUT-OF-SAMPLE check on the model; if these buckets drift,
    re-fit (dispersion for spread, projection.py shrink for the mean).

First run migrates the legacy per-leg history (data/pick6_entries.csv) into
the new log so the accuracy record keeps its continuity.
"""
from __future__ import annotations

import csv
import json
import os
import unicodedata
import urllib.request
from collections import defaultdict

import atomicio
import params
from log_schema import FIELDS, SRC_LEGACY, backfill, ensure_schema

LOG = os.path.join(os.path.dirname(__file__), "..", "data", "predictions_log.csv")
LEGACY = os.path.join(os.path.dirname(__file__), "..", "data", "pick6_entries.csv")


def norm(name: str) -> str:
    nk = unicodedata.normalize("NFKD", name)
    nk = "".join(c for c in nk if not unicodedata.combining(c))
    return "".join(c for c in nk.lower() if c.isalpha() or c == " ").strip()


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


# market -> (boxscore stat group, field)
_STAT = {
    "strikeouts":  ("pitching", "strikeOuts"),
    "hits":        ("batting", "hits"),
    "total_bases": ("batting", "totalBases"),
    "home_runs":   ("batting", "homeRuns"),
    "rbi":         ("batting", "rbi"),
    "runs":        ("batting", "runs"),
}


def final_stats(date: str) -> dict[str, dict[str, int]]:
    """norm(player) -> {market: actual} for FINAL games on date (pitching + batting)."""
    sched = _get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}")
    out: dict[str, dict[str, int]] = {}
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            try:
                box = _get(f"https://statsapi.mlb.com/api/v1/game/{g['gamePk']}/boxscore")
            except Exception:
                continue
            for side in ("home", "away"):
                for pdata in box["teams"][side]["players"].values():
                    stats = pdata.get("stats", {})
                    rec = {}
                    for market, (grp, field) in _STAT.items():
                        v = stats.get(grp, {}).get(field)
                        if v is not None:
                            rec[market] = int(v)
                    if rec:
                        out[norm(pdata["person"]["fullName"])] = rec
    return out


def result_of(side: str, line: float, actual: int) -> str:
    """'1' (lean correct) / '0' (incorrect) / 'X' (actual landed exactly on a
    whole-number line — no side of the line to be on; excluded from rates)."""
    if actual == line and float(line).is_integer():
        return "X"
    correct = actual > line if side == "more" else actual < line
    return "1" if correct else "0"


def migrate_legacy() -> None:
    """One-time: fold the legacy per-leg history into predictions_log.csv.
    Legacy rows repeat a leg once per set it appeared in — dedupe on
    (date, player, market, line, side)."""
    if os.path.exists(LOG) or not os.path.exists(LEGACY):
        return
    seen, rows = set(), []
    for r in csv.DictReader(open(LEGACY, encoding="utf-8")):
        key = (r["date"], r["pitcher"], r.get("market", "strikeouts"),
               r["line"], r["side"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "date": r["date"], "player": r["pitcher"], "game": r["game"],
            "market": r.get("market", "strikeouts"),
            "platform": r.get("platform", ""), "side": r["side"],
            "line": r["line"], "predicted": r["lam"], "model_p": r["model_p"],
            "rw_proj": r.get("rw_proj", ""), "rw_agree": r.get("rw_agree", ""),
            "actual": r.get("actual_ks", ""),
            "result": {"1": "1", "0": "0", "P": "X"}.get(r.get("leg_won", ""), ""),
            # All legacy history predates the mlb-edge cutover, so this is a
            # fact about those rows rather than the "unknown" default.
            "mu_source": SRC_LEGACY, "mu_version": "", "model_p_uncal": "",
        })
    atomicio.write_rows(LOG, FIELDS, rows)
    print(f"Migrated {len(rows)} legacy rows -> {LOG}")


def main() -> None:
    migrate_legacy()
    if not os.path.exists(LOG):
        print(f"No log at {LOG} — run log_predictions.py first.")
        return
    ensure_schema(LOG)
    rows = backfill(list(csv.DictReader(open(LOG, encoding="utf-8"))))

    pending_dates = sorted({r["date"] for r in rows if r["result"] == ""})
    results = {d: final_stats(d) for d in pending_dates}
    graded_now = 0
    for r in rows:
        if r["result"] != "":
            continue
        market = r.get("market", "strikeouts")
        actual = results.get(r["date"], {}).get(norm(r["player"]), {}).get(market)
        if actual is None:
            continue  # game not Final yet (or player didn't play) — leave pending
        r["actual"] = actual
        r["result"] = result_of(r["side"], float(r["line"]), actual)
        graded_now += 1

    atomicio.write_rows(LOG, FIELDS, rows)
    print(f"Graded {graded_now} new predictions.\n")

    graded = [r for r in rows if r["result"] in ("1", "0")]
    if not graded:
        print("No graded predictions yet.")
        return

    # ---- accuracy record ----------------------------------------------------
    def _rate(rs):
        n = len(rs)
        hit = sum(1 for r in rs if r["result"] == "1")
        pred = sum(float(r["model_p"]) for r in rs) / n
        return n, hit / n, pred

    print("ACCURACY RECORD (out-of-sample; exact-on-line outcomes excluded)")
    # PITCHERS ONLY in the report (2026-07-13); batter history stays in the
    # CSV and keeps grading for continuity, but is no longer reported.
    groups = [("pitcher (K)", [r for r in graded
                               if r.get("market", "strikeouts") == "strikeouts"])]
    for tag, rs in [(t, g) for t, g in groups if g]:
        n, hit, pred = _rate(rs)
        print(f"  {tag:12} n={n:<4} stated {pred*100:.1f}%  "
              f"realized {hit*100:.1f}%  gap {(hit-pred)*100:+.1f} pts")

    # ---- dual-track: the RAW projection's own probabilities, same rows -----
    def _raw_rate(rs):
        legs = []
        for r in rs:
            try:
                praw = float(r["raw_p_more"])
            except (KeyError, TypeError, ValueError):
                continue
            actual, line = float(r["actual"]), float(r["line"])
            if actual == line:
                continue
            won = (actual > line) == (praw >= 0.5)
            legs.append((max(praw, 1 - praw), won))
        if not legs:
            return None
        n = len(legs)
        return n, sum(1 for _, w in legs if w) / n, sum(p for p, _ in legs) / n

    print("\nRAW TRACK (no anchor, no ceiling — the source model on its own)")
    for tag, rs in [(t, g) for t, g in groups if g]:
        rr = _raw_rate(rs)
        if rr is None:
            print(f"  {tag:12} (no raw-track rows yet)")
            continue
        n, hit, pred = rr
        print(f"  {tag:12} n={n:<4} stated {pred*100:.1f}%  "
              f"realized {hit*100:.1f}%  gap {(hit-pred)*100:+.1f} pts")

    # calibration buckets: stated probability vs realized frequency
    print("\nCALIBRATION BUCKETS")
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]
    for lo, hi in bins:
        grp = [r for r in graded if lo <= float(r["model_p"]) < hi]
        if not grp:
            continue
        n, hit, pred = _rate(grp)
        print(f"  {lo:.2f}-{hi:4.2f}  n={n:<4} stated {pred*100:.1f}%  "
              f"realized {hit*100:.1f}%  gap {(hit-pred)*100:+.1f} pts")
    # ---- proper scoring metrics (Brier / log-loss) + point accuracy --------
    import math as _m
    pit = [r for r in graded if r.get("market", "strikeouts") == "strikeouts"]
    ps = [(float(r["model_p"]), r["result"] == "1") for r in pit]
    if ps:
        brier = sum((p - (1.0 if w else 0.0)) ** 2 for p, w in ps) / len(ps)
        ll = -sum(_m.log(max(p if w else 1 - p, 1e-9)) for p, w in ps) / len(ps)
        errs = [abs(float(r["predicted"]) - float(r["actual"])) for r in pit
                if r.get("predicted") and r.get("actual") != ""]
        mae = sum(errs) / len(errs)
        rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5
        print(f"\nSCORING METRICS (pitcher)  Brier {brier:.4f}   log-loss {ll:.4f}"
              f"   projection MAE {mae:.2f} K   RMSE {rmse:.2f} K")
        print("  (baselines: coin-flip Brier 0.2500, log-loss 0.6931 — must beat"
              " both for the probabilities to carry information)")

    # ---- confidence tiers (labels only — grouping, not actions) ------------
    TIERS = [("A+", 0.62, 1.01), ("A", 0.59, 0.62), ("B", 0.56, 0.59),
             ("C", 0.53, 0.56), ("D", 0.00, 0.53)]
    print("\nCONFIDENCE TIERS (pitcher)")
    for tag, lo, hi in TIERS:
        grp = [(p, w) for p, w in ps if lo <= p < hi]
        if not grp:
            continue
        n = len(grp)
        pred = sum(p for p, _ in grp) / n
        real = sum(1 for _, w in grp if w) / n
        print(f"  {tag:3} [{lo*100:.0f}-{hi*100:.0f}%)  n={n:<4} "
              f"stated {pred*100:.1f}%  realized {real*100:.1f}%")

    # ---- BY SOURCE: which estimator produced the mu, and how did it do? ----
    # The headline diagnostic. Each source carries its own mean correction, so
    # a source-blind average hides exactly the failure this split exists to
    # catch: between 2026-07-13 and 07-18 the feed fell through to the owned
    # kmodel while scoring kept applying the mlb-edge affine to it. Aggregate
    # bias barely moved (the affine's fixed point sits mid-range) but the
    # kmodel rows alone would have shown it inside two days.
    def _bias(rs):
        e = [float(r["actual"]) - float(r["predicted"]) for r in rs
             if r.get("predicted") and r.get("actual") not in ("", None)]
        return (sum(e) / len(e), sum(abs(x) for x in e) / len(e)) if e else (None, None)

    by_src = defaultdict(list)
    for r in pit:
        by_src[r.get("mu_source") or "unknown"].append(r)
    if len(by_src) > 1 or "unknown" not in by_src:
        print("\nBY PROJECTION SOURCE (bias is actual - predicted, in K)")
        print(f"  {'source':<18} {'n':>4} {'stated':>7} {'realized':>9} "
              f"{'bias':>7} {'MAE':>6} {'Brier':>7}")
        for src in sorted(by_src, key=lambda s: -len(by_src[s])):
            rs = by_src[src]
            n, hit, pred = _rate(rs)
            bias, mae = _bias(rs)
            sp = [(float(r["model_p"]), r["result"] == "1") for r in rs]
            br = sum((p - (1.0 if w else 0.0)) ** 2 for p, w in sp) / len(sp)
            print(f"  {src:<18} {n:>4} {pred*100:>6.1f}% {hit*100:>8.1f}% "
                  f"{bias:>+7.2f} {mae:>6.2f} {br:>7.4f}"
                  if bias is not None else
                  f"  {src:<18} {n:>4} {pred*100:>6.1f}% {hit*100:>8.1f}% "
                  f"{'—':>7} {'—':>6} {br:>7.4f}")

    # ---- bias by projection range: is a correction wrong at the extremes? ---
    # A global mean can look unbiased while the slope is wrong, which is the
    # damage a bad correction actually does — it compresses or inflates the
    # spread between pitchers without moving the average.
    withmu = [r for r in pit if r.get("predicted") and r.get("actual") not in ("", None)]
    if len(withmu) >= 30:
        srt = sorted(withmu, key=lambda r: float(r["predicted"]))
        k = len(srt) // 3
        print("\nBIAS BY PROJECTION RANGE (a slope error hides in the average)")
        for tag, chunk in (("low  ", srt[:k]), ("mid  ", srt[k:2 * k]),
                           ("high ", srt[2 * k:])):
            if not chunk:
                continue
            b_, m_ = _bias(chunk)
            mus = sum(float(r["predicted"]) for r in chunk) / len(chunk)
            print(f"  {tag} mean mu {mus:5.2f}   bias {b_:+.2f} K   "
                  f"MAE {m_:.2f}   (n={len(chunk)})")
        lo_b = _bias(srt[:k])[0]
        hi_b = _bias(srt[2 * k:])[0]
        if lo_b is not None and hi_b is not None and abs(hi_b - lo_b) > 1.0:
            print(f"  => low/high bias differ by {abs(hi_b-lo_b):.2f} K: the "
                  f"SLOPE is off, not the level.\n     Re-fit per source "
                  f"(calibration/fit_mean.py) rather than shifting the mean.")

    # ---- daily error analysis: biggest projection misses, latest slate -----
    if pit:
        last = max(r["date"] for r in pit)
        day = sorted((r for r in pit if r["date"] == last),
                     key=lambda r: -abs(float(r["predicted"]) - float(r["actual"])))
        print(f"\nLARGEST PROJECTION MISSES — {last}")
        for r in day[:5]:
            print(f"  {r['player']:22} predicted {float(r['predicted']):5.2f}  "
                  f"line {r['line']:>4}  actual {r['actual']:>2}  "
                  f"({'lean correct' if r['result'] == '1' else 'lean wrong'})")

    # ---- what is actually live right now -----------------------------------
    # Constants no longer live only in git, so the daily report has to say what
    # is serving. Otherwise "which correction produced this record?" becomes
    # unanswerable the moment a refit promotes.
    from projection import CORRECTIONS
    from calibrate import GROUPS
    from dispersion import DISPERSION_R
    print(f"\nLIVE CONSTANTS  ({params.describe()})")
    print("  mean correction  " + "   ".join(
        f"{s}: {a:+.2f}{b:+.2f}mu" for s, (a, b) in sorted(CORRECTIONS.items())))
    print(f"  calibration      pitcher: alpha={GROUPS['pitcher'][0]:+.3f} "
          f"beta={GROUPS['pitcher'][1]:.2f}"
          + ("   [beta=0 -> every row states the base rate; the confidence "
             "ranking is OFF]" if GROUPS["pitcher"][1] == 0 else ""))
    print(f"  dispersion       r={DISPERSION_R}")

    print("\n(pitcher drift at scale => re-fit the mean PER SOURCE on FROZEN"
          "\n slates (calibration/fit_mean.py); confidence spread returns when"
          "\n the weekly calibration refit finds ranking skill)")


if __name__ == "__main__":
    main()
