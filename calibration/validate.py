"""Does the strikeout projection carry information? (Phase 0)

    python validate.py

WHY THIS EXISTS. pick6/calibrate.py serves beta=0 — every pitcher row states
the same 53.0% — because refit_calibration.py found no demonstrated ranking
skill. But that verdict rests on ~119 board rows. The same question can be
asked of ~1,900 HELD-OUT gamelog starts, 16x the data, and it needs no book
lines to answer. Answering it is the precondition for every later phase: there
is no point calibrating, ensembling, or simulating a ranking that does not
discriminate.

FOUR QUESTIONS, in increasing order of difficulty:

  1. ACCURACY   — is the mean right? (MAE, bias)
  2. RANKING    — does it order pitchers correctly? (Spearman rho)
                  This is the one beta=0 is really about. A model can be
                  perfectly unbiased and still rank at chance, in which case
                  no calibration will ever produce useful confidence.
  3. INTERVALS  — is NB(mu, r) honest? (PIT coverage)
  4. THRESHOLDS — does it beat a NAIVE baseline at a decision boundary?

Question 4 is the honest analog of beating a line, and the baseline choice is
what makes it honest. Scoring against a fixed threshold (say 5.5 for everyone)
is trivial — a model only needs to know that aces strike out more than
back-end starters, which season-to-date K/BF already tells you. So the
threshold here is each pitcher's OWN naive projection: prior K/BF times
expected batters faced. Clearing 50% against that means the model knows
something beyond the pitcher's own recent form, which is the only thing that
could ever beat a market.

A real line is harder still — it aggregates information neither model has —
so section 5 scores the 486 real logged lines separately. Small, but real.

Stdlib only, so it runs on the VPS as well as the laptop.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from datetime import date as _date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pick6"))
from dispersion import DISPERSION_R                      # noqa: E402
from feed import norm                                    # noqa: E402
from kmodel_params import COEF, FEATURES, INTERCEPT, MEAN, PARKS, SD  # noqa: E402

GAMELOGS = r"C:\Users\carin\OneDrive\Dokument\stike\mlb-edge\data\all_starters_gamelogs_2024_2026.csv"
LINES = r"C:\Users\carin\OneDrive\Dokument\stike\mlb-edge\data\exports\vps\predictions.csv"
VAL_FROM = "2026-05-01"          # same split train_kmodel.py held out
PRIOR_BF = 150.0
_EPS = 1e-9


# ---- distribution helpers ----------------------------------------------------

def nb_pmf(k: int, mu: float, r: float = DISPERSION_R) -> float:
    mu = max(mu, _EPS)
    return math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                    + r * math.log(r / (r + mu)) + k * math.log(mu / (r + mu)))


def p_over(mu: float, line: float) -> float:
    """P(K > line) for a half-integer line."""
    return max(0.0, 1.0 - sum(nb_pmf(i, mu) for i in range(math.ceil(line))))


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, average ranks for ties. Pearson on ranks."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                rk[order[t]] = avg
            i = j + 1
        return rk

    a, b = ranks(xs), ranks(ys)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else 0.0


# ---- rebuild the served model's features, strictly from prior starts --------

def _truthy(v: str) -> float:
    """is_home ships as the string 'True'/'False', not 0/1."""
    return 1.0 if str(v).strip().lower() in ("true", "1", "t", "yes") else 0.0


def load_rows() -> list[dict]:
    out, rejected = [], 0
    for r in csv.DictReader(open(GAMELOGS, encoding="utf-8")):
        try:
            bf = float(r["BF"])
            if bf <= 0:
                continue
            home = _truthy(r["is_home"])
            out.append({"date": r["date"], "pid": r["pitcher_id"],
                        "pitcher": r["pitcher"], "K": int(float(r["K"])),
                        "BF": bf, "is_home": home,
                        "opp_k_pct": float(r["opp_k_pct"]) if r.get("opp_k_pct") else None,
                        # Park factor keys on the VENUE, which is the home
                        # club's park — the pitcher's own team only when he is
                        # at home.
                        "venue": r.get("team" if home else "opponent", "")})
        except (ValueError, KeyError):
            rejected += 1
    # Loud, because a silent reject-everything is how this function returned 0
    # rows on its first run: float('True') raised and the except ate all 11,796.
    if rejected:
        print(f"  WARNING: {rejected} gamelog rows rejected as unparseable")
    if not out:
        raise SystemExit("load_rows(): every row rejected — check the schema, "
                         "do not treat this as 'no data'.")
    out.sort(key=lambda r: (r["date"], r["pid"]))
    return out


def build(rows: list[dict]) -> list[dict]:
    """Attach strictly-prior features — same construction as train_kmodel.py,
    so this validates the model that is actually served rather than a variant."""
    hist: dict[str, list[dict]] = defaultdict(list)
    tot_k = tot_bf = 0.0
    out = []
    for r in rows:
        h = hist[r["pid"]]

        def _kbf(last):
            hh = h if last is None else h[-last:]
            k = sum(x["K"] for x in hh)
            bf = sum(x["BF"] for x in hh)
            base = tot_k / tot_bf if tot_bf > 500 else 0.22
            return (k + PRIOR_BF * base) / (bf + PRIOR_BF)

        if len(h) >= 2:
            rest = (_date.fromisoformat(r["date"])
                    - _date.fromisoformat(h[-1]["date"])).days
            roll_bf_3 = sum(x["BF"] for x in h[-3:]) / min(len(h), 3)
            feats = {"roll_kbf_3": _kbf(3), "roll_kbf_10": _kbf(10),
                     "prior_kbf": _kbf(None), "roll_bf_3": roll_bf_3,
                     "opp_k_pct": r["opp_k_pct"], "is_home": r["is_home"],
                     "rest_days": float(min(max(rest, 3), 12)),
                     "park_k": PARKS.get(r.get("venue"), 1.0)}
            z = INTERCEPT
            for f, m, s, c in zip(FEATURES, MEAN, SD, COEF):
                x = feats.get(f)
                z += c * ((m if x is None else x) - m) / s
            # NAIVE baseline: the pitcher's own recent form, nothing else. The
            # threshold the model has to beat to have earned its features.
            out.append({**r, "mu": math.exp(z),
                        "naive": _kbf(None) * roll_bf_3})
        h.append(r)
        tot_k += r["K"]
        tot_bf += r["BF"]
    return out


# ---- report ------------------------------------------------------------------

def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    if not os.path.exists(GAMELOGS):
        print(f"gamelogs not found: {GAMELOGS}")
        return
    rows = build(load_rows())
    val = [r for r in rows if r["date"] >= VAL_FROM]
    print(f"{len(rows)} starts with features; {len(val)} held out (>= {VAL_FROM})")
    if len(val) < 200:
        print("held-out set too small to conclude anything.")
        return

    mus = [r["mu"] for r in val]
    ks = [float(r["K"]) for r in val]

    # 1. accuracy
    section("1. ACCURACY — is the mean right?")
    errs = [k - m for m, k in zip(mus, ks)]
    mae = sum(abs(e) for e in errs) / len(errs)
    bias = sum(errs) / len(errs)
    rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    nmae = sum(abs(k - r["naive"]) for r, k in zip(val, ks)) / len(val)
    print(f"  model  MAE {mae:.3f} K   bias {bias:+.3f} K   RMSE {rmse:.3f}")
    print(f"  naive  MAE {nmae:.3f} K   (prior K/BF x recent BF)")
    print(f"  -> model is {100*(nmae-mae)/nmae:+.1f}% better than naive on MAE")

    # 2. ranking — the question beta=0 is really about
    section("2. RANKING — does it order pitchers correctly?")
    rho = spearman(mus, ks)
    rho_n = spearman([r["naive"] for r in val], ks)
    print(f"  Spearman rho (model vs actual K):  {rho:+.4f}")
    print(f"  Spearman rho (naive vs actual K):  {rho_n:+.4f}")
    print(f"  n = {len(val)} held-out starts")
    se = 1.0 / math.sqrt(len(val) - 1)
    print(f"  approx SE {se:.4f} -> rho is {abs(rho)/se:.1f} SE from zero")
    if rho < 0.05:
        print("  VERDICT: no usable discrimination. Calibration cannot fix this;")
        print("           the features must improve before confidence returns.")
    else:
        print(f"  VERDICT: the ranking carries information (rho {rho:.3f}).")
        print("           If the BOARD still shows none, the defect is in the")
        print("           serving path, not the model — check data/features/.")

    # 3. intervals
    section("3. INTERVALS — is NB(mu, r) honest?")
    pits = []
    for m, k in zip(mus, ks):
        below = sum(nb_pmf(i, m) for i in range(int(k)))
        pits.append(below + 0.5 * nb_pmf(int(k), m))
    c50 = sum(1 for p in pits if 0.25 <= p <= 0.75) / len(pits)
    c80 = sum(1 for p in pits if 0.10 <= p <= 0.90) / len(pits)
    print(f"  central-50 coverage {c50*100:5.1f}%  (nominal 50%)")
    print(f"  central-80 coverage {c80*100:5.1f}%  (nominal 80%)")
    print(f"  r = {DISPERSION_R}")
    if c50 < 0.45 or c80 < 0.75:
        print("  -> intervals too NARROW: r is too high (overconfident spread)")
    elif c50 > 0.56 or c80 > 0.86:
        print("  -> intervals too WIDE: r is too low (underconfident spread)")
    else:
        print("  -> coverage is nominal; dispersion is honest at this mu range")

    # 4. thresholds vs the naive baseline
    section("4. THRESHOLDS — does it beat each pitcher's own recent form?")
    print("  Threshold = that pitcher's naive projection, rounded to .5.")
    print("  Beating 50% here means the model knows something beyond form.\n")
    buckets = defaultdict(lambda: [0, 0, 0.0])
    won = tot = 0
    for r, k in zip(val, ks):
        line = round(r["naive"] * 2) / 2
        if line == int(line):
            line += 0.5                        # avoid exact-integer pushes
        if k == line:
            continue
        pm = p_over(r["mu"], line)
        side_more = pm >= 0.5
        p = pm if side_more else 1 - pm
        ok = side_more == (k > line)
        won += ok
        tot += 1
        b = min(int(p * 20) / 20, 0.75)
        buckets[b][0] += 1
        buckets[b][1] += ok
        buckets[b][2] += p
    print(f"  overall {won}/{tot} = {100*won/tot:.1f}%")
    print(f"\n  {'stated':>8} {'n':>6} {'realized':>10} {'gap':>8}")
    for b in sorted(buckets):
        n, w, sp = buckets[b]
        if n < 25:
            continue
        print(f"  {sp/n*100:7.1f}% {n:>6} {100*w/n:9.1f}% {100*(w/n - sp/n):+7.1f}p")

    # 5. real lines
    section("5. REAL LINES — the actual bar")
    if not os.path.exists(LINES):
        print("  no line archive found; skipping.")
        return
    lines = {}
    for r in csv.DictReader(open(LINES, encoding="utf-8")):
        if r.get("line") and r.get("date"):
            lines[(r["date"], norm(r["pitcher"]))] = float(r["line"])
    idx = {(r["date"], norm(r["pitcher"])): r for r in rows}
    hit = n = 0
    for key, line in lines.items():
        r = idx.get(key)
        if r is None or r["K"] == line:
            continue
        pm = p_over(r["mu"], line)
        hit += (pm >= 0.5) == (r["K"] > line)
        n += 1
    if n:
        print(f"  {hit}/{n} = {100*hit/n:.1f}% on real book lines")
        print("  (a real line aggregates information the model does not have,")
        print("   so this is the hardest of the five and the only one that")
        print("   speaks to market performance)")
    else:
        print("  no overlap between the line archive and featured starts.")


if __name__ == "__main__":
    main()
