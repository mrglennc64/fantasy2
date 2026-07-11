"""Fit a MEAN correction for the strikeout projection (mu = expected_ks).

    python fit_mean.py

Why: 7/5 + 7/7 forward legs show pitcher K legs predicted ~70% vs realized
~47-56% and WORSENING. refit_dispersion.py already re-validated r=16.6 on
frozen data — so the gap is a bias in mu itself, not dispersion. r cannot fix
a mean error; this fits the mean directly.

Data (leakage-free ONLY — every mu was logged BEFORE its game):
  1. mlb-edge exports/vps/predictions.csv  (6/6-6/13, logged pre-game, real lines)
  2. data/predictions_log.csv strikeout rows (frozen at log time; includes the
     migrated legacy history)
  3. data/slates/<date>.csv frozen archives (7/8+, settled days only)
Actuals come from MLB StatsAPI final boxscores (grade.final_stats).

Two candidate corrections, evaluated WALK-FORWARD by date (fit on days < d,
score day d) against the raw model:
  affine:  mu' = a + b*mu               (global recentering, no line needed)
  anchor:  mu' = line + s*(mu - line)   (shrink toward the published line;
                                         the line is a strong consensus mean —
                                         s<1 = "trust our disagreement with
                                         the consensus only s much")
Scored on what the accuracy record measures: the model-chosen side's stated p
vs realized frequency, plus log-loss. Paste the winning constants into
pick6/projection.py with provenance.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pick6"))
from dispersion import DISPERSION_R                          # noqa: E402
from grade import final_stats                                # noqa: E402
from feed import norm                                        # noqa: E402

HERE = os.path.dirname(__file__)
PRED = r"C:\Users\carin\OneDrive\Dokument\stike\mlb-edge\data\exports\vps\predictions.csv"
PRED_LOG = os.path.join(HERE, "..", "data", "predictions_log.csv")
LEGACY_LOG = os.path.join(HERE, "..", "data", "pick6_entries.csv")
SLATES = os.path.join(HERE, "..", "data", "slates")

_EPS = 1e-9


def nb_logpmf(k: int, mu: float, r: float = DISPERSION_R) -> float:
    mu = max(mu, _EPS)
    return (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
            + r * math.log(r / (r + mu)) + k * math.log(mu / (r + mu)))


def p_more(mu: float, line: float) -> float:
    need = math.ceil(line)
    cdf = sum(math.exp(nb_logpmf(i, mu)) for i in range(need))
    return max(0.0, min(1.0, 1.0 - cdf))


# ---- collect frozen (date, mu, line, actual) --------------------------------

def collect() -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    # 1) mlb-edge pre-game log: one row per pitcher-date, prefer draftkings line
    best: dict[tuple[str, str], dict] = {}
    if os.path.exists(PRED):
        for r in csv.DictReader(open(PRED, encoding="utf-8")):
            if not r.get("expected_ks") or not r.get("line"):
                continue
            key = (r["date"], norm(r["pitcher"]))
            if key not in best or r.get("bookmaker") == "draftkings":
                best[key] = {"date": r["date"], "key": key[1],
                             "mu": float(r["expected_ks"]), "line": float(r["line"])}
    rows.extend(best.values())
    seen.update((r["date"], r["key"]) for r in rows)

    # 2) our own logged strikeout rows (projection frozen at log time).
    #    New log first; legacy history as fallback (grade.py migrates it).
    for path, name_col, mu_col in ((PRED_LOG, "player", "predicted"),
                                   (LEGACY_LOG, "pitcher", "lam")):
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if (r.get("market") or "strikeouts") != "strikeouts":
                continue
            key = (r["date"], norm(r[name_col]))
            if key in seen:
                continue
            seen.add(key)
            rows.append({"date": r["date"], "key": key[1],
                         "mu": float(r[mu_col]), "line": float(r["line"])})

    # 3) frozen slate archives (only days that have settled by run time)
    if os.path.isdir(SLATES):
        for fn in sorted(os.listdir(SLATES)):
            # skip consensus snapshots (different schema: role/player/fp_*)
            if not fn.endswith(".csv") or fn.endswith("_consensus.csv"):
                continue
            ds = fn[:-4]
            for r in csv.DictReader(open(os.path.join(SLATES, fn), encoding="utf-8")):
                key = (ds, norm(r["pitcher"]))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"date": ds, "key": key[1],
                             "mu": float(r["expected_ks"]),
                             "line": float(r["line"]) if r.get("line") else None})
    return rows


def settle(rows: list[dict]) -> list[dict]:
    out = []
    for ds in sorted({r["date"] for r in rows}):
        actuals = final_stats(ds)
        day = [r for r in rows if r["date"] == ds]
        n = 0
        for r in day:
            a = actuals.get(r["key"], {}).get("strikeouts")
            if a is not None:
                out.append({**r, "actual": int(a)})
                n += 1
        print(f"  {ds}: {n}/{len(day)} settled", flush=True)
    return out


# ---- corrections -------------------------------------------------------------

def fit_affine(pairs: list[dict]) -> tuple[float, float]:
    """(a, b) maximizing NB log-lik of actual under mu' = a + b*mu. Coarse grid
    then refine — 2 params, small data; robustness over elegance."""
    best, best_ll = (0.0, 1.0), -1e18
    for pass_, (a_rng, b_rng) in enumerate((
            ([x * 0.25 for x in range(-8, 9)], [x * 0.05 for x in range(8, 25)]),
            (None, None))):
        if pass_ == 1:
            a0, b0 = best
            a_rng = [a0 + x * 0.05 for x in range(-5, 6)]
            b_rng = [b0 + x * 0.01 for x in range(-5, 6)]
        for a in a_rng:
            for b in b_rng:
                ll = sum(nb_logpmf(r["actual"], a + b * r["mu"]) for r in pairs)
                if ll > best_ll:
                    best_ll, best = ll, (a, b)
    return best


def fit_anchor(pairs: list[dict]) -> float:
    """s in mu' = line + s*(mu - line), NB MLE. Uses only rows with a line."""
    pl = [r for r in pairs if r.get("line") is not None]
    best_s, best_ll = 1.0, -1e18
    for s in [x * 0.05 for x in range(0, 25)]:
        ll = sum(nb_logpmf(r["actual"], max(r["line"] + s * (r["mu"] - r["line"]), _EPS))
                 for r in pl)
        if ll > best_ll:
            best_ll, best_s = ll, s
    return best_s


def anchor_ci(pairs: list[dict], level_drop: float = 1.92) -> tuple[float, float]:
    """~95% profile-likelihood interval for s: every s whose log-lik is within
    `level_drop` (chi2_1/2) of the MLE. Answers 'is n big enough to exclude
    s>0?' — a wide interval means keep collecting, not keep concluding."""
    pl = [r for r in pairs if r.get("line") is not None]

    def ll(s):
        return sum(nb_logpmf(r["actual"],
                             max(r["line"] + s * (r["mu"] - r["line"]), _EPS))
                   for r in pl)

    grid = [x * 0.02 for x in range(0, 61)]          # 0.00 .. 1.20
    lls = {s: ll(s) for s in grid}
    peak = max(lls.values())
    inside = [s for s, v in lls.items() if v >= peak - level_drop]
    return (min(inside), max(inside)) if inside else (0.0, 1.2)


CORRECTIONS = {
    "raw":    lambda r, fit: r["mu"],
    "affine": lambda r, fit: fit["affine"][0] + fit["affine"][1] * r["mu"],
    "anchor": lambda r, fit: (r["line"] + fit["anchor"] * (r["mu"] - r["line"])
                              if r.get("line") is not None else r["mu"]),
}


# ---- walk-forward evaluation --------------------------------------------------

def evaluate(pairs: list[dict]) -> None:
    by_day = defaultdict(list)
    for r in pairs:
        by_day[r["date"]].append(r)
    dates = sorted(by_day)

    MIN_TRAIN = 40
    scored = {name: [] for name in CORRECTIONS}   # (pred_p, won, logloss_term)
    for d in dates:
        train = [r for dd in dates for r in by_day[dd] if dd < d]
        if len(train) < MIN_TRAIN:
            continue
        fit = {"affine": fit_affine(train), "anchor": fit_anchor(train)}
        for r in by_day[d]:
            if r.get("line") is None:
                continue
            actual_more = r["actual"] > r["line"]
            for name, cfn in CORRECTIONS.items():
                mu = max(cfn(r, fit), _EPS)
                pm = p_more(mu, r["line"])
                side_more = pm >= 0.5
                p = pm if side_more else 1.0 - pm
                won = (side_more == actual_more) if r["actual"] != r["line"] else None
                if won is None:
                    continue  # push on whole-number line
                ll = -math.log(max(p if won else 1 - p, _EPS))
                scored[name].append((p, won, ll))

    print("\nWALK-FORWARD (fit on days < d, score day d — the model-chosen side)")
    print(f"  {'correction':10} {'n':>4} {'predicted':>10} {'realized':>9} "
          f"{'gap':>7} {'log-loss':>9}")
    for name, legs in scored.items():
        if not legs:
            continue
        n = len(legs)
        pred = sum(p for p, _, _ in legs) / n
        real = sum(1 for _, w, _ in legs if w) / n
        ll = sum(l for _, _, l in legs) / n
        print(f"  {name:10} {n:>4} {pred*100:>9.1f}% {real*100:>8.1f}% "
              f"{(real-pred)*100:>+6.1f}p {ll:>9.4f}")

    # confident-bucket view: the rows that dominate the confidence ranking
    print("\n  HIGH-CONFIDENCE ROWS ONLY (stated p >= 0.65)")
    for name, legs in scored.items():
        hi = [(p, w) for p, w, _ in legs if p >= 0.65]
        if not hi:
            print(f"  {name:10}    0 legs clear 0.65")
            continue
        n = len(hi)
        pred = sum(p for p, _ in hi) / n
        real = sum(1 for _, w in hi if w) / n
        print(f"  {name:10} {n:>4} {pred*100:>9.1f}% {real*100:>8.1f}% "
              f"{(real-pred)*100:>+6.1f}p")


def main() -> None:
    rows = collect()
    print(f"collected {len(rows)} frozen pre-game projections; settling vs StatsAPI...")
    pairs = settle(rows)
    if len(pairs) < 60:
        print(f"only {len(pairs)} settled — too thin, aborting.")
        return

    bias = sum(r["actual"] - r["mu"] for r in pairs) / len(pairs)
    print(f"\n{len(pairs)} settled frozen starts")
    print(f"mean bias (actual - mu): {bias:+.2f} K "
          f"{'(model OVER-projects)' if bias < 0 else '(model UNDER-projects)'}")
    # bias by projection tercile — is it uniform or worst at the extremes?
    srt = sorted(pairs, key=lambda r: r["mu"])
    k = len(srt) // 3
    for tag, chunk in (("low-mu ", srt[:k]), ("mid-mu ", srt[k:2 * k]),
                       ("high-mu", srt[2 * k:])):
        b = sum(r["actual"] - r["mu"] for r in chunk) / len(chunk)
        mus = sum(r["mu"] for r in chunk) / len(chunk)
        print(f"  {tag}  mean mu {mus:.2f}   bias {b:+.2f} K   (n={len(chunk)})")

    fit_full = {"affine": fit_affine(pairs), "anchor": fit_anchor(pairs)}
    a, b = fit_full["affine"]
    lo, hi = anchor_ci(pairs)
    print(f"\nfull-sample fits:  affine mu' = {a:+.2f} + {b:.2f}*mu"
          f"    anchor mu' = line + {fit_full['anchor']:.2f}*(mu - line)")
    print(f"anchor s ~95% profile-likelihood interval: [{lo:.2f}, {hi:.2f}]"
          f"   (production s = {__import__('projection').SHRINK_TO_LINE_S})")
    if hi > 0.1:
        print("  interval does NOT exclude a useful s — keep collecting frozen"
              " days; raise s only when the walk-forward supports it.")

    evaluate(pairs)
    print("\n=> paste the winning correction into pick6/projection.py (with this"
          "\n   run's numbers as provenance). If 'raw' wins walk-forward, the June"
          "\n   sample doesn't support a correction — the July drift is then either"
          "\n   variance or a model change since 6/13; keep the anchor until data says otherwise.")


if __name__ == "__main__":
    main()
