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
sys.path.insert(0, os.path.dirname(__file__))
from dispersion import DISPERSION_R                          # noqa: E402
from markets import over_threshold                           # noqa: E402
from grade import final_stats                                # noqa: E402
from feed import norm                                        # noqa: E402
from log_schema import SRC_LEGACY, SRC_UNKNOWN, legacy_source  # noqa: E402

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
    # over_threshold, not ceil: a whole-number line pushes on equality, so More
    # needs line+1 there. Must match production (markets.p_over) exactly or the
    # affine fitted here is fitted against a different question than it serves.
    need = over_threshold(line)
    cdf = sum(math.exp(nb_logpmf(i, mu)) for i in range(need))
    return max(0.0, min(1.0, 1.0 - cdf))


# ---- collect frozen (date, mu, line, actual) --------------------------------

def collect() -> list[dict]:
    """Frozen (date, pitcher, mu, line, src) rows.

    Every row carries the ESTIMATOR that produced its mu. A correction fitted
    across a mix of estimators describes none of them: sources 1 and 3 are
    mlb-edge by construction, while source 2 is genuinely mixed from
    2026-07-13 (when feed.py began falling through to the owned kmodel), so it
    reports whatever its mu_source column says.

    Dedupe is keyed on (date, pitcher, SRC), not (date, pitcher). Two
    estimators projecting the same start are two observations of two different
    models — collapsing them silently discards whichever arrived second, which
    since 07-13 means quietly throwing away exactly the kmodel rows a kmodel
    fit would need.
    """
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    # 1) mlb-edge pre-game log: one row per pitcher-date, prefer draftkings line
    best: dict[tuple[str, str], dict] = {}
    if os.path.exists(PRED):
        for r in csv.DictReader(open(PRED, encoding="utf-8")):
            if not r.get("expected_ks") or not r.get("line"):
                continue
            key = (r["date"], norm(r["pitcher"]))
            if key not in best or r.get("bookmaker") == "draftkings":
                best[key] = {"date": r["date"], "key": key[1],
                             "mu": float(r["expected_ks"]),
                             "line": float(r["line"]), "src": SRC_LEGACY}
    rows.extend(best.values())
    seen.update((r["date"], r["key"], r["src"]) for r in rows)

    # 2) our own logged strikeout rows (projection frozen at log time).
    #    New log first; legacy history as fallback (grade.py migrates it).
    for path, name_col, mu_col, fixed_src in (
            (PRED_LOG, "player", "predicted", None),
            (LEGACY_LOG, "pitcher", "lam", SRC_LEGACY)):
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if (r.get("market") or "strikeouts") != "strikeouts":
                continue
            src = fixed_src or r.get("mu_source") or legacy_source(r["date"])
            key = (r["date"], norm(r[name_col]), src)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"date": r["date"], "key": key[1],
                         "mu": float(r[mu_col]), "line": float(r["line"]),
                         "src": src})

    # 3) frozen slate archives (only days that have settled by run time)
    if os.path.isdir(SLATES):
        for fn in sorted(os.listdir(SLATES)):
            # skip consensus snapshots (different schema: role/player/fp_*)
            if not fn.endswith(".csv") or fn.endswith("_consensus.csv"):
                continue
            ds = fn[:-4]
            for r in csv.DictReader(open(os.path.join(SLATES, fn), encoding="utf-8")):
                key = (ds, norm(r["pitcher"]), SRC_LEGACY)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"date": ds, "key": key[1],
                             "mu": float(r["expected_ks"]),
                             "line": float(r["line"]) if r.get("line") else None,
                             "src": SRC_LEGACY})
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


def affine_ci_b(pairs: list[dict], level_drop: float = 1.92) -> tuple[float, float]:
    """~95% profile-likelihood interval for the SLOPE b in mu' = a + b*mu,
    maximizing over a at each b. Same construction as anchor_ci above.

    b is the parameter that matters. A wrong intercept shifts every projection
    equally; a wrong slope changes the SPREAD between pitchers, which is the
    entire ranking. b=0.50 applied to the wrong estimator is what flattened the
    board. So the gate asks the only question that protects against repeating
    that: is this shrink distinguishable from doing nothing at all? If the
    interval covers 1.0, the honest answer is identity — regardless of how much
    the point estimate appears to improve things in-sample.
    """
    a_grid = [x * 0.25 for x in range(-8, 9)]

    def ll(b):
        return max(sum(nb_logpmf(r["actual"], max(a + b * r["mu"], _EPS))
                       for r in pairs) for a in a_grid)

    grid = [x * 0.02 for x in range(0, 76)]          # 0.00 .. 1.50
    lls = {b: ll(b) for b in grid}
    peak = max(lls.values())
    inside = [b for b, v in lls.items() if v >= peak - level_drop]
    return (min(inside), max(inside)) if inside else (0.0, 1.5)


# Promotion gate. A correction ships only if it clears every condition on its
# OWN source's data. Failing any one leaves that source at identity — it never
# borrows another source's coefficients, which is the bug this file now exists
# to prevent.
MIN_N = 150            # settled starts for that source
MIN_TEST_DAYS = 5      # walk-forward days scored; one lucky day is not evidence
MIN_REL_GAIN = 0.01    # 1% relative, on BOTH Brier and log-loss


def gate(src: str, n: int, test_days: int, sc_raw: list, sc_aff: list,
         b_ci: tuple[float, float]) -> tuple[bool, str]:
    """(promote?, reason). Reason is printed either way, so a refusal reads as
    a decision with a cause rather than as missing output."""
    if n < MIN_N:
        return False, f"n={n} < {MIN_N} settled starts"
    if test_days < MIN_TEST_DAYS:
        return False, f"only {test_days} walk-forward test day(s) < {MIN_TEST_DAYS}"
    if not sc_raw or not sc_aff:
        return False, "no held-out rows scored"

    def brier(sc):
        return sum((p - (1.0 if w else 0.0)) ** 2 for p, w, _ in sc) / len(sc)

    def logloss(sc):
        return sum(l for _, _, l in sc) / len(sc)

    b_raw, b_aff = brier(sc_raw), brier(sc_aff)
    l_raw, l_aff = logloss(sc_raw), logloss(sc_aff)
    if b_aff > b_raw * (1 - MIN_REL_GAIN):
        return False, f"Brier {b_aff:.4f} vs raw {b_raw:.4f} (< {MIN_REL_GAIN:.0%} gain)"
    if l_aff > l_raw * (1 - MIN_REL_GAIN):
        return False, f"log-loss {l_aff:.4f} vs raw {l_raw:.4f} (< {MIN_REL_GAIN:.0%} gain)"
    if b_ci[0] <= 1.0 <= b_ci[1]:
        return False, f"b 95% CI [{b_ci[0]:.2f}, {b_ci[1]:.2f}] includes 1.0 (= identity)"
    return True, (f"Brier {b_aff:.4f}<{b_raw:.4f}, log-loss {l_aff:.4f}<{l_raw:.4f}, "
                  f"b CI [{b_ci[0]:.2f}, {b_ci[1]:.2f}] excludes 1.0")


CORRECTIONS = {
    "raw":    lambda r, fit: r["mu"],
    "affine": lambda r, fit: fit["affine"][0] + fit["affine"][1] * r["mu"],
    "anchor": lambda r, fit: (r["line"] + fit["anchor"] * (r["mu"] - r["line"])
                              if r.get("line") is not None else r["mu"]),
}


# ---- walk-forward evaluation --------------------------------------------------

def evaluate(pairs: list[dict]) -> tuple[dict, int]:
    """Walk-forward score each correction. Returns (scored, n_test_days) so the
    gate can judge the same numbers that get printed."""
    by_day = defaultdict(list)
    for r in pairs:
        by_day[r["date"]].append(r)
    dates = sorted(by_day)

    MIN_TRAIN = 40
    scored = {name: [] for name in CORRECTIONS}   # (pred_p, won, logloss_term)
    test_days = 0
    for d in dates:
        train = [r for dd in dates for r in by_day[dd] if dd < d]
        if len(train) < MIN_TRAIN:
            continue
        fit = {"affine": fit_affine(train), "anchor": fit_anchor(train)}
        day_scored = False
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
                day_scored = True
        test_days += day_scored

    print("\nWALK-FORWARD (fit on days < d, score day d — the model-chosen side)")
    print(f"  {'correction':10} {'n':>4} {'predicted':>10} {'realized':>9} "
          f"{'gap':>7} {'log-loss':>9} {'Brier':>8}")
    for name, legs in scored.items():
        if not legs:
            continue
        n = len(legs)
        pred = sum(p for p, _, _ in legs) / n
        real = sum(1 for _, w, _ in legs if w) / n
        ll = sum(l for _, _, l in legs) / n
        br = sum((p - (1.0 if w else 0.0)) ** 2 for p, w, _ in legs) / n
        print(f"  {name:10} {n:>4} {pred*100:>9.1f}% {real*100:>8.1f}% "
              f"{(real-pred)*100:>+6.1f}p {ll:>9.4f} {br:>8.4f}")

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
    return scored, test_days


def fit_source(src: str, pairs: list[dict]) -> tuple[float, float]:
    """Diagnose, fit and gate ONE estimator. Returns the (a, b) to serve."""
    import projection as _pj
    prod = _pj.CORRECTIONS.get(src, _pj.IDENTITY)
    print(f"\n{'=' * 72}\nSOURCE: {src}   ({len(pairs)} settled frozen starts)")
    print(f"  production: mu' = {prod[0]:+.2f} + {prod[1]:.2f}*mu"
          + ("   [identity]" if tuple(prod) == _pj.IDENTITY else ""))
    if len(pairs) < 20:
        print("  too thin to diagnose — staying at production.")
        return prod

    bias = sum(r["actual"] - r["mu"] for r in pairs) / len(pairs)
    print(f"  mean bias (actual - mu): {bias:+.2f} K "
          f"{'(OVER-projects)' if bias < 0 else '(UNDER-projects)'}")
    # bias by projection tercile — is it uniform or worst at the extremes?
    srt = sorted(pairs, key=lambda r: r["mu"])
    k = len(srt) // 3
    for tag, chunk in (("low-mu ", srt[:k]), ("mid-mu ", srt[k:2 * k]),
                       ("high-mu", srt[2 * k:])):
        if not chunk:
            continue
        b_ = sum(r["actual"] - r["mu"] for r in chunk) / len(chunk)
        mus = sum(r["mu"] for r in chunk) / len(chunk)
        print(f"    {tag}  mean mu {mus:.2f}   bias {b_:+.2f} K   (n={len(chunk)})")

    a, b = fit_affine(pairs)
    b_ci = affine_ci_b(pairs)
    print(f"  full-sample affine: mu' = {a:+.2f} + {b:.2f}*mu"
          f"    b 95% CI [{b_ci[0]:.2f}, {b_ci[1]:.2f}]")

    scored, test_days = evaluate(pairs)
    ok, why = gate(src, len(pairs), test_days, scored["raw"], scored["affine"], b_ci)
    if ok:
        print(f"\n  => PROMOTE {src}: mu' = {a:+.2f} + {b:.2f}*mu   ({why})")
        return (round(a, 2), round(b, 2))

    # KEEP PRODUCTION, not identity. These are different things and conflating
    # them cost real accuracy the first time this ran: mlbedge_slate had 173
    # settled starts and a slope CI of [0.48, 0.68] — its affine is its OWN
    # fit and clearly justified — but the gate refused on an unrelated
    # condition (too few walk-forward days) and the old code reverted it to
    # mu' = mu. Its own walk-forward said that was worse: 36.5% realized on
    # raw vs 42.9% on the affine.
    #
    # IDENTITY is the DEFAULT for a source that has never been fitted, so it
    # cannot inherit a foreign correction. A gate REFUSAL means "not enough
    # evidence to change what is live" — which is an argument for leaving
    # production alone, never for discarding a fit that was itself gated in.
    print(f"\n  => KEEP PRODUCTION for {src}: mu' = {prod[0]:+.2f} + "
          f"{prod[1]:.2f}*mu   (gate: {why})")
    if tuple(prod) == _pj.IDENTITY:
        print("     Production is identity, so this source stays uncorrected. "
              "It does NOT\n     inherit another source's fit — that is the "
              "failure this gate prevents.")
    else:
        print(f"     Not enough evidence to CHANGE the live correction; that "
              f"is not\n     evidence to remove it. Full-sample fit was "
              f"{a:+.2f} + {b:.2f}*mu for reference.")
    return prod


def main() -> None:
    rows = collect()
    print(f"collected {len(rows)} frozen pre-game projections; settling vs StatsAPI...")
    pairs = settle(rows)
    if len(pairs) < 60:
        print(f"only {len(pairs)} settled — too thin, aborting.")
        return

    by_src = defaultdict(list)
    for r in pairs:
        by_src[r["src"]].append(r)

    print(f"\n{len(pairs)} settled frozen starts across {len(by_src)} source(s):")
    for src in sorted(by_src, key=lambda s: -len(by_src[s])):
        days = len({r["date"] for r in by_src[src]})
        print(f"  {src:<18} {len(by_src[src]):>4} starts over {days:>3} days")

    n_unknown = len(by_src.get(SRC_UNKNOWN, []))
    if n_unknown:
        print(f"\n  NOTE: {n_unknown} rows have an unknown source (logged from "
              f"2026-07-13,\n  before mu_source existed, when the feed chain "
              f"could serve either\n  estimator). Counted but NOT fitted — "
              f"labelling them to grow the\n  sample would poison the very fit "
              f"they would be feeding.")

    fitted = {}
    for src in sorted(by_src):
        if src == SRC_UNKNOWN:
            continue
        fitted[src] = fit_source(src, by_src[src])

    print(f"\n{'=' * 72}\nRESULTING per-source mean corrections:")
    for src, (a, b) in sorted(fitted.items()):
        print(f"    {src!r:<22}: ({a:.2f}, {b:.2f}),")
    print("\nA source absent here has no settled data yet and keeps whatever "
          "\npick6/projection.py _DEFAULTS says.")

    if "--write" not in sys.argv:
        print("\n(report-only; re-run with --write to apply to data/params.json)")
        return
    import params_io
    path = params_io.update(
        "mean_correction", {s: [a, b] for s, (a, b) in fitted.items()},
        {src: {"n": len(by_src[src]),
               "days": len({r["date"] for r in by_src[src]})}
         for src in fitted},
        written_by="calibration/fit_mean.py")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
