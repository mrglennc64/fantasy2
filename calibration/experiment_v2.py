"""Do additional features beat the served 8-feature model? (Phase 3, step 1)

    python experiment_v2.py

MEASURE BEFORE PLUMBING. Phase 0 (validate.py) established that the served
model barely beats a naive baseline — MAE 1.754 vs 1.774 K, rho 0.409 vs
0.369 on 1,098 held-out starts. So the question is not "can we add features",
it is "does any candidate move THOSE numbers". This script answers that
against the identical split and metrics, before a line of serving code changes.

CANDIDATES. Every one is a rolling rate over prior starts only, and every one
is present BOTH in the training corpus and in the StatsAPI gameLog the serving
path already fetches. That constraint is deliberate: `strikes` / strikePercentage
would probably be the single best stuff proxy available, but it is in the
gameLog and NOT in the gamelog CSV, and training on a feature the server
cannot build is how the existing model ended up training on confirmed-lineup
K% while serving team-level K%.

  pitch_per_bf   deep counts -> more swings and misses
  k_per_pitch    strikeouts per pitch thrown; purer stuff than K/BF
  bb_per_bf      control. High-K pitchers often walk more, so this is not
                 simply "worse pitcher"
  h_per_bf       contact allowed, roughly the inverse of missing bats
  ip_per_start   the leash — innings is opportunity, and opportunity is Ks
  hr_per_bf      contact quality allowed
  is_lhp         handedness

Fit is Poisson GLM (log link), same family as production, so a win here is
directly shippable as plain constants with no ML runtime on the VPS. A tree
model is fitted alongside purely as a ceiling check: if GBM cannot beat the
GLM either, the features are the limit, not the functional form.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from datetime import date as _date

import numpy as np
from sklearn.linear_model import PoissonRegressor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pick6"))
from validate import GAMELOGS, VAL_FROM, spearman, _truthy    # noqa: E402
from kmodel_params import PARKS                               # noqa: E402

PRIOR_BF = 150.0

V1 = ["roll_kbf_3", "roll_kbf_10", "prior_kbf", "roll_bf_3",
      "opp_k_pct", "is_home", "rest_days", "park_k"]

V2_CANDIDATES = ["pitch_per_bf", "k_per_pitch", "bb_per_bf", "h_per_bf",
                 "ip_per_start", "hr_per_bf", "is_lhp"]


def load() -> list[dict]:
    out = []
    for r in csv.DictReader(open(GAMELOGS, encoding="utf-8")):
        try:
            bf = float(r["BF"])
            if bf <= 0:
                continue
            home = _truthy(r["is_home"])
            out.append({
                "date": r["date"], "pid": r["pitcher_id"], "K": int(float(r["K"])),
                "BF": bf, "is_home": home,
                "P": float(r["pitches"] or 0), "H": float(r["H"] or 0),
                "BB": float(r["BB"] or 0), "HR": float(r["HR"] or 0),
                "IP": float(r["IP"] or 0),
                "is_lhp": 1.0 if (r.get("throws") or "").upper() == "L" else 0.0,
                "opp_k_pct": float(r["opp_k_pct"]) if r.get("opp_k_pct") else None,
                "venue": r.get("team" if home else "opponent", ""),
            })
        except (ValueError, KeyError):
            continue
    if not out:
        raise SystemExit("no rows parsed — check the gamelog schema")
    out.sort(key=lambda r: (r["date"], r["pid"]))
    return out


def build(rows: list[dict]) -> list[dict]:
    """Strictly-prior rolling features. Every window looks only at starts that
    had already happened, so nothing here can leak the outcome."""
    hist: dict[str, list[dict]] = defaultdict(list)
    tot_k = tot_bf = 0.0
    out = []
    for r in rows:
        h = hist[r["pid"]]
        if len(h) >= 2:
            def _kbf(last):
                hh = h if last is None else h[-last:]
                base = tot_k / tot_bf if tot_bf > 500 else 0.22
                return ((sum(x["K"] for x in hh) + PRIOR_BF * base)
                        / (sum(x["BF"] for x in hh) + PRIOR_BF))

            def _rate(num, den, window=10, prior=0.0, w=50.0):
                hh = h[-window:]
                n = sum(x[num] for x in hh)
                d = sum(x[den] for x in hh)
                return (n + w * prior) / (d + w) if (d + w) else prior

            w10 = h[-10:]
            rest = (_date.fromisoformat(r["date"])
                    - _date.fromisoformat(h[-1]["date"])).days
            out.append({**r,
                "roll_kbf_3": _kbf(3), "roll_kbf_10": _kbf(10),
                "prior_kbf": _kbf(None),
                "roll_bf_3": sum(x["BF"] for x in h[-3:]) / min(len(h), 3),
                "rest_days": float(min(max(rest, 3), 12)),
                "park_k": PARKS.get(r.get("venue"), 1.0),
                # --- candidates ---
                "pitch_per_bf": _rate("P", "BF", prior=3.9),
                "k_per_pitch": _rate("K", "P", prior=0.055, w=200.0),
                "bb_per_bf": _rate("BB", "BF", prior=0.08),
                "h_per_bf": _rate("H", "BF", prior=0.22),
                "hr_per_bf": _rate("HR", "BF", prior=0.03),
                "ip_per_start": sum(x["IP"] for x in w10) / len(w10),
            })
        h.append(r)
        tot_k += r["K"]
        tot_bf += r["BF"]
    return out


def fit_eval(train, val, feats, label, tree=False):
    """Fit on train, score on val. Returns (label, mae, bias, rho)."""
    def mat(rows):
        X = np.array([[(r[f] if r.get(f) is not None else 0.0) for f in feats]
                      for r in rows], dtype=float)
        return X, np.array([r["K"] for r in rows], dtype=float)

    Xtr, ytr = mat(train)
    Xv, yv = mat(val)
    # impute missing opp_k_pct with the training mean, as the server does
    mu_, sd_ = Xtr.mean(0), Xtr.std(0)
    sd_[sd_ == 0] = 1.0
    Ztr, Zv = (Xtr - mu_) / sd_, (Xv - mu_) / sd_
    if tree:
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(loss="poisson", max_iter=300,
                                          learning_rate=0.05, max_depth=4,
                                          min_samples_leaf=40, random_state=0)
        m.fit(Ztr, ytr)
        pred = m.predict(Zv)
    else:
        m = PoissonRegressor(alpha=1e-4, max_iter=3000)
        m.fit(Ztr, ytr)
        pred = m.predict(Zv)
    err = yv - pred
    return (label, float(np.abs(err).mean()), float(err.mean()),
            spearman(list(pred), list(yv)), m, mu_, sd_)


def main() -> None:
    rows = build(load())
    train = [r for r in rows if r["date"] < VAL_FROM]
    val = [r for r in rows if r["date"] >= VAL_FROM]
    print(f"{len(train)} train / {len(val)} held out (>= {VAL_FROM})")
    print(f"\nBASELINE TO BEAT (Phase 0, served model): MAE 1.754  rho 0.409")

    results = []
    results.append(fit_eval(train, val, V1, "v1 (8 features, refit)"))

    # one at a time: which candidate earns its place on its own?
    print(f"\n{'-'*64}\nONE CANDIDATE AT A TIME (added to v1)\n{'-'*64}")
    print(f"  {'feature':<22} {'MAE':>7} {'d MAE':>7} {'rho':>8} {'d rho':>8}")
    base = results[0]
    solo = []
    for c in V2_CANDIDATES:
        r = fit_eval(train, val, V1 + [c], c)
        solo.append((c, base[1] - r[1], r[3] - base[3]))
        print(f"  {c:<22} {r[1]:>7.4f} {base[1]-r[1]:>+7.4f} "
              f"{r[3]:>8.4f} {r[3]-base[3]:>+8.4f}")

    keep = [c for c, dmae, drho in solo if dmae > 0.001 or drho > 0.002]
    print(f"\n  candidates that help individually: {keep or 'NONE'}")

    print(f"\n{'-'*64}\nCOMBINED\n{'-'*64}")
    results.append(fit_eval(train, val, V1 + V2_CANDIDATES, "v1 + ALL candidates"))
    if keep:
        results.append(fit_eval(train, val, V1 + keep, "v1 + helpful only"))
    results.append(fit_eval(train, val, V1 + V2_CANDIDATES, "GBM ceiling check",
                            tree=True))

    print(f"  {'model':<26} {'MAE':>7} {'bias':>7} {'rho':>8}")
    for label, mae, bias, rho, *_ in results:
        print(f"  {label:<26} {mae:>7.4f} {bias:>+7.3f} {rho:>8.4f}")

    best = min(results, key=lambda r: r[1])
    print(f"\n  best: {best[0]}  (MAE {best[1]:.4f}, rho {best[3]:.4f})")
    gain = 1.754 - best[1]
    print(f"  vs served baseline: MAE {gain:+.4f} K "
          f"({100*gain/1.754:+.1f}%), rho {best[3]-0.409:+.4f}")

    # Two different questions, and conflating them is how a 1% improvement
    # gets sold as progress. "Better on held-out data" is not "better enough
    # to change a decision at a line."
    #
    # Scale: actual K has SD ~2.2 around the projection. Moving MAE by 0.03 K
    # shifts P(K > line) by well under a point for a typical half-integer
    # line, which cannot move 51.9% against real lines to anywhere useful.
    # A material gain would be ~0.15 K of MAE — an order of magnitude more.
    MATERIAL_MAE = 0.15
    MATERIAL_RHO = 0.05
    drho = best[3] - 0.409
    print(f"\n  statistically better than v1: "
          f"{'yes' if gain > 0.005 else 'no'}")
    print(f"  MATERIALLY better (MAE {MATERIAL_MAE:+.2f} K or rho "
          f"{MATERIAL_RHO:+.2f}): "
          f"{'yes' if (gain >= MATERIAL_MAE or drho >= MATERIAL_RHO) else 'NO'}")
    if gain < MATERIAL_MAE and drho < MATERIAL_RHO:
        print("\n  => The gamelog corpus is close to exhausted. These features")
        print("     are real but too small to change a decision at a line.")
        print("     The missing information is not in box scores: it is")
        print("     Statcast (whiff / CSW / pitch mix), confirmed lineup")
        print("     composition, and umpire zone. Ship these only as a free")
        print("     increment, not as the fix.")
    else:
        print("\n  => material. Plumb into train_kmodel.py + kmodel.py serving.")

    # Negative result worth stating loudly, because it retires a whole branch
    # of the roadmap: if the tree model cannot clear the GLM by much, the
    # limit is the FEATURES, not the functional form. No amount of
    # XGBoost/LightGBM tuning recovers information the inputs do not carry.
    glm = min((r for r in results if "GBM" not in r[0]), key=lambda r: r[1])
    gbm = next((r for r in results if "GBM" in r[0]), None)
    if gbm:
        print(f"\n  FUNCTIONAL FORM: GBM {gbm[1]:.4f} vs best GLM {glm[1]:.4f} "
              f"= {glm[1]-gbm[1]:+.4f} K")
        if glm[1] - gbm[1] < 0.05:
            print("  => nonlinearity is NOT the bottleneck. A gradient-boosted")
            print("     mean correction would not have helped either.")


if __name__ == "__main__":
    main()
