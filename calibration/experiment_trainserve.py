"""Phase A: measure the cost of the opp_k_pct train/serve mismatch.

    python experiment_trainserve.py

THE DEFECT. train_kmodel.py:76 writes confirmed-lineup `lineup_k_pct` into the
`opp_k_pct` slot whenever the backfill has it; kmodel.py:249 always serves
team-level `opp_k_pct`. One column name, two different variables:

    lineup_k_pct   2,914 rows   mean 0.2191   SD 0.0138
    opp_k_pct     11,796 rows   mean 0.2231   SD 0.0246

The served variable has 1.8x the spread of the one 18% of training rows
carried, so a single fitted coefficient is applied to a wider variable than
part of the fit saw. Nothing reports it, because the column name matches.

WHAT THIS MEASURES. Two configurations, trained on the same split, scored on
the SAME held-out rows:

    mixed     what ships today - lineup_k_pct substituted where available
    servable  team opp_k_pct everywhere, i.e. what kmodel.project() can build

The comparison must be on common rows or it is not a comparison: 837 of the
backfill starts exist ONLY in the lineup file and carry no team value at all,
so `servable` cannot score them. All evaluation is restricted to held-out rows
that both configurations can build, and the dropped rows are reported rather
than quietly excluded.

WHAT IT CANNOT SETTLE. If `servable` scores worse, that is not an argument for
keeping the mismatch — it is an argument for SERVING lineup_k_pct (roadmap
Phase A option 1). Training on a feature the server cannot build is the failure
mode, regardless of which trains better. This experiment sizes the trade; it
does not license the status quo.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pick6"))
sys.path.insert(0, os.path.dirname(__file__))
import train_kmodel as T  # noqa: E402

HERE = os.path.dirname(__file__)
LOGS = T.LOGS
BACKFILL = T.BACKFILL
VAL_FROM = T.VAL_FROM
FEATURES = T.FEATURES


def load(mode: str) -> list[dict]:
    """mode='mixed' reproduces production training; mode='servable' uses only
    the team-level feature the server can actually build."""
    lineup = {}
    if os.path.exists(BACKFILL):
        for r in csv.DictReader(open(BACKFILL, encoding="utf-8")):
            if r.get("lineup_k_pct"):
                lineup[(r["date"], str(r["pitcher_id"]))] = r

    rows, seen = [], set()
    for r in csv.DictReader(open(LOGS, encoding="utf-8")):
        try:
            key = (r["date"], str(r["pitcher_id"]))
            bf = lineup.get(key)
            if mode == "mixed":
                opp = float(bf["lineup_k_pct"]) if bf else float(r["opp_k_pct"])
            else:
                opp = float(r["opp_k_pct"])
            rows.append({
                "date": r["date"], "pitcher": r["pitcher"], "pid": r["pitcher_id"],
                "K": int(float(r["K"])), "BF": float(r["BF"]),
                "pitches": float(r["pitches"] or 0), "opp_k_pct": opp,
                "is_home": 1.0 if r["is_home"] == "True" else 0.0,
                "venue": r["team"] if r["is_home"] == "True" else r["opponent"],
                "servable": True,
            })
            seen.add(key)
        except (KeyError, ValueError):
            continue

    # Backfill-only starts carry no team value. 'mixed' includes them on the
    # lineup feature; 'servable' cannot represent them at all.
    for key, r in lineup.items():
        if key in seen:
            continue
        if mode != "mixed":
            continue
        try:
            home = str(r["is_home"]) == "True"
            rows.append({
                "date": r["date"], "pitcher": r["pitcher"], "pid": r["pitcher_id"],
                "K": int(float(r["K"])), "BF": float(r["BF"]),
                "pitches": float(r["pitches"] or 0),
                "opp_k_pct": float(r["lineup_k_pct"]),
                "is_home": 1.0 if home else 0.0,
                "venue": r["team"] if home else r["opponent"],
                "servable": False,
            })
        except (KeyError, ValueError):
            continue
    rows.sort(key=lambda r: r["date"])
    return rows


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def fit_and_score(mode: str, common: set) -> dict:
    rows = T.build_features(load(mode))
    tr = [r for r in rows if r["date"] < VAL_FROM]
    te = [r for r in rows
          if r["date"] >= VAL_FROM and (r["date"], r["pid"]) in common]

    Xtr = np.array([[r[f] for f in FEATURES] for r in tr])
    ytr = np.array([r["K"] for r in tr], dtype=float)
    mean, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    m = PoissonRegressor(alpha=1e-4, max_iter=5000)
    m.fit((Xtr - mean) / sd, ytr)

    Xte = np.array([[r[f] for f in FEATURES] for r in te])
    pred = m.predict((Xte - mean) / sd)
    act = np.array([r["K"] for r in te], dtype=float)
    return {"n_train": len(tr), "n_test": len(te),
            "mae": float(np.abs(pred - act).mean()),
            "bias": float((act - pred).mean()),
            "rho": spearman(list(pred), list(act)),
            "coef_opp": float(m.coef_[FEATURES.index("opp_k_pct")]),
            "sd_opp": float(sd[FEATURES.index("opp_k_pct")])}


def main() -> None:
    mixed_rows = T.build_features(load("mixed"))
    serv_rows = T.build_features(load("servable"))
    m_keys = {(r["date"], r["pid"]) for r in mixed_rows if r["date"] >= VAL_FROM}
    s_keys = {(r["date"], r["pid"]) for r in serv_rows if r["date"] >= VAL_FROM}
    common = m_keys & s_keys
    dropped = m_keys - s_keys
    print(f"held-out rows: mixed {len(m_keys):,}   servable {len(s_keys):,}   "
          f"common {len(common):,}")
    print(f"  {len(dropped):,} held-out rows exist only in 'mixed' "
          f"(backfill-only, no team value) and are excluded from BOTH scores\n")

    res = {m: fit_and_score(m, common) for m in ("mixed", "servable")}
    print(f"{'config':10} {'train':>7} {'test':>6} {'MAE':>8} {'bias':>8} {'rho':>8} "
          f"{'coef':>8} {'feat SD':>9}")
    for m, r in res.items():
        print(f"{m:10} {r['n_train']:>7,} {r['n_test']:>6,} {r['mae']:>8.4f} "
              f"{r['bias']:>+8.3f} {r['rho']:>+8.4f} {r['coef_opp']:>+8.4f} "
              f"{r['sd_opp']:>9.4f}")

    d_mae = res["mixed"]["mae"] - res["servable"]["mae"]
    print(f"\nservable - mixed: MAE {-d_mae:+.4f} K "
          f"({-d_mae/res['mixed']['mae']*100:+.2f}%), "
          f"rho {res['servable']['rho']-res['mixed']['rho']:+.4f}")
    print("\nThe coefficient and feature-SD columns are the point: the two "
          "configs\nstandardise opp_k_pct by different scales, which is exactly "
          "the mismatch\nproduction ships — fit on one scale, served on the other.")


if __name__ == "__main__":
    sys.exit(main())
