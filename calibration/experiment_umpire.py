"""Does the home-plate umpire's strikeout tendency predict a start's K rate?

    python experiment_umpire.py

The last untested item in Phase C. gameinfo.csv carries umphome for 98% of
224,877 games back to 1898, so the feature is free — the only question is
whether it carries signal the model does not already have.

Same harness as experiment_swstr.py, for the same reason: everything is built
from PRIOR games only and scored walk-forward on starts the feature has never
seen. An umpire's K tendency is computed from the games he worked BEFORE the
one being predicted, which is also the only form that could ever be served —
tomorrow's plate umpire is announced, his history is not a forecast.

PRIOR EXPECTATION, recorded before running so the result cannot be rationalised
after the fact: under +0.3% of MAE. Swinging-strike rate is a direct measure of
the pitcher's own skill and bought +0.26%; an umpire is a third party assigned
roughly at random, whose zone affects both teams. If this clears +1% it is a
surprise worth acting on. If it does not, Phase C is answered and the roadmap's
stated conclusion stands.

Shrinkage matters here more than in the SwStr test. An umpire with 12 prior
games has a K rate dominated by which pitchers he happened to draw, so the raw
rate is mostly noise about the pitchers, not signal about the umpire. Rates are
shrunk toward the league mean by PRIOR_G pseudo-games; without that, this
measures roster luck and calls it an umpire effect.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(__file__)
RS = os.path.join(HERE, "..", "data", "retrosheet")
GAMEINFO = os.path.join(RS, "gameinfo.csv")
PITCHING = os.path.join(RS, "pitching.csv")

SEASONS = ("2024", "2025")
MIN_PRIOR_STARTS = 5
MIN_PRIOR_BF = 100
MIN_BF = 15
MIN_UMP_GAMES = 10        # prior games worked before his rate is usable
PRIOR_G = 30.0            # pseudo-games of league-average shrinkage


def load_umpires() -> dict[str, tuple[str, str]]:
    """gid -> (home-plate umpire, date), for the seasons under test."""
    out = {}
    with open(GAMEINFO, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            d = r.get("date") or ""
            if d[:4] not in SEASONS:
                continue
            u = r.get("umphome")
            if u:
                out[r["gid"]] = (u, d)
    return out


def load_starts() -> list[dict]:
    rows = []
    with open(PITCHING, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r.get("p_gs") != "1" or r.get("stattype") not in (None, "", "value"):
                continue
            d = r.get("date") or ""
            if d[:4] not in SEASONS:
                continue
            try:
                bf, k = float(r["p_bfp"] or 0), float(r["p_k"] or 0)
            except ValueError:
                continue
            if bf < MIN_BF:
                continue
            rows.append({"gid": r["gid"], "pid": r["id"], "date": d, "BF": bf, "K": k})
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


def ols(X, y):
    n, p = len(X), len(X[0]) + 1
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for xi, yi in zip(X, y):
        v = [1.0] + xi
        for i in range(p):
            b[i] += v[i] * yi
            for j in range(p):
                A[i][j] += v[i] * v[j]
    for i in range(p):
        piv = max(range(i, p), key=lambda r: abs(A[r][i]))
        A[i], A[piv] = A[piv], A[i]
        b[i], b[piv] = b[piv], b[i]
        if abs(A[i][i]) < 1e-12:
            continue
        for r in range(i + 1, p):
            f = A[r][i] / A[i][i]
            for c in range(i, p):
                A[r][c] -= f * A[i][c]
            b[r] -= f * b[i]
    coef = [0.0] * p
    for i in range(p - 1, -1, -1):
        if abs(A[i][i]) < 1e-12:
            continue
        coef[i] = (b[i] - sum(A[i][j] * coef[j] for j in range(i + 1, p))) / A[i][i]
    return coef


def main() -> None:
    umps = load_umpires()
    print(f"games with a home-plate umpire in {'/'.join(SEASONS)}: {len(umps):,}")

    starts = load_starts()
    starts = [s for s in starts if s["gid"] in umps]
    for s in starts:
        s["ump"], _ = umps[s["gid"]]
    print(f"started outings joined to an umpire: {len(starts):,}")

    # League mean, for shrinkage.
    league = sum(s["K"] for s in starts) / sum(s["BF"] for s in starts)
    print(f"league K/BF {league:.4f}\n")

    # Chronological walk. Both accumulators hold ONLY the past.
    starts.sort(key=lambda s: (s["date"], s["gid"]))
    ump_acc = defaultdict(lambda: {"K": 0.0, "BF": 0.0, "g": set()})
    pit_acc = defaultdict(lambda: {"K": 0.0, "BF": 0.0, "n": 0})

    samples = []
    for s in starts:
        u, p = ump_acc[s["ump"]], pit_acc[s["pid"]]
        if (p["n"] >= MIN_PRIOR_STARTS and p["BF"] >= MIN_PRIOR_BF
                and len(u["g"]) >= MIN_UMP_GAMES):
            # Shrunk umpire rate: his own prior games, pulled to league mean.
            ump_rate = ((u["K"] + PRIOR_G * 8.0 * league)
                        / (u["BF"] + PRIOR_G * 8.0))
            samples.append({
                "y": s["K"] / s["BF"],
                "prior_kbf": p["K"] / p["BF"],
                "ump_kbf": ump_rate,
                "date": s["date"], "BF": s["BF"], "K": s["K"],
            })
        u["K"] += s["K"]; u["BF"] += s["BF"]; u["g"].add(s["gid"])
        p["K"] += s["K"]; p["BF"] += s["BF"]; p["n"] += 1

    print(f"scoreable starts: {len(samples):,}")
    if len(samples) < 500:
        print("too few; aborting.")
        return

    samples.sort(key=lambda r: r["date"])
    cut = samples[int(len(samples) * 0.7)]["date"]
    tr = [r for r in samples if r["date"] < cut]
    te = [r for r in samples if r["date"] >= cut]
    print(f"train {len(tr):,} (< {cut})   test {len(te):,} (>= {cut})\n")
    y_te = [r["y"] for r in te]

    print("SINGLE-FEATURE RANK CORRELATION vs actual K/BF (held out)")
    for f in ("prior_kbf", "ump_kbf"):
        tag = "  <- the incumbent" if f == "prior_kbf" else ""
        print(f"  {f:12} rho {spearman([r[f] for r in te], y_te):+.4f}{tag}")

    # How much does the umpire feature even vary? A feature with no spread
    # cannot move a prediction regardless of its correlation.
    uv = sorted(r["ump_kbf"] for r in te)
    print(f"\n  ump_kbf spread (held out): p10 {uv[len(uv)//10]:.4f}  "
          f"median {uv[len(uv)//2]:.4f}  p90 {uv[9*len(uv)//10]:.4f}")
    print(f"  -> p90-p10 = {uv[9*len(uv)//10]-uv[len(uv)//10]:.4f} K/BF, "
          f"about {(uv[9*len(uv)//10]-uv[len(uv)//10])*25:.2f} K over a 25-batter start")

    print("\nHELD-OUT PREDICTION OF K/BF (OLS on train only)")
    print(f"  {'features':26} {'MAE':>9} {'rho':>9}")
    base = None
    for feats, label in ((["prior_kbf"], "prior_kbf (incumbent)"),
                         (["ump_kbf"], "ump_kbf alone"),
                         (["prior_kbf", "ump_kbf"], "prior_kbf + ump_kbf")):
        coef = ols([[r[f] for f in feats] for r in tr], [r["y"] for r in tr])
        pred = [coef[0] + sum(c * r[f] for c, f in zip(coef[1:], feats)) for r in te]
        mae = sum(abs(a - b) for a, b in zip(pred, y_te)) / len(te)
        if base is None:
            base = mae
        print(f"  {label:26} {mae:9.5f} {spearman(pred, y_te):+9.4f}   "
              f"{(base-mae)/base*100:+5.2f}%")

    print("\nPRIOR EXPECTATION WAS: under +0.3% of MAE (see module docstring).")


if __name__ == "__main__":
    sys.exit(main())
