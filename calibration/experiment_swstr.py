"""Does prior-window swinging-strike rate predict FUTURE strikeouts better than
prior-window K/BF — the feature the served model already uses?

    python experiment_swstr.py

THE QUESTION. Phase 3 step 1 established that the gamelog corpus is exhausted:
seven box-score-derived features bought 1.6% of MAE, and a tuned GBM on the same
features beat the best GLM by 0.013 K. Capacity was not the bottleneck; the
features were. The only way out is information the box score does not carry.

Retrosheet's plays.csv carries it — a pitch-by-pitch sequence per plate
appearance, 99.7% populated for 2024-2025. That yields SwStr% and CSW%, the
standard public "stuff" proxies, without a Statcast dependency.

WHY THE OBVIOUS TEST IS WRONG. Correlating a pitcher's SwStr% against his K/BF
over the SAME window returns rho ~+0.80, which proves nothing: a swinging strike
is mechanically a component of a strikeout. The number is inflated by identity,
not by predictive power.

The honest test is lagged and walk-forward, exactly as train_kmodel.py and
validate.py do it: build every feature from PRIOR starts only, then predict the
strikeout rate of a start the feature has never seen. A feature earns its place
only if it beats prior K/BF — the incumbent — on held-out starts.

SERVING CAVEAT, stated up front. Retrosheet ends at 2025 and is released with a
lag, so nothing here can serve a 2026 board. This experiment decides whether the
SIGNAL is real and worth the cost of finding a live equivalent (MLB StatsAPI
exposes strikePercentage per game log, which the server already fetches). Fitting
on a feature that cannot be served is the mistake this repo has now made twice
— see the "train on what you can serve" standing rule in docs/ROADMAP.md.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(__file__)
RS = os.path.join(HERE, "..", "data", "retrosheet")
PLAYS = [os.path.join(RS, f"{y}plays.csv") for y in (2024, 2025)]
PITCHING = os.path.join(RS, "pitching.csv")

MIN_PRIOR_STARTS = 5     # a rolling rate needs a base before it means anything
MIN_PRIOR_BF = 100
MIN_BF = 15              # the start being predicted must be a real outing

# Pitch codes. Retrosheet encodes one char per pitch; these are the strike
# events. S/M/Q are all swinging strikes (M = missed bunt, Q = swinging on a
# pitchout); T is a foul tip, which is a whiff the batter got a piece of.
SWING_MISS = set("SMQ")
CALLED = set("C")


def per_start_pitch_features() -> dict[tuple[str, str], dict]:
    """(gid, pitcher) -> pitch counts, from the play-by-play."""
    out: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"pit": 0, "sw": 0, "cs": 0})
    for path in PLAYS:
        if not os.path.exists(path):
            print(f"missing {path}")
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                p = row.get("pitcher")
                seq = row.get("pitches") or ""
                if not p or not seq:
                    continue
                d = out[(row["gid"], p)]
                d["pit"] += len(seq)
                for c in seq:
                    if c in SWING_MISS:
                        d["sw"] += 1
                    elif c in CALLED:
                        d["cs"] += 1
    return out


def starts_from_pitching() -> list[dict]:
    """One row per STARTED outing, 2024-2025, with the strikeout outcome."""
    rows = []
    with open(PITCHING, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r.get("p_gs") != "1" or r.get("stattype") not in (None, "", "value"):
                continue
            d = r.get("date") or ""
            if not (d.startswith("2024") or d.startswith("2025")):
                continue
            try:
                bf = float(r["p_bfp"] or 0)
                k = float(r["p_k"] or 0)
            except ValueError:
                continue
            if bf < MIN_BF:
                continue
            rows.append({"gid": r["gid"], "pid": r["id"], "date": d,
                         "BF": bf, "K": k})
    rows.sort(key=lambda x: (x["pid"], x["date"]))
    return rows


def spearman(xs: list[float], ys: list[float]) -> float:
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


def ols(X: list[list[float]], y: list[float]) -> list[float]:
    """Least squares with an intercept, via normal equations + Gaussian
    elimination. Stdlib only, matching the rest of calibration/."""
    n, p = len(X), len(X[0]) + 1
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for xi, yi in zip(X, y):
        v = [1.0] + xi
        for i in range(p):
            b[i] += v[i] * yi
            for j in range(p):
                A[i][j] += v[i] * v[j]
    for i in range(p):                       # Gaussian elimination
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
    print("loading play-by-play pitch sequences...")
    pf = per_start_pitch_features()
    print(f"  {len(pf):,} (game, pitcher) outings with pitch data")

    starts = starts_from_pitching()
    print(f"  {len(starts):,} started outings 2024-2025 with >= {MIN_BF} BF")

    # Attach pitch counts; walk each pitcher forward accumulating PRIOR totals.
    samples = []
    cur_pid = None
    acc = None
    for s in starts:
        if s["pid"] != cur_pid:
            cur_pid = s["pid"]
            acc = {"K": 0.0, "BF": 0.0, "pit": 0, "sw": 0, "cs": 0, "n": 0}
        d = pf.get((s["gid"], s["pid"]))
        # Predict THIS start from what came before it.
        if (acc["n"] >= MIN_PRIOR_STARTS and acc["BF"] >= MIN_PRIOR_BF
                and acc["pit"] > 0):
            samples.append({
                "y": s["K"] / s["BF"],
                "prior_kbf": acc["K"] / acc["BF"],
                "prior_swstr": acc["sw"] / acc["pit"],
                "prior_csw": (acc["cs"] + acc["sw"]) / acc["pit"],
                "date": s["date"], "BF": s["BF"], "K": s["K"],
            })
        acc["K"] += s["K"]
        acc["BF"] += s["BF"]
        acc["n"] += 1
        if d:
            acc["pit"] += d["pit"]
            acc["sw"] += d["sw"]
            acc["cs"] += d["cs"]

    print(f"  {len(samples):,} starts predictable from >= {MIN_PRIOR_STARTS} prior starts\n")
    if len(samples) < 500:
        print("too few samples; aborting.")
        return

    # Walk-forward split by date, same shape as train_kmodel.py's VAL_FROM.
    samples.sort(key=lambda r: r["date"])
    cut = samples[int(len(samples) * 0.7)]["date"]
    tr = [r for r in samples if r["date"] < cut]
    te = [r for r in samples if r["date"] >= cut]
    print(f"train {len(tr):,} starts (< {cut})   test {len(te):,} starts (>= {cut})\n")

    y_te = [r["y"] for r in te]

    print("SINGLE-FEATURE RANK CORRELATION vs actual K/BF (held-out starts)")
    for f in ("prior_kbf", "prior_swstr", "prior_csw"):
        rho = spearman([r[f] for r in te], y_te)
        tag = "  <- the incumbent" if f == "prior_kbf" else ""
        print(f"  {f:12} rho {rho:+.4f}{tag}")

    print("\nHELD-OUT PREDICTION OF K/BF (OLS fitted on train only)")
    print(f"  {'features':28} {'MAE':>8} {'rho':>8}")
    combos = [
        (["prior_kbf"], "prior_kbf (incumbent)"),
        (["prior_swstr"], "prior_swstr"),
        (["prior_csw"], "prior_csw"),
        (["prior_kbf", "prior_swstr"], "prior_kbf + prior_swstr"),
        (["prior_kbf", "prior_csw"], "prior_kbf + prior_csw"),
        (["prior_kbf", "prior_swstr", "prior_csw"], "all three"),
    ]
    base_mae = None
    for feats, label in combos:
        coef = ols([[r[f] for f in feats] for r in tr], [r["y"] for r in tr])
        pred = [coef[0] + sum(c * r[f] for c, f in zip(coef[1:], feats)) for r in te]
        mae = sum(abs(p - a) for p, a in zip(pred, y_te)) / len(te)
        rho = spearman(pred, y_te)
        if base_mae is None:
            base_mae = mae
        gain = (base_mae - mae) / base_mae * 100
        print(f"  {label:28} {mae:8.5f} {rho:+8.4f}   {gain:+5.2f}% vs incumbent")

    # Same thing in strikeout COUNTS, which is what the board actually prices.
    print("\nSAME, EXPRESSED AS K PER START (rate x that start's BF)")
    for feats, label in (combos[0], combos[3]):
        coef = ols([[r[f] for f in feats] for r in tr], [r["y"] for r in tr])
        errs = []
        for r in te:
            p = coef[0] + sum(c * r[f] for c, f in zip(coef[1:], feats))
            errs.append(abs(p * r["BF"] - r["K"]))
        print(f"  {label:28} MAE {sum(errs)/len(errs):.4f} K")
    print("\n  (kmodel's held-out MAE on its own split: 1.754 K — not directly"
          "\n   comparable, since this uses actual BF rather than projecting it.)")


if __name__ == "__main__":
    sys.exit(main())
