"""Re-fit the NB strikeout dispersion on settled starts.

    python refit_dispersion.py [start] [end]     # default: last 30 days -> yesterday

mu comes from FROZEN slate archives (data/slates/<date>.csv, written at generation time
by pick6/archive_slate.py) joined to MLB StatsAPI final boxscores (actual K).

*** LEAKAGE WARNING (learned 2026-07-08): for dates with no frozen archive this
falls back to /v2/slate, which RE-PROJECTS past dates with current season stats
— each game's own strikeouts are inside the projection, so the fit collapses
toward Poisson (r=500 on 123 leaked starts vs r=16.6 on frozen ones). Days that
used the leaky fallback are counted and, if any exist, the recommendation is
suppressed. Only trust a fit whose sample is fully frozen. ***

Fits r by MLE (nb.fit_dispersion), then validates walk-forward: for each day,
fit on earlier days only and score the held-out day's mid-PIT coverage against
the CURRENT production r. Reads nothing and writes nothing in pick6/ — paste
the recommended r into pick6/dispersion.py (with provenance) when the held-out
coverage supports it.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from datetime import date as _date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pick6"))
from nb import fit_dispersion, nb_pmf                       # noqa: E402
from dispersion import DISPERSION_R                          # noqa: E402
from grade import final_stats                                # noqa: E402
from feed import norm, slate_lambdas                         # noqa: E402

MIN_TRAIN_STARTS = 40
SLATES = os.path.join(os.path.dirname(__file__), "..", "data", "slates")


def day_lambdas(ds: str) -> tuple[str, dict[str, float]]:
    """('frozen'|'REPROJECTED', norm(pitcher) -> mu). Frozen archive preferred."""
    p = os.path.join(SLATES, f"{ds}.csv")
    if os.path.exists(p):
        rows = csv.DictReader(open(p, encoding="utf-8"))
        return "frozen", {norm(r["pitcher"]): float(r["expected_ks"]) for r in rows}
    try:
        return "REPROJECTED", slate_lambdas(ds)
    except Exception:
        return "REPROJECTED", {}


def collect_pairs(start: str, end: str) -> tuple[dict[str, list[tuple[float, int]]], int]:
    """(date -> [(mu, actual_ks)], leaky_day_count) for settled starts."""
    by_day: dict[str, list[tuple[float, int]]] = defaultdict(list)
    leaky = 0
    d0, d1 = _date.fromisoformat(start), _date.fromisoformat(end)
    d = d0
    while d <= d1:
        ds = d.isoformat()
        src, lams = day_lambdas(ds)
        if lams:
            actuals = final_stats(ds)
            for key, mu in lams.items():
                a = actuals.get(key, {}).get("strikeouts")
                if a is not None:
                    by_day[ds].append((float(mu), int(a)))
            if by_day.get(ds) and src == "REPROJECTED":
                leaky += 1
        print(f"  {ds}: {len(by_day.get(ds, []))} settled starts [{src}]", flush=True)
        d += timedelta(days=1)
    return by_day, leaky


def mid_pit(actual: int, mu: float, r: float) -> float:
    below = sum(nb_pmf(i, mu, r) for i in range(actual))
    return below + 0.5 * nb_pmf(actual, mu, r)


def coverage(pits: list[float]) -> tuple[int, float, float]:
    n = len(pits)
    c50 = sum(1 for p in pits if 0.25 <= p <= 0.75) / n
    c80 = sum(1 for p in pits if 0.10 <= p <= 0.90) / n
    return n, c50, c80


def main() -> None:
    end = sys.argv[2] if len(sys.argv) > 2 else (_date.today() - timedelta(days=1)).isoformat()
    start = sys.argv[1] if len(sys.argv) > 1 else (
        _date.fromisoformat(end) - timedelta(days=29)).isoformat()
    print(f"collecting settled starts {start} -> {end} ...")
    by_day, leaky = collect_pairs(start, end)
    dates = sorted(by_day)
    pairs = [p for d in dates for p in by_day[d]]
    if len(pairs) < MIN_TRAIN_STARTS:
        print(f"only {len(pairs)} starts — not enough to fit. Widen the range.")
        return

    bias = sum(a - mu for mu, a in pairs) / len(pairs)
    r_new = fit_dispersion(pairs)
    print(f"\n{len(pairs)} settled starts over {len(dates)} days")
    print(f"mean projection bias (actual - mu): {bias:+.2f} K "
          f"{'(model OVER-projects)' if bias < -0.25 else '(model UNDER-projects)' if bias > 0.25 else '(roughly unbiased)'}")
    print(f"full-sample MLE r = {r_new:.1f}   (production r = {DISPERSION_R})")

    # walk-forward: held-out PIT coverage under production r vs re-fit-as-you-go
    held_prod, held_wf = [], []
    seen: list[tuple[float, int]] = []
    for d in dates:
        if len(seen) >= MIN_TRAIN_STARTS:
            r_wf = fit_dispersion(seen)
            for mu, a in by_day[d]:
                held_prod.append(mid_pit(a, mu, DISPERSION_R))
                held_wf.append(mid_pit(a, mu, r_wf))
        seen.extend(by_day[d])

    print("\n  HELD-OUT interval coverage      central-50 (~50%)  central-80 (~80%)")
    for tag, pits in ((f"production r={DISPERSION_R}", held_prod),
                      ("walk-forward re-fit", held_wf)):
        if pits:
            n, c50, c80 = coverage(pits)
            print(f"  {tag:28} n={n:>4}   {c50*100:5.1f}%          {c80*100:5.1f}%")

    # ---- does dispersion vary with mu? -------------------------------------
    # The "dispersion by archetype" question, asked in the only form the data
    # can answer: split by PROJECTED mu rather than by hand-labelled tiers like
    # "elite" vs "back-end", which are judgements, not measurements. An ace and
    # a back-end starter genuinely may not share a variance — but one r is the
    # right answer until the bands separate by more than noise.
    #
    # Reported, not shipped. Serving a banded r means p_over() taking r(mu),
    # and that only earns its complexity if held-out coverage actually improves
    # — which this print is here to establish first.
    srt = sorted(pairs, key=lambda p: p[0])
    k = len(srt) // 3
    if k >= MIN_TRAIN_STARTS // 3 and k > 0:
        print("\n  DISPERSION BY PROJECTION BAND (is one r enough?)")
        bands = (("low  ", srt[:k]), ("mid  ", srt[k:2 * k]), ("high ", srt[2 * k:]))
        rs = []
        for tag, chunk in bands:
            if len(chunk) < 20:
                continue
            r_b = fit_dispersion(chunk)
            mus = sum(mu for mu, _ in chunk) / len(chunk)
            pits = [mid_pit(a, mu, DISPERSION_R) for mu, a in chunk]
            _, c50, c80 = coverage(pits)
            rs.append(r_b)
            print(f"    {tag} mean mu {mus:5.2f}   MLE r = {r_b:6.1f}   "
                  f"coverage under production r: {c50*100:4.1f}% / {c80*100:4.1f}%"
                  f"   (n={len(chunk)})")
        if len(rs) >= 2 and min(rs) > 0:
            spread = max(rs) / min(rs)
            if spread < 2.0:
                print(f"    => bands within {spread:.1f}x of each other — one "
                      f"global r is adequate; do NOT add banding.")
            else:
                print(f"    => bands differ by {spread:.1f}x. Worth testing a "
                      f"banded r(mu), but ship it only\n       if held-out "
                      f"coverage beats the single r — r is weakly identified "
                      f"on\n       small samples and the NLL curve is famously "
                      f"flat (see module docstring).")

    if leaky:
        print(f"\n  !! {leaky} day(s) used RE-PROJECTED lambdas (no frozen archive) —")
        print("     outcome leakage inflates r. NO recommendation from this sample;")
        print("     wait until archive_slate.py has covered the whole range.")
    else:
        print(f"\n  => recommended r for pick6/dispersion.py: {r_new:.1f}  (fully frozen sample)")
        print("     (coverage below ~46% under production r but closer under the re-fit")
        print("      means the old r was too confident; update it. Bias is NOT fixed by")
        print("      r — a large bias needs a mean correction in the projection itself.)")


if __name__ == "__main__":
    main()
