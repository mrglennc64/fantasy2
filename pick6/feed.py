"""Data feed: model lambda (projected strikeouts) for EVERY probable starter on
a date, so the picker scores the whole board — not just the legs a slate
pre-selects.

PRIMARY SOURCE: the OWNED kmodel (pick6/kmodel.py), StatsAPI-only.
BENCHMARK: strike/mlb-edge, logged alongside but never served.

This order was inverted on 2026-07-19, and the reason is structural rather than
a preference between two estimators.

A system can only learn from a model whose inputs it can see. mlb-edge is an
external service: its features are invisible here, its version is unknowable,
and it cannot be retrained from this repo. Every graded row it produced taught
the system nothing except that the row was right or wrong — no attribution, no
feature-level diagnosis, no path to improvement. The kmodel exposes all eight
of its features per projection (see log_features.py) and retrains from
gamelogs, so its errors are actionable.

Three supporting facts, all measured:
  - the kmodel covers 30/30 probables, so nothing is lost on coverage
  - it is the only estimator with held-out numbers: MAE 1.78 K, bias -0.03 on
    1,893 starts (calibration/train_kmodel.py)
  - mlbedge_predict had served 100% of the board with ZERO graded rows, so the
    correction applied to it was inherited on an assumption, never measured

mlb-edge remains as (a) the benchmark column, the same standing RotoWire has —
displayed, never a filter — and (b) the fallback when the kmodel legitimately
cannot project, which is a starter with fewer than 2 prior starts this season.

Names are accent-folded before matching ("Martín Pérez" == "Martin Perez").
"""
from __future__ import annotations

import json
import time
import unicodedata
import urllib.parse
import urllib.request

BASE = "https://strike.perfecthold.online/api"

# Wall-clock ceiling for benchmark fetching. The benchmark is nice-to-have; the
# board is not. A degraded upstream must cost the run a few seconds, never the
# slate — 2026-07-13 was an upstream failure taking the whole day's board with
# it, and that must not be reachable from a column nobody scores against.
BENCH_BUDGET_S = 90.0

# WHICH estimator produced a projection. This is not bookkeeping: the mean
# correction in projection.py is fitted per estimator, and applying one
# estimator's correction to another's output is a real, silent accuracy bug —
# it happened between 2026-07-13 and 07-18, when the mlb-edge affine
# (mu' = 2.25 + 0.50*mu) kept being applied after the chain had fallen through
# to the owned kmodel, halving the across-pitcher spread of every projection.
SRC_SLATE = "mlbedge_slate"
SRC_PREDICT = "mlbedge_predict"
SRC_KMODEL = "kmodel"


def _get(url: str, timeout: float = 60.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def norm(name: str) -> str:
    """Lowercase, strip accents/punctuation -> stable match key."""
    nk = unicodedata.normalize("NFKD", name)
    nk = "".join(c for c in nk if not unicodedata.combining(c))
    return "".join(c for c in nk.lower() if c.isalpha() or c == " ").strip()


def slate_lambdas(date: str) -> dict[str, float]:
    """norm(pitcher) -> expected_ks for all priced rows on the slate."""
    d = _get(f"{BASE}/v2/slate?date={urllib.parse.quote(date)}")
    out = {}
    for r in d.get("rows", []) or []:
        exp = r.get("expected_ks")
        if exp is not None:
            out[norm(r["pitcher"])] = float(exp)
    return out


def predict_lambda(pitcher: str, date: str, line: float,
                   timeout: float = 60.0) -> float | None:
    """Per-pitcher projection (no odds needed). Shorter timeout when used for
    the benchmark, where 30 sequential calls at 60s each would be unusable."""
    q = urllib.parse.urlencode({"pitcher": pitcher, "line": line, "date": date})
    try:
        d = _get(f"{BASE}/v2/predict?{q}", timeout)
    except Exception:
        return None
    exp = d.get("expected_ks")
    return float(exp) if exp is not None else None


def lambdas_for(board: list[dict], date: str) -> dict[str, dict]:
    """Map each board pitcher -> projection RECORD. Rows need 'pitcher'/'line'.

    Chain: OWNED kmodel -> batch slate -> per-pitcher predict. The upstream
    links now exist only for starters the kmodel cannot project (fewer than 2
    prior starts this season), so the board never loses a row to a rookie.

    Returns keys as the ORIGINAL board pitcher name (so callers can join back).
    Each value is:
        mu            the served projection
        source        WHICH estimator produced it — load-bearing, because the
                      mean correction in projection.py is looked up by it. A
                      projection arriving without provenance gets identity
                      rather than a foreign model's fit.
        version       which fitted params, within that source
        detail        the kmodel's feature vector (None for upstream sources,
                      whose internals we cannot see — which is the whole
                      argument for not serving them)
        bench_mu      what mlb-edge said for the same start, or None. Logged
                      and displayed, never served, never a filter. This is what
                      makes the head-to-head answerable from the record instead
                      of from an opinion.
        bench_source  which upstream endpoint produced bench_mu
    """
    try:
        slate = slate_lambdas(date)
    except Exception:
        slate = {}
    out: dict[str, dict] = {}
    fell_back, benched, deadline = [], 0, time.monotonic() + BENCH_BUDGET_S
    for b in board:
        name = b.get("name") or b.get("pitcher")
        key = norm(name)

        # Benchmark first (cheap when the batch slate covered this start), so
        # it can be attached whichever estimator ends up serving.
        bench, bench_src = slate.get(key), SRC_SLATE if key in slate else ""
        rec = _kmodel_detail(name, date)

        if rec is not None:
            if bench is None and time.monotonic() < deadline:
                bench = predict_lambda(name, date, b["line"], timeout=8.0)
                bench_src = SRC_PREDICT if bench is not None else ""
                benched += bench is not None
            out[name] = {"mu": rec["mu"], "source": SRC_KMODEL,
                         "version": rec.get("version", ""), "detail": rec,
                         "bench_mu": bench, "bench_source": bench_src}
            continue

        # kmodel abstained (< 2 prior starts). Upstream serves this row, and
        # there is no benchmark to record because upstream IS the projection.
        if bench is not None:
            out[name] = {"mu": bench, "source": SRC_SLATE, "version": "",
                         "detail": None, "bench_mu": None, "bench_source": ""}
            fell_back.append(name)
            continue
        lam = predict_lambda(name, date, b["line"])
        if lam is not None:
            out[name] = {"mu": lam, "source": SRC_PREDICT, "version": "",
                         "detail": None, "bench_mu": None, "bench_source": ""}
            fell_back.append(name)
    if fell_back:
        print(f"(upstream served {len(fell_back)} row(s) the owned model could "
              f"not project: {', '.join(fell_back)})")
    print(f"(owned kmodel served {len(out) - len(fell_back)}/{len(out)} rows; "
          f"benchmark attached to {benched + sum(1 for r in out.values() if r['bench_source'] == SRC_SLATE)})")
    return out


_kmodel_broken = False


def _kmodel_detail(name: str, date: str) -> dict | None:
    """Owned-model projection, or None. Import failure and per-pitcher failure
    are reported separately: a broken import silently indistinguishable from
    'this pitcher has < 2 starts' would hide the whole fallback being dead on a
    day the upstream is also down — precisely when it is the only thing left.
    """
    global _kmodel_broken
    try:
        from kmodel import project_detail
    except Exception as e:                       # import-level: fatal, say so
        if not _kmodel_broken:
            _kmodel_broken = True
            print(f"WARNING: kmodel fallback unavailable ({e!r}) — the "
                  f"projection chain has no last link.")
        return None
    try:
        return project_detail(name, date)        # per-pitcher: expected miss
    except Exception:
        return None


def lambdas_flat(board: list[dict], date: str) -> dict[str, float]:
    """name -> mu only, for callers that don't care about provenance."""
    return {k: v["mu"] for k, v in lambdas_for(board, date).items()}
