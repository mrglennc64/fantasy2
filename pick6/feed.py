"""Phase 1 data feed: model lambda (projected strikeouts) for EVERY probable
starter on a date, so the Pick6 picker scores the whole DK board — not just the
4-leg card the slate pre-selects.

Primary source: strike/mlb-edge `/v2/slate?date=D` returns ~30 `rows` (all priced
starts) each with `expected_ks`. Fallback: `/v2/predict?pitcher=..&date=..` for a
board pitcher missing from the slate rows (e.g. no book odds that day).

Names are accent-folded before matching ("Martín Pérez" == "Martin Perez").
"""
from __future__ import annotations

import json
import unicodedata
import urllib.parse
import urllib.request

BASE = "https://strike.perfecthold.online/api"

# WHICH estimator produced a projection. This is not bookkeeping: the mean
# correction in projection.py is fitted per estimator, and applying one
# estimator's correction to another's output is a real, silent accuracy bug —
# it happened between 2026-07-13 and 07-18, when the mlb-edge affine
# (mu' = 2.25 + 0.50*mu) kept being applied after the chain had fallen through
# to the owned kmodel, halving the across-pitcher spread of every projection.
SRC_SLATE = "mlbedge_slate"
SRC_PREDICT = "mlbedge_predict"
SRC_KMODEL = "kmodel"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
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


def predict_lambda(pitcher: str, date: str, line: float) -> float | None:
    """Per-pitcher projection fallback (no odds needed)."""
    q = urllib.parse.urlencode({"pitcher": pitcher, "line": line, "date": date})
    try:
        d = _get(f"{BASE}/v2/predict?{q}")
    except Exception:
        return None
    exp = d.get("expected_ks")
    return float(exp) if exp is not None else None


def lambdas_for(board: list[dict], date: str) -> dict[str, dict]:
    """Map each board pitcher -> projection RECORD. Rows need 'pitcher'/'line'.

    Chain: batch slate -> per-pitcher predict -> OWNED kmodel (StatsAPI-only).
    The last link means a dead upstream service degrades the source, never
    the app (7/13: empty slate then HTTP 500 killed the whole day's board).
    Returns keys as the ORIGINAL board pitcher name (so callers can join back).

    Each value is {"mu", "source", "version", "detail"} rather than a bare
    float. `source` is the load-bearing field — downstream the mean correction
    is looked up by it, so a projection that arrives without its provenance
    gets the identity correction rather than a foreign model's fit. `detail`
    carries the kmodel's feature vector for the sidecar; it is None for
    upstream sources, whose internals we don't see.
    """
    try:
        slate = slate_lambdas(date)
    except Exception:
        slate = {}
    out: dict[str, dict] = {}
    km = 0
    for b in board:
        name = b.get("name") or b.get("pitcher")
        key = norm(name)
        if key in slate:
            out[name] = {"mu": slate[key], "source": SRC_SLATE,
                         "version": "", "detail": None}
            continue
        lam = predict_lambda(name, date, b["line"])
        if lam is not None:
            out[name] = {"mu": lam, "source": SRC_PREDICT,
                         "version": "", "detail": None}
            continue
        rec = _kmodel_detail(name, date)
        if rec is not None:
            out[name] = {"mu": rec["mu"], "source": SRC_KMODEL,
                         "version": rec.get("version", ""), "detail": rec}
            km += 1
    if km:
        print(f"(kmodel fallback projected {km} pitchers — upstream slate had "
              f"{len(slate)} rows)")
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
