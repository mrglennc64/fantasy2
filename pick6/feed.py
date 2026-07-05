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


def lambdas_for(board: list[dict], date: str) -> dict[str, float]:
    """Map each board pitcher -> lambda. board rows need 'pitcher' and 'line'.

    Returns keys as the ORIGINAL board pitcher name (so callers can join back).
    """
    slate = slate_lambdas(date)
    out: dict[str, float] = {}
    for b in board:
        key = norm(b["pitcher"])
        if key in slate:
            out[b["pitcher"]] = slate[key]
        else:
            lam = predict_lambda(b["pitcher"], date, b["line"])
            if lam is not None:
                out[b["pitcher"]] = lam
    return out
