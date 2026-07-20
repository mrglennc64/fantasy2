"""Negative-Binomial strikeout model: mean-parameterized, with an MLE fit of the
shared dispersion `r` on settled starts.

Mean-parameterized NB with mean mu and size r:
    Var(Y) = mu * (1 + mu / r)              # r -> inf recovers Poisson (Var = mu)
    P(Y=k) = C(k+r-1, k) * (r/(r+mu))**r * (mu/(r+mu))**k

Strikeouts are OVER-dispersed vs Poisson (long right tails: 10+ K games), so a
Poisson P(side) is over-confident at the extremes. Fitting r on real (mu=model
projection, y=actual) pairs shrinks those probabilities toward the calibrated
truth without touching the underlying projection.
"""
from __future__ import annotations

import math

EPS = 1e-9


def nb_logpmf(k: int, mu: float, r: float) -> float:
    mu = max(mu, EPS)
    return (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
            + r * math.log(r / (r + mu)) + k * math.log(mu / (r + mu)))


def nb_pmf(k: int, mu: float, r: float) -> float:
    return math.exp(nb_logpmf(k, mu, r))


def nb_p_more(mu: float, line: float, r: float) -> float:
    """P(K > line) = P(K >= floor(line)+1) under NB(mu, r).

    floor+1, not ceil: whole-number lines push on equality (see
    markets.over_threshold, the canonical definition). Kept inline here only
    because this module is deliberately dependency-free pure math.
    """
    need = math.floor(line) + 1
    cdf = sum(nb_pmf(i, mu, r) for i in range(need))
    return max(0.0, 1.0 - cdf)


def fit_dispersion(pairs: list[tuple[float, float]],
                   lo: float = 0.5, hi: float = 500.0) -> float:
    """MLE of shared r over (mu, y) pairs via golden-section on the log-likelihood.

    Each start keeps its own model mean mu_i; only the dispersion r is fit.
    """
    def nll(r: float) -> float:
        return -sum(nb_logpmf(int(round(y)), mu, r) for mu, y in pairs)

    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(d)
    return (a + b) / 2
