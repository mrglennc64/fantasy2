"""Fitted strikeout dispersion for the Negative-Binomial side model.

Provenance: MLE fit (calibration/nb.py fit_dispersion) on 147 settled starts in
live_settled.csv (6/28-7/3, 2026). Head-to-head reliability (calibration/
compare.py): NegBinom cut the weighted mean |gap| from 3.2 -> 1.6 pts and fixed
the 60-65% band (Poisson -6.9 pts overconfident -> NB +0.4 pts) that Pick6 legs
live in. Re-fit as the settled sample grows (target n>=400).

Re-validated 2026-07-08 (calibration/refit_dispersion.py + walk_forward.py) on
278 FROZEN starts: on the 147 current-era starts r=16.6 is the exact MLE with
nominal PIT coverage (central-50 50.3%, central-80 81.0%) and the NLL curve is
flat across r in [12, 60] — kept. The 7/7 forward calibration gap traced to the
un-capped batter baseline legs, not K dispersion. CAUTION: never fit r on
/v2/slate re-projections of past dates — the API recomputes with current season
stats (outcome leakage; it fit r->500, i.e. fake Poisson). Fit only on frozen
lambdas: data/slates/<date>.csv (archive_slate.py) or live_settled.csv.
"""
import params

# NB size/dispersion. Var(Y) = mu * (1 + mu/r); r -> inf recovers Poisson.
_DEFAULT_R = 16.6

# The upper bound is not arbitrary: a leaked fit (re-projecting past dates with
# current season stats) drove r to 500 — i.e. fake Poisson, zero overdispersion
# — which is the documented failure mode above. A learned r at the ceiling is a
# leak, not a discovery, so refuse it and keep the fitted value.
DISPERSION_R = params.scalar("dispersion_r", _DEFAULT_R, lo=2.0, hi=200.0)
