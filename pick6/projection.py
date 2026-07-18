"""Line-free mean correction for the strikeout projection, PER SOURCE.

ARCHITECTURAL RULE (2026-07-11): the market can be a benchmark; it cannot be
the model's brain. The published line must never be used as:
  - expected mean
  - model input feature
  - calibration target
  - probability anchor
  - regression baseline
It appears exactly twice downstream, both legitimate: as the threshold the
probability is ABOUT (P(actual > line) is the question being answered), and
as a reference column next to the prediction on the dashboard.

The correction is a global affine recentering fitted ONLY on frozen
(projection, actual) pairs — no line anywhere in the fit:

    mu' = a + b * mu_raw

WHY IT IS KEYED BY SOURCE (2026-07-18). A correction describes ONE estimator's
bias. It is not a property of the strikeout market, and it does not transfer.
The original a=+2.25, b=0.50 was fitted on 190 frozen mlb-edge starts and is a
true statement about mlb-edge. But on 2026-07-13 that upstream died and
feed.py began falling through to the owned kmodel, which trains out unbiased
(MAE 1.78 K, bias -0.03 on 1,893 held-out starts) — and scoring.py kept
applying the mlb-edge affine to it for five days.

The damage was not the level. The affine's fixed point is a/(1-b) = 4.5, which
sits in the middle of the kmodel's realistic output range, so the mean barely
moved. b = 0.50 HALVED THE SPREAD: roughly [3.5, 6.5] -> [4.0, 5.5]. Every
probability collapsed toward the base rate and the confidence ranking went
flat. When a correction is wrong, b is the parameter that hurts.

So: identity is the default for any source we have not fitted, and an
unrecognised source gets identity rather than inheriting whatever fit happens
to be nearest. calibration/fit_mean.py fits and gates each source separately
and will not promote a correction until that source's own frozen data both
beats raw out-of-sample and excludes b = 1.
"""
from __future__ import annotations

import params

IDENTITY = (0.00, 1.00)

_DEFAULTS = {
    # mlb-edge upstream: refit 2026-07-11, 190 frozen pre-game starts, NB MLE.
    # Walk-forward stated 61.1% / realized 52.9% (raw was 50.4%).
    "mlbedge_slate":     (2.25, 0.50),
    "mlbedge_predict":   (2.25, 0.50),   # same estimator, different endpoint
    # Owned GLM: held-out bias -0.03 K on 1,893 starts. There is no bias to
    # correct, and no frozen serving history yet to fit one from.
    "kmodel":            IDENTITY,
    # Batter means are built line-free in batter_feed.py and pass through.
    "statsapi_baseline": IDENTITY,
}

# Learned fits from calibration/fit_mean.py override the defaults per source.
# Bounds are generous but finite: b is what changes the spread between
# pitchers, so a b outside [0, 2] is a fitting failure, not a discovery.
CORRECTIONS = params.pairs("mean_correction", _DEFAULTS,
                           lo_a=-5.0, hi_a=5.0, lo_b=0.0, hi_b=2.0)

# Back-compat: calibration/fit_mean.py reports production constants by these
# names. They describe the mlb-edge entry specifically — there is no longer a
# single global affine to name.
AFFINE_A, AFFINE_B = CORRECTIONS["mlbedge_slate"]


def corrected_mu(market: str, mu: float, source: str = "unknown") -> float:
    """Independent mean used for probability computation.

    IDENTITY as the lookup default is the safety property, not a convenience:
    an unknown or misspelt source can never inherit another estimator's fit,
    which is the exact failure this module now exists to prevent. Non-strikeout
    markets pass through (their means are already line-free).
    """
    if market != "strikeouts":
        return mu
    a, b = CORRECTIONS.get(source, IDENTITY)
    return a + b * mu
