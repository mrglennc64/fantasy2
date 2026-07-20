# Roadmap: from projection scripts to a prediction operating system

Rewritten 2026-07-20, against measured numbers from the live record — not from
the shape of a good architecture diagram.

The 2026-07-19 version of this document sequenced the work by what the data
*could* support. Phases 0, 2 and the first step of 3 have since run, and two of
them returned negative results decisive enough to invalidate the sequence they
came from. This rewrite records what was learned and re-sequences around it.

**The headline change: capacity is no longer the bottleneck — data is.** The
old Phase 3 ("put the flexibility where the 11,796 rows are") was tested and
did not pay. Everything below follows from that.

---

## What we now know

### The projection is well-centered and ranks real — barely better than naive

`calibration/validate.py` on 1,098 held-out starts:

```
MAE            1.754 K   vs naive 1.774 K      -> +1.2% over "recent K/BF x recent BF"
Spearman rho   +0.409    vs naive +0.369       -> 13.5 SE from zero: real, but marginal gain
PIT coverage   55.4% / 84.4% (nominal 50/80)   -> intervals honest; r=16.6 is fine
own-form bar   696/1098 = 63.4%                -> beats each pitcher's own recent form
real book lines 68/131 = 51.9%                 -> coin flip against a market
```

The last two lines are the whole story. The model **does** know something beyond
a pitcher's recent form (63.4%). That knowledge **does not survive contact with
a market line** (51.9%). A line aggregates information the model does not have,
and clearing it is a far higher bar than being unbiased.

### Functional form is not the bottleneck — this is retired on evidence

`calibration/experiment_v2.py`, seven candidate gamelog features on the
identical split:

```
v1 (8 features, refit)   MAE 1.7527   rho 0.4105
v1 + all candidates      MAE 1.7406   rho 0.4153
v1 + helpful only        MAE 1.7388   rho 0.4173
GBM ceiling check        MAE 1.7260   rho 0.4209
```

A **tuned gradient-boosted tree beats the best GLM by 0.013 K.** Best total gain
over the served model is 1.6% of MAE. Actual K has SD ~2.2, so 0.03 K of MAE
moves P(K > line) by well under a point. 51.9% does not become useful on that.

This retires — on measurement, not argument — the GBM/XGBoost/LightGBM branch,
the ensemble branch, and any "different link function" branch. Only
`ip_per_start` and `k_per_pitch` helped individually, and the box-score corpus
that produced them is close to exhausted.

### The live record is contaminated; the Phase 0 record is not

Until 2026-07-20 the Over threshold used `ceil(line)`, which counts the push as
a win on a whole-number line (fixed in `markets.over_threshold`, commit
`81a3b42`). This inflated P(more) by exactly `pmf(line)` and flipped published
leans — 3 of the 6 whole-number rows on the 2026-07-19 board alone.

The two corpora are affected differently, and the distinction matters:

- **mlb-edge sportsbook export: all 486 lines are half-integer.** Phase 0's
  51.9% and every metric above are unaffected. Re-run post-fix: identical.
- **The live PrizePicks board is mixed** — roughly 30% whole-number lines. So
  the served record (52.7%, n=165) carries a one-directional MORE bias, and the
  calibration fitted on it inherited that bias.

`calibration/threshold_repair.py` supplies corrected probabilities for fitting
without rewriting the served columns. **It has not yet been run on the host.**
Until it is, treat the live hit rate as provisional.

### The live record is not distinguishable from a coin flip

```
Jul 18   55.1%   n=127   95% CI [46.5%, 63.8%]
Jul 19   52.7%   n=165   95% CI [45.1%, 60.3%]
```

Both intervals contain 50%, and the rate fell as n grew — an early run
regressing toward its true value, converging on Phase 0's independent 51.9%.
The reference yardstick implied by the published 2-pick multiplier is **57.7%
per selection** (`config.implied_leg_probability`) — the natural scale for
judging whether a probability estimate is meaningfully far from 50%.
Distinguishing a true 55% from a coin flip needs ~385 graded rows (about a
month at the current ~14/day); distinguishing 57.7% needs ~162, which would
already be visible if it were there.

---

## The reframe, revised

The old framing was right that a system must measure its own error to improve,
and Phase 2 delivered the precondition: the owned kmodel is primary
(`0a29e6b`), so the served model's features are visible, attributable, and
retrainable. That structural blocker is gone.

What replaced it is narrower and harder:

**The model has exhausted what box scores can tell it. The remaining error is
in information the gamelog corpus does not contain.**

Whiff rate, called-strike-plus-whiff, pitch mix, release/velocity trends,
confirmed lineup composition, umpire zone. None are derivable from
`all_starters_gamelogs_2024_2026.csv`. No amount of modeling recovers them.

So the sequence is no longer "add capacity, then calibrate." It is **close the
train/serve gaps you already have, then buy new information, and only then
revisit capacity.**

---

## Phase A — Close the train/serve mismatch (highest gain per unit effort)

**Cost: 1–2 days. Not blocked. Do this first.**

`train_kmodel.py:76` substitutes confirmed-lineup `lineup_k_pct` **into the
`opp_k_pct` slot** when available; `kmodel.py:249` always serves team-level
`opp_k_pct`. One feature name, two different variables. Measured:

```
                     n        mean      SD
lineup_k_pct     2,914      0.2191   0.0138     <- trained on (25% of rows)
opp_k_pct       11,796      0.2231   0.0246     <- served (100% of rows)
```

**The served variable has 1.8x the spread of the one a quarter of the training
rows carried.** A single coefficient is fitted across that blend and then
applied to the wider variable at serve time, so its effect is systematically
mis-scaled — and nothing anywhere reports it, because the column name matches.

`data/lineups_backfill.csv` (2,914 rows, all carrying `lineup_k_pct`) already
holds the good feature. Two honest options, in preference order:

1. **Serve it.** Fetch the confirmed lineup when it posts and build
   `lineup_k_pct` at serve time, falling back to `opp_k_pct` with an explicit
   `imputed` flag. Lineups post ~3h before first pitch; the cron already polls
   every 30 min from 15:00 UTC.
2. **Train on what is servable.** Refit on `opp_k_pct` only. Loses information
   but removes the mismatch immediately, and is the correct fallback if lineup
   timing proves unreliable.

This is a *correctness* fix, not a capacity one, which is exactly why it is
first: it costs days, not weeks, and nothing downstream is trustworthy while
the served feature distribution differs from the trained one.

While here: audit for other train/serve divergences the same way. The
2026-07-13 mlb-edge affine being applied to owned-kmodel projections
(`f05c912`) and this lineup mismatch are the same class of defect, found twice.

## Phase B — Run the repair, re-read the record

**Cost: hours. Not blocked. Do alongside A.**

On the host: `python calibration/threshold_repair.py` (read-only) to size how
many logged rows carried the inflated probability and how many leans flipped.
Then decide whether the calibration needs a forced refit or simply needs to
re-accumulate. `refit_calibration.py` already repairs rows at fit time.

The output is also the honest restatement of the accuracy record, which every
number in this document depends on.

## Phase C — New information, or accept the ceiling

**Cost: 1–3 weeks. Blocked by A. This is the only remaining path to demonstrated
skill against a market line.**

Phase 3 step 1 established that the gamelog corpus is spent. The features that
would plausibly move a strikeout projection all live outside it:

- **Whiff% / CSW%** — the single best public "stuff" proxy. Baseball Savant.
- **Pitch mix and velocity trend** — regime changes the rolling K/BF lags.
- **Umpire zone tendency** — `umpires.json` already sits unused in
  `stike/mlb-edge/data/`. Cheapest of the four.
- **Confirmed lineup composition** — platoon splits and per-batter K%, which
  Phase A's option 1 partly delivers as a side effect.

Sequenced by cost: umpire (free, local) -> lineup composition (falls out of A)
-> Savant whiff/CSW (needs ingestion) -> pitch mix.

`strikePercentage` deserves a specific note: it is likely the best stuff proxy
reachable without a new data source, but it is **gameLog-only** — present in
what the server fetches, absent from the training CSV. Training on it before
the training corpus carries it would repeat exactly the mistake Phase A exists
to fix. Backfill it into training first, or leave it.

**The honest gate on this phase:** if whiff/CSW plus umpire plus lineup do not
move real-line accuracy off 51.9% on a walk-forward split, the correct
conclusion is that this market is efficient at this feature budget, and the
project's value is the measurement apparatus rather than skill against the
line. That is a
legitimate outcome and should be stated rather than engineered around.

## Phase D — The learning database

**Cost: 2–3 days. Not blocked, but no longer urgent.**

Unchanged in design from the previous roadmap and still ~60% built:
`data/features/<date>.csv` freezes serving-time features, `predictions_log.csv`
carries `mu_source` / `mu_version` / `model_p_uncal`, `data/params.json` closes
the refit loop behind a gate. What is missing is that they cannot be queried
together.

Consolidate into **SQLite** (stdlib, so `pick6/` and `web/` stay deployable):
`predictions`, `features`, `context`, and `runs` — the last being the audit
trail of what the system decided about itself and why.

CSVs stay the write path (append-only, crash-safe, human-readable); the DB is a
derived index, rebuildable and never the system of record.

**Demoted from Phase 1 to here deliberately.** It makes analysis repeatable, but
it does not make the model better, and the previous roadmap's own findings came
out of one-off scripts without it. Do it when the analysis cadence justifies it.

## Phase E — Calibration and dispersion

**Cost: 2–3 days. Still blocked, now explicitly on C.**

Unchanged and still correct: isotonic/beta/Platt on ~165 rows overfits worse
than the 2-parameter sigmoid. The gate exists (`calibration/gate.py`, held-out
Brier *and* log-loss, 1% relative margin). The missing input is a ranking worth
calibrating.

Order of operations:
1. Real-line accuracy clears 50% on a walk-forward split (Phase C)
2. `beta` lifts off zero under the existing sigmoid refit
3. **Then** test isotonic/beta against it, at n > 500, on the same gate

PIT coverage is already nominal (55.4% / 84.4%), so banded `r(mu)` has no
measured problem to solve. Ship it only if held-out coverage beats the single
`r` — the NLL curve is flat and `r` is weakly identified at these samples.

## Phase F — Batters

**Cost: 2–4 weeks. Unchanged: highest ceiling, lowest readiness, still gated.**

`pick6_today.py:84` filters the board to strikeouts, so batter props have not
been scored, logged, or displayed since `c9162b6`. Seven archived rows exist.
The feature list (xwOBA, barrel%, hard-hit%, expected PA, park, weather,
bullpen, lineup slot) is right; the sequencing is:

1. Pitcher side clears a real walk-forward bar (A + C + E)
2. Statcast ingestion (offline dependency only)
3. Rebuild `batter_feed.py` on expected-PA -> xwOBA -> outcome distribution
4. Re-enable behind the same per-source correction and gate machinery

Doing this now doubles the surface area of an unvalidated system.

Note `batter_feed.py` currently holds the worst error handling in the repo —
handlers guarding 21 and 32 lines, all silent. It is dormant, so this is not
urgent, but it must be fixed *before* re-enabling, not after.

---

## Retired, with the evidence that retired it

| Branch | Why it is closed |
|---|---|
| GBM/XGBoost on the mean correction | GBM beat the best GLM by 0.013 K on the same features. Wrong layer *and* insufficient ceiling. |
| Ensembles | Weighting estimators that individually sit at 51.9% produces a well-engineered coin flip. |
| Neural networks | 11,796 rows, 8–15 features. Off by orders of magnitude. |
| More gamelog-derived features | Seven tested; only two helped; 1.6% MAE total. Corpus exhausted. |
| Monte Carlo for single props | The NB CDF is the exact answer. MC adds sampling noise to a closed form. Still correct for **correlated multi-leg**, later. |
| More heuristic rules | No mechanism by which they would beat a fitted model that already ranks at rho 0.41. |

---

## Sequence

| Phase | Work | Cost | Blocked by | Why now |
|---|---|---|---|---|
| A | Fix train/serve lineup mismatch | 1–2 d | — | Serving a distribution the model wasn't fitted for |
| B | Run threshold repair, re-read record | hrs | — | Every number here depends on it |
| C | New information (umpire, lineup, Savant) | 1–3 w | A | Only remaining path to skill against a line |
| D | Learning DB (sqlite) | 2–3 d | — | Repeatability; does not improve the model |
| E | Calibration + dispersion | 2–3 d | C | Needs a ranking worth calibrating |
| F | Batters + Statcast | 2–4 w | E | Highest ceiling, currently dead code |
| — | Correlated multi-leg Monte Carlo | 1 w | E | The version of MC that adds information |

A and B are days and are unblocked. C is the decision point: it either produces
skill against the line, or establishes there is none at this feature budget.

## The standing rules

Carried forward unchanged — they are what makes the rest trustworthy.

- **The line is a benchmark, never the brain.** It enters only as the threshold
  a probability is about, and as a reference column.
- **Nothing ships on in-sample improvement.** Walk-forward, held-out Brier *and*
  log-loss, both must improve, by a relative margin.
- **Gate refusal keeps production; identity is only the default for the
  unfitted.** Learned the hard way on 2026-07-19.
- **`pick6/` and `web/` stay stdlib-only.** Training may use sklearn/numpy;
  anything reaching serving exports to plain constants or JSON.
- **Report what is live.** Constants no longer live only in git, so the daily
  diagnostic states what is actually serving.
- **Train on what you can serve.** Added 2026-07-20. Violated twice already —
  the mlb-edge affine on kmodel projections, and `lineup_k_pct`. Both were
  silent, and both were found by accident rather than by a check.
- **A negative result is a deliverable.** Phase 0 and Phase 3 step 1 each cost
  days and produced no accuracy. They are the two most valuable things in this
  document, because they closed branches that would have cost weeks.
