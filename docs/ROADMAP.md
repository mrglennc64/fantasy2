# Roadmap: from projection scripts to a prediction operating system

Written 2026-07-19, against measured numbers from the live record — not from
the shape of a good architecture diagram.

The framing in the review is right: the next leap does not come from another
heuristic or another coefficient. It comes from a system that measures its own
error and improves. But the proposed *sequence* would build most of that system
on top of a core whose skill is currently unproven, and several items would
actively overfit at today's sample sizes. This document reorders the work by
what the data can support, and says plainly what to skip.

---

## The reality check

Three measurements have to sit at the top, because they constrain everything.

**1. The projection is decent in absolute terms.**
`calibration/train_kmodel.py` reports held-out MAE **1.78 K**, bias **−0.03 K**
on 1,893 starts. Yesterday's live board independently reproduced it: MAE 1.78 K,
bias +0.33 K on 26 starts. The mean is well-centered. That is real and it is
worth protecting.

**2. Skill against the line is unproven, and the current sample points the
wrong way.** From `calibration/fit_mean.py` on 173 settled frozen starts,
walk-forward on held-out days:

```
correction    n   predicted   realized      gap
raw          63      62.3%      36.5%    -25.8 pts
affine       63      61.5%      42.9%    -18.7 pts
anchor       63      54.2%      49.2%     -5.0 pts
```

All three at or below a coin flip, and the one that shrinks toward the
published line does best. 63 legs over 3 scoreable days is thin — this is not
a verdict. But it is the only line-relative evidence that exists, and it does
not currently support building an ensemble, a Monte Carlo layer, or a
neural network on top.

Note these are two different claims and both can be true. Being unbiased is
not the same as out-predicting a well-informed consensus. The line
aggregates information the model does not have; clearing it is a much higher bar than being well-centered.

**3. The corpus is lopsided.** 11,796 gamelog rows for training. **486 rows of
real book lines, across 8 dates.** So:

- questions about *projection quality and ranking* can be answered now, at scale
- questions about *accuracy relative to the market* cannot, and will only become
  answerable as the board accumulates

That asymmetry should drive the sequence. Validate what is measurable now;
defer what needs data you do not yet have.

---

## The reframe

The review's block diagram is the right target. But the version of it that
matters here is narrower than "add more layers":

**A system can only learn from a model whose inputs it can see.**

Today `pick6/feed.py` tries mlb-edge `/v2/slate`, then `/v2/predict`, and only
then the owned kmodel. The primary source is an external service. Its features
are invisible, its version is unknowable, and it cannot be retrained from this
repo. Every graded row it produces teaches the system nothing except that the
row was right or wrong.

That is the actual blocker on a self-improving platform, and no amount of
Statcast, Monte Carlo, or SHAP fixes it. **Priority 2 below is therefore the
structural change the whole roadmap depends on.**

---

## Phase 0 — Answer "is there signal?" before building on it

**Cost: 2–3 days. Decisive. Everything downstream is unfalsifiable without it.**

`beta = 0` says the ranking has no demonstrated skill, but that verdict rests
on 119 board rows. The same question can be asked of **1,893 held-out gamelog
starts** — 16× the data — and it does not need book lines to answer:

- **Discrimination.** Spearman rank correlation between projected and actual K
  on held-out starts. Does the model order pitchers correctly at all?
- **Distributional calibration.** PIT histogram under NB(mu, r=16.6). Are the
  intervals honest across the mu range?
- **Threshold accuracy without a book.** For synthetic thresholds at each
  half-integer, does `P(K > t)` track realized frequency? This is the
  line-relative question minus the market's information advantage.

Then compare that to the same metrics on the 63 live legs. **The gap between
them is the finding.** If the model ranks well on gamelogs and badly on the
board, the defect is in the serving path, not the model — feature drift,
name-matching, or stale opponent data — and `data/features/<date>.csv`
(already shipping) is what localises it. If it ranks badly in both, the model
needs real work and Phase 3 is the priority.

New: `calibration/validate.py`. Reuses `nb.py`, `kmodel.project_detail()`, and
the existing walk-forward split at `train_kmodel.py:VAL_FROM`.

**Do not skip this.** Every later phase is justified by what it finds.

## Phase 1 — The learning database

**Cost: 2–3 days. This is the review's Priority 1, and it is ~60% built.**

`data/features/<date>.csv` already freezes the serving-time feature vector,
`predictions_log.csv` carries `mu_source` / `mu_version` / `model_p_uncal`, and
`data/params.json` closes the refit loop behind a gate. What is missing is that
these live in three places and cannot be queried together.

Consolidate into **SQLite** — `sqlite3` is in the standard library, so this
respects the no-ML-dependency rule that keeps `pick6/` and `web/` deployable:

```
predictions(date, player_key, market, line, platform, mu_raw, mu_corrected,
            p_uncal, p_final, side, mu_source, mu_version, actual, result,
            error, abs_error)
features(date, player_key, <every feature>, imputed, venue, team, opp)
context(date, player_key, park_k, temp, wind_dir, wind_speed, umpire,
        bullpen_rest, lineup_slot, expected_pa)
runs(id, ts, script, params_hash, n, brier, logloss, promoted, reason)
```

CSVs stay as the write path (append-only, crash-safe, human-readable); the DB
is a derived index rebuilt from them, so it is never the system of record and
can be dropped and regenerated. `runs` is the piece the review does not
mention and that matters most for a self-improving system: **the audit trail of
what the system decided about itself and why.**

Weather and umpire ingestion lands here too — cheap to add, and `umpires.json`
already exists in `stike/mlb-edge/data/` unused.

## Phase 2 — Own the model: make kmodel primary

**Cost: 1–2 days. The structural change everything else depends on.**

Invert `feed.py`'s chain. The owned kmodel becomes the primary projection;
mlb-edge becomes a **benchmark column** logged alongside, exactly as RotoWire
is today — displayed, never a filter, never the served mean.

Why this is the "operating system" move and not a preference:

- Only the owned model exposes its features, so only it can be attributed,
  debugged, or retrained. An external primary makes the learning loop
  decorative.
- It removes an availability dependency that has already taken the board down
  once (2026-07-13).
- It collapses the correction table from three unverified entries to one
  fitted on the estimator that is actually serving.
- `mlbedge_predict` currently serves **100% of the board with zero graded
  rows** — its inherited `(2.25, 0.50)` is an assumption, not a measurement.
  Making it a benchmark retires that risk instead of managing it.

Gate this on Phase 0: if the kmodel ranks *worse* than mlb-edge on held-out
data, do Phase 3 first and invert afterward. But invert.

## Phase 3 — kmodel v2, where the data supports capacity

**Cost: 1–2 weeks. This is where the review's "GBM" belongs.**

The review is right that a straight line cannot model nonlinear error. It is
wrong about *where* to put the flexibility. A GBM on `(raw prediction → actual)`
has ~225 rows and one input; it will fit noise. The kmodel has **11,796 rows**
and room for real capacity. Put the model where the data is.

Features present in the gamelogs and currently **unused**: `throws`, `H`, `BB`,
`ER`, `HR`, `IP`, `pitches`, `opponent`, `opp_k_pa`. Plus:

- **Fix the train/serve mismatch.** `train_kmodel.py:75` trains on
  confirmed-lineup `lineup_k_pct` when available; `kmodel.project()` serves
  team-level `opp_k_pct`. It trains on the better feature and serves the worse
  one. `data/lineups_backfill.csv` (2,914 rows) already has the good one —
  either serve it when the lineup posts, or train on what is servable.
- Pitch count / times-through-order proxies (`pitches`, `BF`, `IP`)
- Umpire strike-zone tendency (`umpires.json`, unused)
- Handedness (`throws`) and platoon composition of the opposing lineup

Bakeoff GLM vs GBM (sklearn offline), ship whichever wins walk-forward on the
existing `VAL_FROM` split. **Serving constraint:** `pick6/` is stdlib-only so
the VPS needs no ML runtime. A GBM must export to plain constants — a compact
JSON of trees walked by ~30 lines of stdlib Python. That is a real limit and it
argues for a shallow, small ensemble, or for a GLM with engineered
interactions if the bakeoff is close.

**Only after this does feature importance earn its place.** For the current GLM
you already have the coefficients; SHAP tells you nothing you cannot read off
`kmodel_params.py`.

## Phase 4 — Calibration and dispersion, once ranking has skill

**Cost: 2–3 days. Blocked on Phase 0/3, deliberately.**

The review's calibration comparison (isotonic / beta / Platt) is the right
menu, at the wrong time. Isotonic on ~119 rows overfits worse than the
2-parameter sigmoid. The gate is already built (`calibration/gate.py`, held-out
Brier **and** log-loss, 1% relative margin); the missing input is a ranking
worth calibrating.

Order of operations, which matters:
1. Ranking demonstrates skill (Phase 0 confirms, Phase 3 delivers if not)
2. `beta` lifts off zero under the existing sigmoid refit
3. **Then** test isotonic/beta against it, at n > 500, on the same gate

Dispersion by archetype is the same story told with variance.
`refit_dispersion.py` now reports MLE `r` per mu band. Ship a banded `r(mu)`
only if held-out PIT coverage beats the single `r` — the NLL curve is famously
flat and `r` is weakly identified on small samples, so bands will *look*
different long before they *are* different.

## Phase 5 — Batters, gated

**Cost: 2–4 weeks. Highest ceiling, lowest current readiness.**

The review ranks Statcast batter features ⭐⭐⭐⭐⭐ on the evidence that "batter
props underperform pitcher props." That evidence does not exist in the current
record: `pick6_today.py:84` filters the board to strikeouts, so batter props
have not been scored, logged, or displayed since commit `c9162b6`. There are 7
archived batter rows total.

When batters do come back, the review's feature list is correct and is the
right list — xwOBA, barrel%, hard-hit%, expected PA, park, weather, bullpen,
lineup slot. The honest sequencing is:

1. Pitcher side clears a real walk-forward bar (Phase 0 + 3 + 4)
2. Statcast ingestion (`pybaseball`, or direct Baseball Savant CSV — not
   currently installed, and it is an *offline* dependency only)
3. Rebuild `batter_feed.py` on expected-PA → xwOBA → outcome distribution
4. Re-enable behind the same per-source correction and gate machinery

Doing this now would double the surface area of an unvalidated system.

---

## What I would skip, and why

**Monte Carlo simulation for single props.** For "will this pitcher exceed 5.5
K," the Negative Binomial CDF is the *exact* answer. Simulating 10,000 draws
approximates a closed form you already compute correctly, adding sampling noise
and runtime for zero information. Monte Carlo becomes genuinely valuable for
**correlated multi-leg** questions — where `correlation.py`'s shared day-factor
already lives — and that is the version worth building, later, once single-leg
probabilities are trustworthy.

**Neural networks.** 11,796 rows and 8–15 features. Off by orders of magnitude.

**Ensembles, for now.** Weighting several models by historical performance
assumes the components have measurable skill to weight. Ensembling estimators
that individually fail to clear 50% produces a well-engineered coin flip. This
becomes correct after Phase 3, not before.

**A GBM on the mean correction.** Covered above: right technique, wrong layer.

**More heuristic rules.** Agreed with the review — ⭐⭐☆☆☆ at best.

---

## Sequence

| Phase | Work | Cost | Blocked by | Why now |
|---|---|---|---|---|
| 0 | Validation harness | 2–3 d | — | Everything else is unfalsifiable without it |
| 1 | Learning DB (sqlite) + weather/umpire | 2–3 d | — | 60% built; makes Phase 0 repeatable |
| 2 | kmodel primary, mlb-edge → benchmark | 1–2 d | 0 | Can't learn from a model you can't see |
| 3 | kmodel v2 (unused features, GBM bakeoff) | 1–2 w | 0, 2 | Capacity belongs where the 11,796 rows are |
| 4 | Calibration + dispersion refits | 2–3 d | 3 | Needs a ranking worth calibrating |
| 5 | Batters + Statcast | 2–4 w | 4 | Highest ceiling, currently dead code |
| — | Correlated multi-leg Monte Carlo | 1 w | 4 | The version of MC that adds information |

Phases 0–2 are about **two weeks** and convert the repo into something that can
actually learn. Phase 3 is where measurable accuracy most likely improves.
Phases 4–5 are real but should not start early.

## The standing rules

Carried forward because they are what makes the rest trustworthy:

- **The line is a benchmark, never the brain.** It enters only as the threshold
  a probability is about, and as a reference column.
- **Nothing ships on in-sample improvement.** Walk-forward, held-out Brier and
  log-loss, both must improve, by a relative margin.
- **Gate refusal keeps production; identity is only the default for the
  unfitted.** Learned the hard way on 2026-07-19.
- **`pick6/` and `web/` stay stdlib-only.** Training may use sklearn/numpy;
  anything reaching serving exports to plain constants or JSON.
- **Report what is live.** Constants no longer live only in git, so the daily
  diagnostic states what is actually serving.
