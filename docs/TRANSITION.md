# Transition: from hypothetical pick'em machinery to a pure prediction system

**Date of change: 2026-07-08** (commits `1909945` → `d5ae2b7`)

This document records what the system was, what it measurably got wrong, what
replaced it, and why the replacement is better. It is written in the current
vocabulary; the removed modules are all recoverable from git history.

---

## 1. What it was

Through 2026-07-07 the repo was built around a hypothetical multi-pick
selection loop layered on top of the projections:

- **Combination builder** (`sim.py`) — assembled 2–5 selection sets across
  platforms, priced them with the platforms' published multipliers, computed
  an expected-value figure per set, and sized hypothetical positions with a
  fractional criterion against a notional $1,000 balance.
- **Suppression thresholds** — a row only "counted" if its stated probability
  cleared the multiplier-implied per-selection probability plus a 5% margin;
  days with no qualifying combination produced *no output at all* ("Stop.").
- **$-denominated record** — `log_entries.py`/`grade.py` tracked results in
  dollars: net figures, percentage return, sets-landed counts.
- **State-labelled dashboard** — LIVE / WAITING banners, lock-time
  labels, hypothetical-loop disclaimers.
- **Raw projections trusted at face value** — the strikeout mean
  (`expected_ks`) fed the Negative-Binomial directly; whatever confidence the
  distribution produced was displayed and acted on.

### What it measurably got wrong

The record the old loop kept convicted its own inputs:

| date | stated confidence | realized | gap |
|---|---|---|---|
| 7/5 (pitcher K rows) | 69.0% | 55.6% | −13.5 pts |
| 7/7 (pitcher K rows) | 71.7% | 46.7% | −25.0 pts |

1 of 8 hypothetical sets landed; the notional record went −$70 (−53.8%).
The drift was *worsening day over day*, and each response (dispersion re-fit,
probability ceiling) treated a symptom.

The root cause landed 2026-07-08, when `calibration/fit_mean.py` tested the
raw strikeout projection on **164 frozen pre-game projections** (logged before
their games, settled against official boxscores — no leakage possible):

- The model's chosen sides realized **50.4%** — indistinguishable from noise —
  while stating 61.3%.
- The high-confidence rows (stated ≥ 65%) realized **55.9%** vs 73.4% stated.
- Fitting `mu' = line + s·(mu − line)` gave **s = 0.00**: the projection's
  disagreement with the published line carried *no walk-forward information*.
  The line alone predicted better than the line plus the model.

In short: the machinery was sizing hypothetical positions off confidence the
data had never supported, and its own suppression thresholds guaranteed that
whenever it *did* emit output, that output was drawn from the most
over-confident tail of the model.

---

## 2. What it is now

A prediction system whose only product is numbers and whose only scoreboard
is accuracy:

- **Every matched board row is scored, always** (`scoring.py`): raw predicted
  value, P(more)/P(less) against the published line, lean, and confidence.
  Ranking orders rows; nothing is suppressed. A near-50% probability is
  displayed as exactly that — an honest statement of no information.
- **Point prediction and probability are decoupled.** The displayed
  prediction is the raw model projection (a real number, e.g. 6.83 K). The
  probability comes from the **line-anchored mean**: a continuous shrink
  coefficient (`projection.py`, currently s = 0.00) fitted walk-forward on
  frozen data. When frozen samples show the projection adds information, the
  coefficient rises by exactly the amount the data supports — a fitted
  parameter, not a state.
- **Accuracy record instead of a currency record** (`grade.py`,
  `predictions_log.csv`): hit rate, stated-vs-realized calibration by
  pitcher/batter and by probability bucket, graded automatically from MLB
  StatsAPI final boxscores. The legacy per-row history was migrated in, so
  continuity is preserved.
- **A leakage-proof measurement loop.** Slate projections and FantasyPros
  consensus are frozen daily at generation time (`archive_slate.py`,
  `consensus.py`). New projection sources feed *nothing* until
  `fit_mean.py` fits them a coefficient on those frozen snapshots — the same
  audit that measured the current source at zero.
- **Independent opinions are annotations, not filters.** RotoWire agreement
  is displayed next to each row (`crosscheck.py`); it no longer removes rows.
- **Neutral dashboard.** Accuracy record, highest-confidence table, full
  scored board, and a factual freshness line ("Predictions for …, numbers
  frozen …") in place of state banners.
- **A terminology guard.** `.githooks/pre-commit` blocks reintroduction of
  the removed vocabulary in any committed code, doc, or page.

---

## 3. Why this is better

1. **The probabilities are honest.** The old loop's worst failure mode —
   stating 73% and realizing 56% — cannot recur silently, because stated
   confidence is now bounded by what frozen out-of-sample data has actually
   supported, and the calibration table on the public dashboard shows the gap
   every day.
2. **Output can't be selection-biased.** Suppression thresholds meant the old
   system only spoke when it was most over-confident. Scoring every row
   removes that filter and makes the calibration statistics interpretable —
   the graded sample is now the *whole* board, not a cherry-picked tail.
3. **The measurement loop is leakage-proof by construction.** Every fit input
   is frozen before its game. The r→500 incident (re-projected past slates
   smuggling outcomes into the dispersion fit) and the s=0.00 finding both
   came from enforcing this rule; it is now the only path into the model.
4. **New sources earn influence numerically.** FantasyPros consensus (and any
   future source) enters through one door: a fitted coefficient on frozen
   snapshots. No source is ever trusted at face value again — that is the
   single lesson the whole old record teaches.
5. **The scoreboard matches the product.** A projection system is good when
   stated 65% realizes 65% — hit rate and calibration measure exactly that.
   The old $-figures measured that only indirectly, through multiplier tables
   and set-construction choices that had nothing to do with model quality.
6. **The vocabulary is enforced, not aspirational.** The pre-commit hook
   makes the policy structural: numeric predictions in, numeric accuracy out,
   and no state language anywhere.

---

## 4. What carried over unchanged

- The Negative-Binomial strikeout distribution with frozen-data-fitted
  dispersion (r = 16.6, PIT-validated — `dispersion.py`).
- The batter baseline with matchup/platoon adjustment and its 70% probability
  ceiling for markets without fitted dispersion (`batter_feed.py`,
  `markets.py`).
- The day-factor correlation model (τ ≈ 0.08, `correlation.py`) for the joint
  probability of several leans landing together.
- Board capture with the stale-board freshness guard, hourly cron, frozen
  daily snapshots, and the static dashboard pipeline.
- Platform multiplier tables (`config.py`) — retained purely as reference
  context for how far from 50% a stated probability must be to mean anything.

## 5. What "better" will look like in the record

- Short term: overall hit rate hovers near 50–55% with small stated-vs-
  realized gaps — *correct behavior* for a model whose strikeout source
  currently carries no information beyond the line.
- Within ~1–2 weeks: enough frozen consensus + slate days settle for
  `fit_mean.py` to fit the FantasyPros coefficient. If it fits above zero
  walk-forward, strikeout confidence rises above 50% *by a measured amount*.
- Ongoing: the public calibration line is the contract. If stated and
  realized diverge, the coefficient (mean) or dispersion (spread) gets
  re-fitted on frozen data — numbers move, never modes.
