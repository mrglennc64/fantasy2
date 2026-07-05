# Fantasy — DraftKings Pick6 pitcher-strikeout edge

Tailors the existing **strike / mlb-edge** strikeout model to **DraftKings Pick6**
(pick'em: DK sets one projection line per pitcher, you choose More/Less; a
*power play* pays a fixed multiplier only if **every** leg hits).

> **PAPER ONLY — NOT betting advice.** The single-leg K model over-projects and
> is not yet calibrated (see below). Everything here is machinery + measurement,
> not a proven edge.

## Why Pick6 is a different problem than sportsbook props

| | Sportsbook (current mlb-edge) | DraftKings Pick6 |
|---|---|---|
| Line + price | line **and** American odds (vig) | one projection number, **no odds** |
| Edge source | `model_prob − implied_prob` (beat the vig) | `model P(side)` vs DK's soft line |
| Payout | per-leg decimal odds | fixed multiplier by pick count, **all-or-nothing** |
| What matters most | finding mispriced odds | **model calibration** (no market to bail you out) |

Per-leg breakeven for an N-pick power play at multiplier `M`: `p = (1/M)^(1/N)`.

| Picks | Base mult | Breakeven / leg |
|--:|--:|--:|
| 2 | 3× | 57.7% |
| 3 | 6× | 55.0% |
| 4 | 10× | 56.2% |
| 5 | 20× | 54.9% |
| 6 | 35× | 55.3% |

(Verify multipliers live — DK changes them and applies per-leg flex boosts like
the 1.1× / 0.9× seen on the board; `config.entry_multiplier` handles boosts.)

## Calibration status (the make-or-break number)

`calibration/backtest.py` on 147 settled starts (`live_settled.csv`, 6/28–7/3):

```
MAE 2.01 K   bias −0.25 K (model OVER-projects)
pred 0.60-0.65  ->  realized 55.4%   (−6.9 pts, OVERCONFIDENT)
pred 0.90-1.01  ->  realized 89.1%   (−5.0 pts, OVERCONFIDENT)
weighted mean |gap| = 3.2 pts   (want < ~3 before trusting breakevens)
```

**Verdict:** the model is overconfident precisely in the 60–65% band — the exact
band Pick6 legs live in. This is the same failure that sank the 7/4 card
(Imanaga "62% Under" → 8 K). **Fix calibration (Negative-Binomial for K
overdispersion, and/or shrink λ) before staking anything.** That is Phase 2 and
it gates everything downstream.

## Layout

```
pick6/config.py        multiplier table + breakeven + entry-EV math
pick6/sim.py           leg scoring (Poisson P(More/Less)), entry builder, exact outcome matrix
pick6/pick6_today.py   join live mlb-edge λ to the DK board, score & build entries
calibration/backtest.py  reliability test of P(side) vs realized (run this first)
data/pick6_board_*.csv   captured DK Pick6 boards (line + per-leg boosts)
```

Run:
```
python pick6/config.py                 # breakeven reference table
python calibration/backtest.py         # calibration reliability (uses C:\strike-data\...\live_settled.csv)
python pick6/pick6_today.py 2026-07-05 # score today's DK board with the live slate
```

## Data-layer plan (from reference-repo research)

Three repos were reviewed. **None projects strikeouts** — mlb-edge already does
that better — but they supply the ingestion layer:

- **lbenz730/fantasy_baseball** — MLB StatsAPI boxscore scraper + ready-made
  per-start pitcher K logs (2020–2026, ~11k rows) → calibration/training fuel.
  Add `hydrate=probablePitcher` to the schedule call for daily starters.
- **fantasy-toolz/mlb-predictions** — Baseball Savant (Statcast) pitcher CSV
  fetcher + a DraftKings pitcher-name/odds JSON parser → line ingestion + name
  matching. (Skip its team win-prob model.)
- **edwarddistel/yahoo-fantasy-baseball-reader** — tangential (season-long
  fantasy). Keep only its OAuth2 token-refresh pattern if Yahoo data is ever
  needed.

### Roadmap
- **Phase 0 (done):** multiplier/breakeven math, entry builder, outcome matrix.
- **Phase 1:** automate DK Pick6 board capture (currently manual from screenshot)
  + StatsAPI probable-pitcher feed.
- **Phase 2 (critical):** calibrate — NegBinomial `p_more`, refit overdispersion
  on settled data, re-run backtest until mean |gap| < 3 pts.
- **Phase 3:** cross-check every leg against a second projection (RotoWire
  Props-vs-Projections); only play legs where both agree on the same side.
- **Phase 4:** entry construction (short 2–3 pick sets, correlation, contrarian
  fades) per thelines.com Pick6 strategy.
- **Phase 5:** log `pick6_entries.csv`, grade daily, prove ROI on paper before
  real stakes.
