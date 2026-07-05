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

## Calibration status — Phase 2 DONE (Negative-Binomial)

`calibration/compare.py` fits the K overdispersion by MLE on 147 settled starts
(`live_settled.csv`, 6/28–7/3) and compares side-probability reliability:

| Model | Weighted mean \|gap\| | 60–65% band (Pick6 zone) |
|---|--:|--:|
| Poisson (old) | 3.2 pts | −6.9 pts (overconfident) |
| **NegBinomial (fitted r=16.6)** | **1.6 pts** | **+0.4 pts (calibrated)** |
| NB + 0.85 λ-shrink | 2.9 pts | −4.9 (overcorrects — rejected) |

Fitted dispersion `r=16.6` ⇒ K variance ≈1.32× Poisson (real 10-K tails). This
is now the live model: `pick6/dispersion.py` holds `r`, `pick6/sim.p_more` uses
NB. The old Poisson result (which mirrored the 7/4 Imanaga "62%→miss" failure)
is preserved as the baseline in `calibration/backtest.py`.

Residual: the 90%+ bucket is still ~5 pts overconfident, but such legs imply
DK's line is wildly off and are rare in practice. Re-fit `r` as settled n grows
(target ≥400).

## Layout

```
pick6/config.py        multiplier table + breakeven + entry-EV math
pick6/dispersion.py    fitted NB dispersion r (from calibration)
pick6/sim.py           leg scoring (NB P(More/Less)), availability + same-game guards, entry builder, outcome matrix
pick6/feed.py          full-slate λ feed (/v2/slate rows + /v2/predict fallback, accent-folded names)
pick6/pick6_today.py   join λ to the DK board, score whole board, step down 3->2 picks, build entries
calibration/nb.py      NB pmf + MLE fit of the dispersion
calibration/compare.py Poisson vs NB reliability head-to-head
calibration/backtest.py  Poisson baseline reliability (kept for reference)
data/pick6_board_*.csv   captured DK Pick6 boards (line + per-leg boosts + More/Less availability)
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
- **Phase 2 (done):** NegBinomial `p_more` fitted on settled data; mean |gap|
  3.2 → 1.6 pts. Re-fit r as the sample grows.
- **Phase 1 (done):** `/v2/slate` actually returns ~30 `rows` (whole slate), not
  just the 4-leg card — `feed.py` reads those (accent-folded name match, per-
  pitcher `/v2/predict` fallback) so the picker scores all 12 board pitchers.
  Also fixed two real bugs: enforce DK More/Less availability (was recommending
  unofferable sides) and step 3→2 picks when the board is thin. Still manual:
  DK board capture from screenshot — automate next.
- **Phase 3:** cross-check every leg against a second projection (RotoWire
  Props-vs-Projections); only play legs where both agree on the same side.
- **Phase 4:** entry construction (short 2–3 pick sets, correlation, contrarian
  fades) per thelines.com Pick6 strategy.
- **Phase 5:** log `pick6_entries.csv`, grade daily, prove ROI on paper before
  real stakes.
