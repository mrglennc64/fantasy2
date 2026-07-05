# Fantasy — multi-platform pick'em edge (MLB)

Tailors the **strike / mlb-edge** projections to **pick'em fantasy** props across
platforms — **PrizePicks, Underdog, DraftKings Pick6, Sleeper, Betr, ParlayPlay**
(each sets its own projection line; you pick More/Less; a *power play* pays a
fixed multiplier only if **every** leg hits). The projection is platform-agnostic;
only the LINE and the PAYOUT table differ, so the picker **line-shops** every leg
to the best-paying app and builds the best **2 / 3 / 4 / 5-pick** at each size.
Markets: pitcher strikeouts (calibrated) + batter hits/TB/HR/RBI/runs (baseline).

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
pick6/markets.py       market registry: per-prop distribution (K ready; TB/runs/ER/hits/HR scaffolded)
pick6/sim.py           market-aware leg scoring, availability + same-game guards, entry builder, outcome matrix
pick6/crosscheck.py    RotoWire second-opinion gate (free proj endpoint; drops disagreements)
pick6/correlation.py   day-factor model: correlation-adjusted joint P + outcome matrix
pick6/feed.py          full-slate λ feed (/v2/slate rows + /v2/predict fallback, accent-folded names)
pick6/batter_feed.py   StatsAPI season-rate batter projections (hits/TB/HR/RBI/runs)
pick6/pick6_today.py   join λ to the DK board, score whole board, step down 3->2 picks, build entries
pick6/log_entries.py   append a day's paper entries to data/pick6_entries.csv (idempotent)
pick6/grade.py         grade logged legs vs MLB StatsAPI finals; report ROI + out-of-sample calibration
calibration/nb.py      NB pmf + MLE fit of the dispersion
calibration/compare.py Poisson vs NB reliability head-to-head
calibration/backtest.py  Poisson baseline reliability (kept for reference)
web/build_site.py      generate the static dashboard (entries + track record + calibration)
deploy/fantasy.nginx   nginx server block for fantasy.perfecthold.online
deploy/deploy.sh       build + rsync the dashboard to the VPS
data/pick6_board_*.csv   captured DK Pick6 boards (market + line + boosts + availability)
data/pick6_entries.csv   logged paper entries (graded by grade.py)
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
- **Phase 3 (done):** RotoWire second-opinion gate. Their free JSON endpoint
  (`/betting/mlb/tables/all-bets-props-plus-proj.php?prop=k`) gives a `proj` per
  pitcher; `crosscheck.py` drops any leg RotoWire disagrees with (e.g. 7/5
  Bradish: model More vs RotoWire Under → gated). Legs RotoWire doesn't cover
  pass through flagged "unconfirmed". Coverage is partial (a handful of pitchers/
  day), so it mostly kills bad legs rather than confirming good ones.
- **Multi-market (done):** `markets.py` makes scoring market-aware via the
  board's `market` column. **Pitcher strikeouts** = calibrated (mlb-edge λ +
  fitted NB). **Batter hits / total bases / home runs / RBI / runs** now score
  off a free **StatsAPI season-rate baseline** (`batter_feed.py`): season
  per-AB/PA rates × projected PAs by lineup slot. Batter props are labelled
  **baseline / lower-confidence** — matchup-neutral (ignore the opposing pitcher
  + park) and their dispersion isn't fitted yet. RotoWire cross-checks TB/runs
  for free; hits/HR/RBI stay unconfirmed. Capture batter lines in
  `data/pick6_board_<date>_batters.csv` (DK's separate tabs). The dashboard has a
  Pitcher↔Batter toggle. Entries can mix markets, which lowers correlation.
- **Phase 4 (done):** day-level correlation. Settled data has a real "K
  environment" factor — after removing sampling noise, latent sd **τ≈0.081**
  (`calibration/correlation.py`). `correlation.py` models a shared multiplier
  D~Normal(1,τ) on every leg's λ and integrates it out: same-side entries (all
  Unders) are positively correlated → higher true P(all-hit) but a fatter tail;
  mixed entries score lower. The picker now ranks and sizes on the
  correlation-adjusted EV and flags same-side concentration. Still short-slate
  (2–3 picks) per thelines.com.
- **Phase 5 (built, accumulating):** `log_entries.py` records each day's paper
  entries to `data/pick6_entries.csv`; `grade.py` scores them against MLB
  StatsAPI finals and reports entry ROI + **out-of-sample leg calibration** (the
  real test of the NB fit, which was in-sample on 147 starts). Daily loop:
  `log_entries.py <date>` in the morning, `grade.py` after games settle. Needs a
  couple of weeks of graded entries before any real stake.

### Daily use
```
python pick6/log_entries.py 2026-07-05   # morning: record paper entries
python pick6/grade.py                     # after games: grade + running ROI/calibration
```

## Deploy (fantasy.perfecthold.online)

Static dashboard on the kv8 VPS (46.202.143.253), same host as strike. One-time:
install `deploy/fantasy.nginx`, create `/var/www/fantasy`, run certbot. Then
`deploy/deploy.sh [date]` builds `web/dist/index.html` and rsyncs it up. The page
shows today's entries, all scored legs (Pitcher↔Batter toggle) with the RotoWire
column, and the paper track record (ROI + out-of-sample calibration).

**Daily autopilot (hardened):** run `sudo bash deploy/setup_vps.sh` on kv8 ONCE.
It creates a dedicated **non-root `fantasy` user** that owns `/opt/fantasy` +
`/var/www/fantasy`, installs logrotate, and installs the daily crontab **for
that user** (not root) — so a repo compromise can only touch Fantasy's own
files, not strike/nginx/certs/the rest of the box. The daily `cron_daily.sh`
pulls the repo, grades settled entries, logs today's entries, rebuilds +
publishes the dashboard, keeps a 14-day-pruned backup of the record, and writes
a logrotated log. The entries CSV is gitignored VPS-owned runtime state (kv8 has
no push creds). Capture the DK boards into `data/` and commit; the cron does the
rest. To de-risk further, pin `git pull` to a reviewed commit.
