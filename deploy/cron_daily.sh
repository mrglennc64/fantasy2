#!/usr/bin/env bash
# Fantasy pipeline — runs HOURLY as the non-root `fantasy` user (poll-safe).
#   pull -> grade settled predictions -> scrape real board (only until captured) ->
#   log -> rebuild + publish. The card auto-updates the hour PrizePicks posts
#   today's board; off-hour polls are harmless no-ops (freshness guard).
# Writes ONLY inside /opt/fantasy and /var/www/fantasy. See deploy/setup_vps.sh
# for the one-time hardened install (dedicated user, logrotate, crontab).
set -euo pipefail

REPO="/opt/fantasy"
WWW="/var/www/fantasy"
PY="$(command -v python3)"
DATE="$(TZ=America/New_York date +%F)"

cd "$REPO"
# Hard-sync to origin (robust: mode/CRLF drift or a wedged tree never blocks the
# update). reset --hard only touches TRACKED files — gitignored runtime state
# (boards, prediction logs, web/dist, .env) is preserved.
git config core.fileMode false 2>/dev/null || true
git fetch --quiet origin main 2>/dev/null && git reset --hard --quiet origin/main 2>/dev/null \
    || echo "git sync skipped (offline)"
# Secrets (FIRECRAWL_API_KEY=fc-...) live in /opt/fantasy/.env (gitignored).
[ -f "$REPO/.env" ] && { set -a; . "$REPO/.env"; set +a; }

echo "=== $(date -u) daily run for $DATE ==="

# One-time fresh start (2026-07-08): archive the pre-transition record so the
# accuracy log begins today under the new scoring. Nothing is deleted — files
# move to data/archive-pre-2026-07-08/. Self-extinguishing: runs only while
# the legacy per-row history file is still present. Frozen slates + consensus
# are kept (they are the input for the coefficient fits).
if [ -f "$REPO/data/pick6_entries.csv" ]; then
    ARC="$REPO/data/archive-pre-2026-07-08"
    mkdir -p "$ARC"
    mv -f "$REPO"/data/pick6_entries.csv "$REPO"/data/pick6_entries.*.bak \
          "$REPO"/data/predictions_log.csv "$REPO"/data/predictions_log.*.bak \
          "$REPO"/data/boards/*_scored.json \
          "$REPO"/data/boards/2026-07-05*.csv "$REPO"/data/boards/2026-07-06*.csv \
          "$REPO"/data/boards/2026-07-07*.csv "$ARC"/ 2>/dev/null || true
    echo "fresh start: archived pre-transition records -> $ARC"
fi

# grade + publish the daily diagnostic (metrics, tiers, largest misses)
{ "$PY" pick6/grade.py | tee "$REPO/web/dist/diag.txt"; } || echo "grade failed"
install -m 644 "$REPO/web/dist/diag.txt" "$WWW/diag.txt" 2>/dev/null || true

# Archive today's slate lambdas FROZEN (poll-safe: writes once, never rewrites).
# Dispersion re-fits need generation-time projections — /v2/slate re-projects past
# dates with current stats, which leaks outcomes into any fit made later.
"$PY" pick6/archive_slate.py "$DATE" || echo "slate archive failed"

# Freeze FantasyPros daily consensus too (candidate replacement mu source —
# expected_ks tested uninformative vs the line 7/8; the consensus shrink
# coefficient is fitted via fit_mean.py on these frozen snapshots before
# anything reads it at scoring time).
"$PY" pick6/consensus.py "$DATE" || echo "consensus archive failed"

# Auto-scrape today's REAL PrizePicks board via Firecrawl. POLL-SAFE: this script
# runs hourly, but only scrapes until it captures today's board — once captured
# it skips (so it doesn't burn Firecrawl credits or re-scrape after the day's
# numbers were frozen).
# The scraper's freshness guard refuses a stale board, so early polls (before
# PrizePicks posts) are harmless no-ops. Net effect: the card auto-appears within
# the hour of the board going live, not at a fixed clock time.
if [ -n "${FIRECRAWL_API_KEY:-}" ]; then
    if [ -f "$REPO/data/boards/$DATE.csv" ]; then
        echo "today's board already captured — skipping scrape (poll-safe)"
    else
        "$PY" pick6/scrape_firecrawl.py "$DATE" prizepicks || echo "board not live yet / scrape failed — retry next poll"
    fi
else
    echo "no FIRECRAWL_API_KEY in .env — skipping auto-scrape (paste a board instead)"
fi

# Log predictions only when today's board exists; always rebuild + publish —
# build_site labels which slate is shown and when its numbers were frozen, so
# the page is never blank.
[ -f "$REPO/data/boards/$DATE.csv" ] && { "$PY" pick6/log_predictions.py "$DATE" || echo "log skipped (already logged)"; }
"$PY" web/build_site.py "$DATE" "$REPO/web/dist/index.html"
install -m 644 "$REPO/web/dist/index.html" "$WWW/index.html"

# Publish fit reports as static files so constants can be reviewed from
# anywhere. Weekly
# (Sunday, once): the strikeout anchor-s refit with its confidence interval
# on the full frozen archive (hits StatsAPI, so not every hour).
if [ "$(date +%u)" = "7" ] && [ ! -f "$REPO/data/.refit-$(date +%G-%V)" ]; then
    "$PY" calibration/fit_mean.py > "$REPO/web/dist/refit_s.txt" 2>&1 || true
    install -m 644 "$REPO/web/dist/refit_s.txt" "$WWW/refit_s.txt" 2>/dev/null || true
    "$PY" calibration/refit_calibration.py > "$REPO/web/dist/refit_calibration.txt" 2>&1 || true
    install -m 644 "$REPO/web/dist/refit_calibration.txt" "$WWW/refit_calibration.txt" 2>/dev/null || true
    touch "$REPO/data/.refit-$(date +%G-%V)"
fi

# housekeeping: dated backup of the record, prune backups older than 14 days.
cp -f "$REPO/data/predictions_log.csv" "$REPO/data/predictions_log.$(date +%Y%m%d).bak" 2>/dev/null || true
find "$REPO/data" \( -name 'predictions_log.*.bak' -o -name 'pick6_entries.*.bak' \) -mtime +14 -delete 2>/dev/null || true
echo "=== published https://fantasy.perfecthold.online ==="
