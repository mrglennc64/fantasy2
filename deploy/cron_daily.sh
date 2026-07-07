#!/usr/bin/env bash
# Fantasy pipeline — runs HOURLY as the non-root `fantasy` user (poll-safe).
#   pull -> grade settled entries -> scrape real board (only until captured) ->
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
git pull --quiet --ff-only || echo "git pull skipped (local changes / offline)"
# Secrets (FIRECRAWL_API_KEY=fc-...) live in /opt/fantasy/.env (gitignored).
[ -f "$REPO/.env" ] && { set -a; . "$REPO/.env"; set +a; }

echo "=== $(date -u) daily run for $DATE ==="
"$PY" pick6/grade.py            || echo "grade failed"

# Auto-scrape today's REAL PrizePicks board via Firecrawl. POLL-SAFE: this script
# runs hourly, but only scrapes until it captures today's board — once captured
# it skips (so it doesn't burn Firecrawl credits or re-scrape after you'd bet).
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

# Only rebuild + publish once today's board exists — otherwise leave the last
# good card up (don't wipe it with an empty "no board yet" page).
if [ -f "$REPO/data/boards/$DATE.csv" ]; then
    "$PY" pick6/log_entries.py "$DATE" || echo "log skipped (already logged)"
    "$PY" web/build_site.py "$DATE" "$REPO/web/dist/index.html"
    install -m 644 "$REPO/web/dist/index.html" "$WWW/index.html"
    echo "published $DATE card"
else
    echo "no board for $DATE yet — leaving the last published card up"
fi

# housekeeping: dated backup of the record, prune backups older than 14 days.
cp -f "$REPO/data/pick6_entries.csv" "$REPO/data/pick6_entries.$(date +%Y%m%d).bak" 2>/dev/null || true
find "$REPO/data" -name 'pick6_entries.*.bak' -mtime +14 -delete 2>/dev/null || true
echo "=== published https://fantasy.perfecthold.online ==="
