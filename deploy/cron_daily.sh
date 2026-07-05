#!/usr/bin/env bash
# Daily Fantasy pipeline — runs on kv8 via cron.
#   1. pull latest (picks up any board CSVs you committed)
#   2. grade any settled paper entries (batter + pitcher markets)
#   3. log today's entries (if a board was captured)
#   4. rebuild the static dashboard and publish it
#
# Install (on kv8, one-time):
#   git clone https://github.com/mrglennc64/Fantasy.git /opt/fantasy
#   crontab -e  ->  add:
#   #  05:00 grade yesterday, 16:30 ET log+publish today's slate
#   30 20 * * *  /opt/fantasy/deploy/cron_daily.sh >> /var/log/fantasy-cron.log 2>&1
set -euo pipefail

REPO="/opt/fantasy"
WWW="/var/www/fantasy"
PY="$(command -v python3)"
DATE="$(TZ=America/New_York date +%F)"

cd "$REPO"
git pull --quiet --ff-only || echo "git pull skipped"

echo "=== $(date -u) daily run for $DATE ==="
"$PY" pick6/grade.py || echo "grade failed"
"$PY" pick6/log_entries.py "$DATE" || echo "log skipped (no board / already logged)"
"$PY" web/build_site.py "$DATE" "$REPO/web/dist/index.html"

install -o www-data -g www-data -m 644 "$REPO/web/dist/index.html" "$WWW/index.html"
# data/pick6_entries.csv is gitignored (VPS-owned runtime state) so daily writes
# never fight `git pull`. Keep a dated backup instead of committing it.
cp -f "$REPO/data/pick6_entries.csv" "$REPO/data/pick6_entries.$(date +%Y%m%d).bak" 2>/dev/null || true
echo "=== published https://fantasy.perfecthold.online ==="
