#!/usr/bin/env bash
# Daily Fantasy pipeline — runs as the non-root `fantasy` user via cron.
#   pull -> grade settled entries -> scrape real board -> log -> rebuild + publish.
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

# Auto-scrape today's REAL PrizePicks board via Firecrawl (Firecrawl proxies the
# request, so this works from the VPS). Overwrite the day's board for a fresh
# pull; if the key is missing or the scrape fails, fall back to any pasted board.
if [ -n "${FIRECRAWL_API_KEY:-}" ]; then
    rm -f "$REPO/data/boards/$DATE.csv" "$REPO/data/boards/${DATE}_batters.csv"
    "$PY" pick6/scrape_firecrawl.py "$DATE" prizepicks || echo "scrape failed — using pasted board if present"
else
    echo "no FIRECRAWL_API_KEY in .env — skipping auto-scrape (paste a board instead)"
fi

"$PY" pick6/log_entries.py "$DATE" || echo "log skipped (no board / already logged)"
"$PY" web/build_site.py "$DATE" "$REPO/web/dist/index.html"

# publish — fantasy user owns WWW; the file is world-readable so nginx (www-data)
# can serve it. No chown needed, so this works without root.
install -m 644 "$REPO/web/dist/index.html" "$WWW/index.html"

# housekeeping: dated backup of the record, prune backups older than 14 days.
cp -f "$REPO/data/pick6_entries.csv" "$REPO/data/pick6_entries.$(date +%Y%m%d).bak" 2>/dev/null || true
find "$REPO/data" -name 'pick6_entries.*.bak' -mtime +14 -delete 2>/dev/null || true
echo "=== published https://fantasy.perfecthold.online ==="
