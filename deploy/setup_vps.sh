#!/usr/bin/env bash
# One-time HARDENED install for the Fantasy daily autopilot on kv8.
# Run ONCE as root:  sudo bash deploy/setup_vps.sh
#
# Creates a dedicated non-root `fantasy` user that owns the repo + web dir and
# runs the daily cron. A repo compromise then can only touch Fantasy's own
# files — NOT strike, the other sites, nginx, certs, or the rest of the box.
set -euo pipefail

USER=fantasy
REPO=/opt/fantasy
WWW=/var/www/fantasy
REPO_URL=https://github.com/mrglennc64/Fantasy.git

[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }

# 1. dedicated system user, no login shell
id -u "$USER" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$USER"

# 2. repo owned by fantasy (fresh clone if missing)
if [ -d "$REPO/.git" ]; then
    git config --global --add safe.directory "$REPO" || true
else
    rm -rf "$REPO"; git clone "$REPO_URL" "$REPO"
fi
mkdir -p "$REPO/logs"
chown -R "$USER:$USER" "$REPO"
chmod +x "$REPO"/deploy/*.sh

# 3. web dir owned by fantasy; nginx (www-data) only needs read+traverse (755)
mkdir -p "$WWW"
chown -R "$USER:$USER" "$WWW"
chmod 755 "$WWW"

# 4. logrotate for the cron log
cp "$REPO/deploy/fantasy.logrotate" /etc/logrotate.d/fantasy

# 5. POLLING crontab for the fantasy user (NOT root) — top of every hour, 13:00-
# 23:00 UTC (~9am-7pm ET), when PrizePicks posts + games run. The script is
# poll-safe (scrapes only until it captures today's board), so the card auto-
# updates within the hour the board goes live instead of at a fixed time.
echo "0 13-23 * * * $REPO/deploy/cron_daily.sh >> $REPO/logs/cron.log 2>&1" | crontab -u "$USER" -

echo
echo "installed. FOR AUTO-SCRAPE, add your (rotated) Firecrawl key:"
echo "  echo 'FIRECRAWL_API_KEY=fc-...' | sudo tee $REPO/.env"
echo "  sudo chown $USER:$USER $REPO/.env && sudo chmod 600 $REPO/.env"
echo "then verify with a manual run:"
echo "  sudo -u $USER $REPO/deploy/cron_daily.sh"
echo "  crontab -u $USER -l"
