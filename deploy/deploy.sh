#!/usr/bin/env bash
# Build the static Fantasy dashboard and publish it to fantasy.perfecthold.online.
#
#   ./deploy.sh [YYYY-MM-DD]     # defaults to today (America/New_York)
#
# Prereqs on the VPS (one-time): nginx server block installed (deploy/fantasy.nginx),
# /var/www/fantasy exists and is writable, TLS via certbot. SSH alias `kv8` set up.
set -euo pipefail

DATE="${1:-$(TZ=America/New_York date +%F)}"
HOST="kv8"
REMOTE="/var/www/fantasy"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/3] building dashboard for $DATE"
python "$ROOT/web/build_site.py" "$DATE" "$ROOT/web/dist/index.html"

echo "[2/3] syncing to $HOST:$REMOTE"
rsync -az --delete "$ROOT/web/dist/" "$HOST:$REMOTE/"

echo "[3/3] done -> https://fantasy.perfecthold.online"
