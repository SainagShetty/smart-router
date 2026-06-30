#!/bin/bash
# smartrouter monitoring watchdog.
#
# Run on a schedule (launchd / cron, every ~5 min). Emails on the *transition*
# into a bad state (and again on recovery), so you don't get spammed while
# something stays down. Checks:
#   1. /health reachable and status:"ok"   (service down)
#   2. /health latency                       (degradation)
#   3. PM2 process online + restart-loop     (crash loop)
#   4. local Ollama tier reachable           (falls back to paid cloud if down)
#
# Config via env (sensible defaults):
#   SMARTROUTER_URL        (default http://127.0.0.1:4000)
#   SMARTROUTER_PM2_NAME   (default smart-router)
#   OLLAMA_URL             (default http://localhost:11434)
#   LATENCY_THRESHOLD_MS   (default 2000)
#   RESTART_JUMP           (default 3)
# Email delivery uses ~/.alert_credentials (iCloud SMTP).

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_DIR="$SCRIPT_DIR/.alert_state"
mkdir -p "$STATE_DIR"

URL="${SMARTROUTER_URL:-http://127.0.0.1:4000}"
PROC="${SMARTROUTER_PM2_NAME:-smart-router}"
OLLAMA="${OLLAMA_URL:-http://localhost:11434}"
LAT_THRESHOLD_MS="${LATENCY_THRESHOLD_MS:-2000}"
RESTART_JUMP="${RESTART_JUMP:-3}"

# shellcheck disable=SC1090
source ~/.alert_credentials 2>/dev/null

send_alert() {
  local subject="${1:-Alert}"; local body="${2:-No details.}"
  curl -s --url "smtp://${SMTP_HOST}:${SMTP_PORT}" --ssl-reqd \
    --mail-from "$ALERT_FROM" --mail-rcpt "$ALERT_TO" \
    --user "${SMTP_USER}:${SMTP_PASSWORD}" \
    -T <(printf "Subject: [ALERT] %s\nFrom: %s\nTo: %s\nContent-Type: text/plain\n\n%s\n\nTimestamp: %s\nHost: %s\n" \
      "$subject" "$ALERT_FROM" "$ALERT_TO" "$body" "$(date)" "$(hostname)")
}

fire() {  # fire <key> <subject> <body> : email only on entering bad state
  local key="$1" subject="$2" body="$3"
  if [ ! -f "$STATE_DIR/$key" ]; then
    send_alert "$subject" "$body"; date > "$STATE_DIR/$key"
  fi
}
recover() {  # recover <key> <subject> : email once if it had been bad
  local key="$1" subject="$2"
  if [ -f "$STATE_DIR/$key" ]; then
    send_alert "$subject" "Recovered at $(date)."; rm -f "$STATE_DIR/$key"
  fi
}
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1"; }

# --- 1 & 2. health endpoint + latency ---
start=$(date +%s%N)
resp=$(curl -s --max-time 8 -w "\n%{http_code}" "$URL/health" 2>/dev/null)
code=$(printf '%s' "$resp" | tail -1)
payload=$(printf '%s' "$resp" | sed '$d')
end=$(date +%s%N)
lat_ms=$(( (end - start) / 1000000 ))

if [ "$code" != "200" ] || ! printf '%s' "$payload" | grep -q '"status"'; then
  fire health_down "smartrouter DOWN" \
    "GET $URL/health returned code=$code. The shared model-router backend is not responding; dependent services will fail. Check: pm2 logs $PROC --err"
  log "health DOWN code=$code"
else
  recover health_down "smartrouter recovered"
  log "health ok ${lat_ms}ms"
  if [ "$lat_ms" -gt "$LAT_THRESHOLD_MS" ]; then
    fire latency_high "smartrouter slow" "GET /health took ${lat_ms}ms (> ${LAT_THRESHOLD_MS}ms threshold)."
  else
    recover latency_high "smartrouter latency normal"
  fi
fi

# --- 3. PM2 process status + restart-loop ---
pm2info=$(pm2 jlist 2>/dev/null)
if [ -n "$pm2info" ]; then
  read -r status restarts <<EOF
$(printf '%s' "$pm2info" | /usr/bin/python3 -c "import sys,json
d=[a for a in json.load(sys.stdin) if a['name']=='$PROC']
print((d[0]['pm2_env']['status'], d[0]['pm2_env']['restart_time']) if d else ('missing', -1))" 2>/dev/null | tr -d "(),'")
EOF
  if [ "$status" != "online" ]; then
    fire proc_down "smartrouter process $status" "PM2 process '$PROC' is '$status' (expected online)."
  else
    recover proc_down "smartrouter process online"
    last=$(cat "$STATE_DIR/last_restart_count" 2>/dev/null || echo "$restarts")
    if [ "$restarts" -ge 0 ] 2>/dev/null && [ "$last" -ge 0 ] 2>/dev/null; then
      if [ "$(( restarts - last ))" -ge "$RESTART_JUMP" ]; then
        send_alert "smartrouter restart loop" "PM2 '$PROC' restarted $(( restarts - last )) times since last check (total $restarts). Possible crash loop — check: pm2 logs $PROC --err"
      fi
    fi
    echo "$restarts" > "$STATE_DIR/last_restart_count"
  fi
  log "pm2 status=$status restarts=$restarts"
fi

# --- 4. local Ollama tier reachable ---
if curl -s --max-time 5 "$OLLAMA/api/tags" >/dev/null 2>&1; then
  recover ollama_down "smartrouter local tier (Ollama) recovered"
else
  fire ollama_down "smartrouter local tier DOWN" \
    "Ollama ($OLLAMA) is unreachable. The local model tier is unavailable; requests will fall back to paid cloud models (higher cost)."
  log "ollama DOWN"
fi
