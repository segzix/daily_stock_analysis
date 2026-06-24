#!/bin/bash
# Auto-start SearXNG search proxy for daily_stock_analysis

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)/.."
SETTINGS="$PROJECT_ROOT/config/searxng_settings.yml"
LOG="$PROJECT_ROOT/logs/searxng.log"
PID_FILE="$PROJECT_ROOT/logs/searxng.pid"

# Load .env
if [ -f "$PROJECT_ROOT/.env" ]; then
  export $(grep -v '^#' "$PROJECT_ROOT/.env" | grep -v '^$' | xargs) 2>/dev/null
fi

# Kill existing instance and wait for it to release the port
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    kill "$OLD_PID" 2>/dev/null
    for _ in $(seq 1 10); do
        if ! kill -0 "$OLD_PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 "$OLD_PID" 2>/dev/null; then
        kill -9 "$OLD_PID" 2>/dev/null
        sleep 2
    fi
fi
# Ensure port 8888 is free
while ss -tlnp 2>/dev/null | grep -q ':8888 '; do
    sleep 1
done

# Clear stale engine suspension cache to prevent persistent CAPTCHA/timeout state
rm -f /tmp/sxng_cache_*.db 2>/dev/null

# Generate config from .env proxy settings
PROXY_HOST="${PROXY_HOST:-}"
PROXY_PORT="${PROXY_PORT:-}"
PROXY_BLOCK=""
if [ "$USE_PROXY" = "true" ] && [ -n "$PROXY_HOST" ] && [ -n "$PROXY_PORT" ]; then
  PROXY_BLOCK="
  proxies:
    all://:
      - http://${PROXY_HOST}:${PROXY_PORT}"
fi

cat > "$SETTINGS" << EOF
use_default_settings: true
general:
  debug: false
  instance_name: "Stock-Search"
search:
  safe_search: 0
  formats:
    - html
    - json
engines:
  - name: brave
    disabled: true
server:
  port: 8888
  bind_address: "127.0.0.1"
  secret_key: "local-stock-analysis-proxy"
  limiter: false
  image_proxy: false
  public_instance: false
valkey:
  url: false
outgoing:
  request_timeout: 10.0
  max_request_timeout: 20.0
  enable_http2: false${PROXY_BLOCK}
EOF

# Start SearXNG
SEARXNG_SETTINGS_PATH="$SETTINGS" \
  /home/segzix/miniconda3/envs/stock/bin/python -m searx.webapp \
  >> "$LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "SearXNG started (PID: $!) — http://127.0.0.1:8888"
