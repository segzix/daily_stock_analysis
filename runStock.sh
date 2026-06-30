#!/bin/bash
# runStock — 启动 InStock 容器 + 运行完整股票分析
# 用法: ./runStock.sh [--stocks 600519] [--no-notify] [--dry-run] [...]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.in-stock.yml"

echo "=== A股智能分析系统 ==="

# runStock starts the local InStock stack, so prefer the local CYQ path for this
# invocation and avoid hitting paid/fragile remote chip APIs first.
RUNSTOCK_ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/runStock.env.XXXXXX")"
cleanup() {
    rm -f "$RUNSTOCK_ENV_FILE"
}
trap cleanup EXIT

if [ -f "$SCRIPT_DIR/.env" ]; then
    grep -v -E \
        '^[[:space:]]*#?[[:space:]]*(ENV_FILE|CHIP_DISTRIBUTION_SOURCE_PRIORITY)=' \
        "$SCRIPT_DIR/.env" > "$RUNSTOCK_ENV_FILE" || true
fi
{
    echo "ENV_FILE=$RUNSTOCK_ENV_FILE"
    echo "CHIP_DISTRIBUTION_SOURCE_PRIORITY=instock,akshare,tushare"
} >> "$RUNSTOCK_ENV_FILE"
export ENV_FILE="$RUNSTOCK_ENV_FILE"

# 1. 启动 InStock 容器（如果未运行）
if ! docker compose -f "$COMPOSE_FILE" ps --status running 2>/dev/null | grep -q InStock; then
    echo "[启动] InStock 容器..."
    docker compose -f "$COMPOSE_FILE" up -d --wait 2>/dev/null || true
fi
echo "[就绪] InStock: http://localhost:9988/"

# 2. 运行分析
conda run -n stock python "$SCRIPT_DIR/main.py" "$@"

# 3. 输出结果位置
echo ""
echo "[结果] 报告: reports/report_*.md"
echo "[结果] 筹码: reports/chip_distribution_latest.json"
