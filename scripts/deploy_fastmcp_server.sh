#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_FILE="$ROOT_DIR/lifesim/fastmcp_server.py"
RUN_DIR="$ROOT_DIR/.run"
LOG_DIR="$ROOT_DIR/logs"
PID_FILE="$RUN_DIR/fastmcp_server.pid"
LOG_FILE="$LOG_DIR/fastmcp_server.log"

# Defaults (can be overridden by env vars)
CONFIG_PATH="${LIFESIM_CONFIG_PATH:-$ROOT_DIR/config.yaml}"
TRANSPORT="${LIFESIM_TRANSPORT:-streamable-http}"
HOST="${LIFESIM_MCP_HOST:-0.0.0.0}"
PORT="${LIFESIM_MCP_PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$RUN_DIR" "$LOG_DIR"

usage() {
  cat <<EOF
Usage: $0 {start|stop|restart|status|logs}

Environment overrides:
  LIFESIM_CONFIG_PATH   Path to config yaml (default: $ROOT_DIR/config.yaml)
  LIFESIM_TRANSPORT     stdio|sse|streamable-http (default: streamable-http)
  LIFESIM_MCP_HOST      MCP host (default: 0.0.0.0)
  LIFESIM_MCP_PORT      MCP port (default: 8000)
  PYTHON_BIN            Python executable (default: python)
EOF
}

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

start_server() {
  if is_running; then
    echo "fastmcp_server is already running (PID: $(cat "$PID_FILE"))."
    return 0
  fi

  if [[ ! -f "$SERVER_FILE" ]]; then
    echo "Server file not found: $SERVER_FILE"
    exit 1
  fi

  if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH"
    exit 1
  fi

  echo "Starting fastmcp_server..."
  echo "  server:    $SERVER_FILE"
  echo "  config:    $CONFIG_PATH"
  echo "  transport: $TRANSPORT"
  echo "  host:      $HOST"
  echo "  port:      $PORT"
  echo "  log:       $LOG_FILE"

  nohup env \
    LIFESIM_CONFIG_PATH="$CONFIG_PATH" \
    LIFESIM_MCP_HOST="$HOST" \
    LIFESIM_MCP_PORT="$PORT" \
    "$PYTHON_BIN" "$SERVER_FILE" --config "$CONFIG_PATH" --transport "$TRANSPORT" \
    >>"$LOG_FILE" 2>&1 &

  echo $! > "$PID_FILE"
  sleep 1

  if is_running; then
    echo "Started successfully (PID: $(cat "$PID_FILE"))."
  else
    echo "Failed to start. Check logs: $LOG_FILE"
    exit 1
  fi
}

stop_server() {
  if ! is_running; then
    echo "fastmcp_server is not running."
    rm -f "$PID_FILE"
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  echo "Stopping fastmcp_server (PID: $pid)..."
  kill "$pid" 2>/dev/null || true

  for _ in {1..10}; do
    if kill -0 "$pid" 2>/dev/null; then
      sleep 1
    else
      rm -f "$PID_FILE"
      echo "Stopped."
      return 0
    fi
  done

  echo "Force killing PID: $pid"
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "Stopped (force)."
}

status_server() {
  if is_running; then
    echo "fastmcp_server is running (PID: $(cat "$PID_FILE"))."
    echo "Log file: $LOG_FILE"
  else
    echo "fastmcp_server is not running."
  fi
}

show_logs() {
  touch "$LOG_FILE"
  tail -n 100 -f "$LOG_FILE"
}

case "${1:-}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_server
    ;;
  status)
    status_server
    ;;
  logs)
    show_logs
    ;;
  *)
    usage
    exit 1
    ;;
esac

