#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_PORT="${UI_PORT:-5173}"
API_ADDR="${API_ADDR:-127.0.0.1:8787}"
RUSTY_GPT_INPUT="${RUSTY_GPT_INPUT:-data/input.txt}"
RUSTY_GPT_BACKEND="${RUSTY_GPT_BACKEND:-cpu}"

CARGO_FEATURE_ARGS=()
API_BACKEND_ARGS=()
if [[ "$RUSTY_GPT_BACKEND" == "cuda" ]]; then
  CARGO_FEATURE_ARGS=(--features cuda)
  API_BACKEND_ARGS=(--backend cuda)
fi

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
cargo run "${CARGO_FEATURE_ARGS[@]}" -- --serve --input "$RUSTY_GPT_INPUT" --server-addr "$API_ADDR" "${API_BACKEND_ARGS[@]}" &
API_PID="$!"

cd "$ROOT_DIR/mini-gpt-ui"
VITE_API_PROXY_TARGET="http://$API_ADDR" npm run dev -- --host 127.0.0.1 --port "$UI_PORT"
