#!/usr/bin/env bash
# Smoke test for the `ui` compose service under compose.override.yaml.
#
# Catches a class of bug where the ui service starts but never serves —
# specifically the empty-named-volume case: ui-node-modules is mounted at
# /app/node_modules so the directory exists from the first boot, which can
# fool a `[ -d node_modules ]` check into skipping `npm ci` and leaving the
# container in an `sh: vite: not found` restart loop.
#
# Usage:  scripts/test_devcontainer_ui.sh
# Exits non-zero on failure, prints offending logs.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PROJECT="rusty-gpt-uitest-$$"
COMPOSE=(docker compose --project-name "$PROJECT" -f compose.yaml -f compose.override.yaml)
PORT="${RUSTY_GPT_UI_TEST_PORT:-5273}"
TIMEOUT="${RUSTY_GPT_UI_TEST_TIMEOUT:-180}"

cleanup() {
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "::: FAIL — last 50 lines of ui logs:" >&2
    "${COMPOSE[@]}" logs --tail=50 ui >&2 || true
  fi
  "${COMPOSE[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
  exit $rc
}
trap cleanup EXIT

echo ":: starting ui service on port $PORT (fresh named volume)"
RUSTY_GPT_UI_PORT="$PORT" "${COMPOSE[@]}" up -d --no-deps --build ui

# Vite needs npm ci (~30s) + tsc + bundle. Poll until it answers or we time out.
echo ":: waiting up to ${TIMEOUT}s for vite to serve"
deadline=$(( $(date +%s) + TIMEOUT ))
while true; do
  status="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" || echo 000)"
  if [[ "$status" == "200" ]]; then
    echo ":: PASS — ui served HTTP $status"
    break
  fi

  # Fail fast if the container has crashed and won't recover.
  state="$("${COMPOSE[@]}" ps --format '{{.State}}' ui 2>/dev/null || true)"
  if [[ "$state" == "exited" ]]; then
    echo "::: ui container exited (state=$state)" >&2
    exit 1
  fi

  if (( $(date +%s) > deadline )); then
    echo "::: timed out after ${TIMEOUT}s (last status=$status, state=$state)" >&2
    exit 1
  fi
  sleep 2
done

# Specifically guard against the original bug: confirm npm actually populated
# the named volume with the vite binary.
echo ":: verifying vite binary exists inside the volume"
"${COMPOSE[@]}" exec -T ui test -x node_modules/.bin/vite
echo ":: PASS — node_modules/.bin/vite is present"
