#!/usr/bin/env bash
# Smoke test for the `server` compose service under compose.override.yaml.
#
# Catches the class of bug we hit when `bash -lc` (login shell) was used to
# invoke cargo: /etc/profile resets PATH and strips /usr/local/cargo/bin,
# so the container restart-loops with `bash: line 1: cargo: command not found`.
#
# The test starts the actual server service from compose and asserts:
#   1. The container reaches the `running` state and stays there.
#   2. `cargo` and `cargo-watch` are invocable on PATH inside the container.
#   3. The compose command actually got past its shell to invoke cargo
#      (no "command not found" in logs).
#
# Does NOT wait for cargo to finish compiling — that takes 5+ min cold and
# isn't what this test is for. Image must already be built (build separately
# if needed: `docker compose build server`).
#
# Usage:  bash scripts/test_devcontainer_server.sh
# Skip:   RUSTY_GPT_SKIP_DEVCONTAINER_TEST=1

set -euo pipefail

if [[ "${RUSTY_GPT_SKIP_DEVCONTAINER_TEST:-}" == "1" ]]; then
    echo ":: SKIPPED (RUSTY_GPT_SKIP_DEVCONTAINER_TEST=1)"
    exit 0
fi

cd "$(git rev-parse --show-toplevel)"

PROJECT="rusty-gpt-srvtest-$$"
COMPOSE=(docker compose --project-name "$PROJECT" -f compose.yaml -f compose.override.yaml)
PORT="${RUSTY_GPT_SERVER_TEST_PORT:-8987}"
SETTLE_SECONDS="${RUSTY_GPT_SERVER_TEST_SETTLE:-12}"

cleanup() {
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "::: FAIL — last 30 lines of server logs:" >&2
        "${COMPOSE[@]}" logs --tail=30 server >&2 || true
        echo "::: container state:" >&2
        "${COMPOSE[@]}" ps server >&2 || true
    fi
    "${COMPOSE[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
    exit $rc
}
trap cleanup EXIT

echo ":: starting server service on port $PORT"
RUSTY_GPT_PORT="$PORT" "${COMPOSE[@]}" up -d --no-deps server

echo ":: waiting ${SETTLE_SECONDS}s for container to settle (cargo invocation happens in <1s)"
sleep "$SETTLE_SECONDS"

state="$("${COMPOSE[@]}" ps --format '{{.State}}' server 2>/dev/null || true)"
echo ":: container state: $state"
if [[ "$state" != "running" ]]; then
    echo "::: FAIL — expected state=running, got state=$state" >&2
    exit 1
fi

# Scan logs for known broken-startup patterns. Each catches a real bug we've
# hit in this stack — extend the list when a new one comes up.
#   "cargo: command not found"          PATH stripped by `bash -lc`
#   "could not determine which binary"  cargo run with no --bin in a multi-bin crate
#   "no bin target named"               --bin pointed at a binary that doesn't exist
log_snapshot="$("${COMPOSE[@]}" logs --tail=80 server 2>&1)"
bad_patterns=(
    'cargo: command not found'
    'cargo-watch: command not found'
    'could not determine which binary'
    'no bin target named'
)
for pat in "${bad_patterns[@]}"; do
    if grep -qF "$pat" <<<"$log_snapshot"; then
        echo "::: FAIL — server logs contain: $pat" >&2
        exit 1
    fi
done

echo ":: verifying cargo + cargo-watch are invocable inside the container"
"${COMPOSE[@]}" exec -T server bash -c 'command -v cargo && command -v cargo-watch'

echo ":: verifying cargo can resolve crates from the registry volume"
"${COMPOSE[@]}" exec -T server bash -c 'cargo --version && cargo-watch --version'

echo ":: PASS — server container is running, cargo + cargo-watch on PATH"
