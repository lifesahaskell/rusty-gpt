#!/usr/bin/env bash
# End-to-end test for the dev `/api/generate` path. Catches the failure mode
# that produced the user-visible "Bad Gateway" in the UI:
#
#   1. compose.override.yaml's server command auto-loads the latest
#      checkpoint via `--load-latest-checkpoint`
#   2. If the running server's hyperparameters don't match the checkpoint's
#      metadata sidecar (block_size, embed_dim, num_heads, num_layers,
#      tokenizer sha256), strict validation panics and the server exits
#   3. The UI's /api proxy sees ECONNREFUSED and returns 502
#
# This test verifies:
#   - server boots cleanly against whatever .mpk is in ./checkpoints
#   - /api/info reports the same shape as the newest metadata sidecar
#     (proves scripts/start_dev_server.sh actually matched the checkpoint)
#   - /api/generate returns HTTP 200 with a "generated" field (direct)
#   - /api/generate returns HTTP 200 via the UI proxy (catches 502)
#
# Uses a stable compose project name so cargo target cache persists across
# runs — first invocation is slow (cold compile), subsequent are ~30s.
#
# Usage:  bash scripts/test_devcontainer_generate.sh
# Skip:   RUSTY_GPT_SKIP_DEVCONTAINER_TEST=1

set -euo pipefail

if [[ "${RUSTY_GPT_SKIP_DEVCONTAINER_TEST:-}" == "1" ]]; then
    echo ":: SKIPPED (RUSTY_GPT_SKIP_DEVCONTAINER_TEST=1)"
    exit 0
fi

cd "$(git rev-parse --show-toplevel)"

PROJECT="${RUSTY_GPT_GENTEST_PROJECT:-rusty-gpt-gentest}"
COMPOSE=(docker compose --project-name "$PROJECT" -f compose.yaml -f compose.override.yaml)
API_PORT="${RUSTY_GPT_GENTEST_API_PORT:-8887}"
UI_PORT="${RUSTY_GPT_GENTEST_UI_PORT:-5873}"
READY_TIMEOUT="${RUSTY_GPT_GENTEST_READY_TIMEOUT:-360}"
PROMPT_PAYLOAD='{"prompt":"fn ","max_tokens":4,"temperature":0.8}'

cleanup() {
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "::: FAIL — last 40 lines of server logs:" >&2
        "${COMPOSE[@]}" logs --tail=40 server >&2 || true
    fi
    "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true
    exit $rc
}
trap cleanup EXIT

# --- 0. Precondition: at least one checkpoint to load --------------------
if ! ls checkpoints/*.mpk >/dev/null 2>&1; then
    echo ":: SKIPPED — no .mpk in ./checkpoints/ (train one first via the trainer service)"
    exit 0
fi

echo ":: starting server + ui (project=$PROJECT, api=$API_PORT, ui=$UI_PORT)"
RUSTY_GPT_PORT="$API_PORT" RUSTY_GPT_UI_PORT="$UI_PORT" \
    "${COMPOSE[@]}" up -d --build server ui

# --- 1. Wait for server to be ready --------------------------------------
echo ":: waiting up to ${READY_TIMEOUT}s for /api/info to return 200"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
while true; do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$API_PORT/api/info" || echo 000)
    if [[ "$code" == "200" ]]; then
        echo ":: server READY"
        break
    fi
    state="$("${COMPOSE[@]}" ps --format '{{.State}}' server 2>/dev/null || true)"
    if [[ "$state" != "running" ]]; then
        echo "::: FAIL — server container state=$state" >&2
        exit 1
    fi
    if "${COMPOSE[@]}" logs --tail=10 server 2>&1 | grep -qE 'Error:|panicked at'; then
        echo "::: FAIL — server logs contain Error/panic" >&2
        exit 1
    fi
    if (( $(date +%s) > deadline )); then
        echo "::: FAIL — timed out after ${READY_TIMEOUT}s (last status=$code)" >&2
        exit 1
    fi
    sleep 5
done

# --- 2. /api/info reflects the metadata sidecar (auto-config worked) ----
echo ":: verifying /api/info shape matches the newest metadata sidecar"
info_json="$(curl -sS "http://127.0.0.1:$API_PORT/api/info")"
latest_meta="$(ls -t checkpoints/*.metadata.json 2>/dev/null | head -1 || true)"
if [[ -n "$latest_meta" ]]; then
    meta_json="$(cat "$latest_meta")"
    # docker compose exec to use the container's jq, but easier: pipe to python if jq missing locally
    if ! command -v jq >/dev/null 2>&1; then
        echo ":: (skipping shape comparison — jq not on host)"
    else
        # /api/info exposes: vocab_size, num_layers, num_heads, block_size,
        # tokenizer_vocab_size, model_tokenizer_vocab_match. embed_dim is
        # not exposed at the API layer (it's a model internal); we verify
        # the fields that ARE visible match the trained metadata.
        for field in vocab_size num_layers num_heads block_size; do
            info_val="$(jq -r ".${field} // empty"            <<<"$info_json")"
            meta_val="$(jq -r ".model_shape.${field} // empty" <<<"$meta_json")"
            if [[ -n "$meta_val" && "$info_val" != "$meta_val" ]]; then
                echo "::: FAIL — $field: server=$info_val metadata=$meta_val (auto-config drift)" >&2
                exit 1
            fi
        done
        echo ":: shape match — server reflects $latest_meta"
    fi
fi

# --- 3. /api/generate directly --------------------------------------------
echo ":: POST /api/generate (direct on $API_PORT)"
direct_resp="$(curl -sS -w '\n%{http_code}' -X POST \
    "http://127.0.0.1:$API_PORT/api/generate" \
    -H 'Content-Type: application/json' \
    -d "$PROMPT_PAYLOAD")"
direct_code="$(tail -n1 <<<"$direct_resp")"
direct_body="$(head -n -1 <<<"$direct_resp")"
if [[ "$direct_code" != "200" ]]; then
    echo "::: FAIL — direct /api/generate returned HTTP $direct_code" >&2
    echo "$direct_body" >&2
    exit 1
fi
if ! grep -q '"generated"' <<<"$direct_body"; then
    echo "::: FAIL — direct response missing 'generated' field" >&2
    echo "$direct_body" | head -c 500 >&2
    exit 1
fi
echo ":: direct OK"

# --- 4. /api/generate via the UI proxy (the original bug surface) -------
echo ":: POST /api/generate (via UI proxy on $UI_PORT) — original 502 surface"
# UI takes a few extra seconds after npm ci to bind; small retry loop.
for _ in 1 2 3 4 5; do
    proxy_resp="$(curl -sS -w '\n%{http_code}' -X POST \
        "http://127.0.0.1:$UI_PORT/api/generate" \
        -H 'Content-Type: application/json' \
        -d "$PROMPT_PAYLOAD" || true)"
    proxy_code="$(tail -n1 <<<"$proxy_resp")"
    [[ "$proxy_code" == "200" ]] && break
    sleep 3
done
proxy_body="$(head -n -1 <<<"$proxy_resp")"
if [[ "$proxy_code" != "200" ]]; then
    echo "::: FAIL — proxy /api/generate returned HTTP $proxy_code (the 502 bug)" >&2
    echo "$proxy_body" | head -c 500 >&2
    "${COMPOSE[@]}" logs --tail=20 ui >&2 || true
    exit 1
fi
if ! grep -q '"generated"' <<<"$proxy_body"; then
    echo "::: FAIL — proxy response missing 'generated' field" >&2
    echo "$proxy_body" | head -c 500 >&2
    exit 1
fi

echo
echo ":: PASS — /api/generate works direct AND via UI proxy"
