#!/usr/bin/env bash
# Boots the rusty-gpt HTTP server in dev mode for the `server` compose
# service. The wrapper exists because `--load-latest-checkpoint` enforces
# strict shape/tokenizer compatibility against the running config — if the
# newest .mpk in checkpoints/ was trained with non-default hyperparameters,
# starting with `cargo run -- --serve --load-latest-checkpoint` panics with
# a metadata mismatch and the proxy returns 502.
#
# We sidestep that by reading the latest <checkpoint>.metadata.json sidecar
# and injecting matching --block-size/--embed-dim/--num-heads/--num-layers
# flags + RUSTY_GPT_BPE_TOKENIZER env so the server config follows the
# checkpoint rather than the other way around.
#
# Behaviour:
#   - Latest .metadata.json found  -> shape flags applied, tokenizer env set
#   - No metadata sidecar          -> fall back to defaults (legacy .mpk
#                                     files without a sidecar load unchecked)
#   - No checkpoint at all         -> still call cargo run; the binary itself
#                                     surfaces the missing-checkpoint error
#
# Called from compose.override.yaml; not intended to run on the host.

set -euo pipefail

: "${CHECKPOINT_DIR:=/workspace/checkpoints}"
: "${SERVER_ADDR:=0.0.0.0:8787}"

EXTRA_ARGS=()

latest_meta="$(ls -t "$CHECKPOINT_DIR"/*.metadata.json 2>/dev/null | head -1 || true)"

if [[ -n "$latest_meta" && -f "$latest_meta" ]]; then
    echo "[start_dev_server] auto-configuring from $latest_meta"

    block_size=$(jq -r '.model_shape.block_size // empty' "$latest_meta")
    embed_dim=$(jq  -r '.model_shape.embed_dim  // empty' "$latest_meta")
    num_heads=$(jq  -r '.model_shape.num_heads  // empty' "$latest_meta")
    num_layers=$(jq -r '.model_shape.num_layers // empty' "$latest_meta")
    tokenizer_path=$(jq -r '.tokenizer.path     // empty' "$latest_meta")

    [[ -n "$block_size"  ]] && EXTRA_ARGS+=(--block-size  "$block_size")
    [[ -n "$embed_dim"   ]] && EXTRA_ARGS+=(--embed-dim   "$embed_dim")
    [[ -n "$num_heads"   ]] && EXTRA_ARGS+=(--num-heads   "$num_heads")
    [[ -n "$num_layers"  ]] && EXTRA_ARGS+=(--num-layers  "$num_layers")

    # The metadata stores the tokenizer path as it was at training time —
    # which may be relative ("checkpoints/foo.json") or absolute. Resolve
    # against /workspace before exporting so the server can find it.
    if [[ -n "$tokenizer_path" ]]; then
        if [[ -f "$tokenizer_path" ]]; then
            export RUSTY_GPT_BPE_TOKENIZER="$tokenizer_path"
        elif [[ -f "/workspace/$tokenizer_path" ]]; then
            export RUSTY_GPT_BPE_TOKENIZER="/workspace/$tokenizer_path"
        else
            echo "[start_dev_server] WARNING: tokenizer '$tokenizer_path' from metadata not found; falling back to RUSTY_GPT_BPE_TOKENIZER=${RUSTY_GPT_BPE_TOKENIZER:-unset}" >&2
        fi
    fi

    echo "[start_dev_server] resolved: block=$block_size embed=$embed_dim heads=$num_heads layers=$num_layers tokenizer=${RUSTY_GPT_BPE_TOKENIZER:-default}"
else
    echo "[start_dev_server] no .metadata.json sidecar in $CHECKPOINT_DIR — using defaults"
fi

CARGO_CMD="cargo run --bin rusty-gpt -- --serve --load-latest-checkpoint --server-addr $SERVER_ADDR ${EXTRA_ARGS[*]}"
echo "[start_dev_server] launching: cargo watch on src/, Cargo.{toml,lock}"
echo "[start_dev_server]            command: $CARGO_CMD"

exec cargo watch \
    --why \
    --watch src \
    --watch Cargo.toml \
    --watch Cargo.lock \
    --shell "$CARGO_CMD"
