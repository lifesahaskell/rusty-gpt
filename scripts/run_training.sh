#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_training.sh <training-data-file>

Runs rusty-gpt training against the provided UTF-8 text file.

Environment overrides:
  RUSTY_GPT_BACKEND=cpu|cuda              Default: cpu
  RUSTY_GPT_MODEL=trivial|single-attention|multi-attention|minigpt|compare
                                         Default: minigpt
  RUSTY_GPT_MINIGPT_CHECKPOINT=<path>     Default: checkpoints/mini_gpt
  RUSTY_GPT_TRAIN_STEPS=<int>             Default: app default
  RUSTY_GPT_EVAL_INTERVAL=<int>           Default: app default
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_FILE="$1"
RUSTY_GPT_BACKEND="${RUSTY_GPT_BACKEND:-cpu}"
RUSTY_GPT_MODEL="${RUSTY_GPT_MODEL:-minigpt}"
RUSTY_GPT_MINIGPT_CHECKPOINT="${RUSTY_GPT_MINIGPT_CHECKPOINT:-checkpoints/mini_gpt}"

if [[ ! -f "$TRAINING_FILE" ]]; then
  echo "Training data file not found: $TRAINING_FILE" >&2
  exit 1
fi
TRAINING_FILE="$(realpath "$TRAINING_FILE")"

CARGO_FEATURE_ARGS=()
TRAINING_BACKEND_ARGS=(--backend "$RUSTY_GPT_BACKEND")
if [[ "$RUSTY_GPT_BACKEND" == "cuda" ]]; then
  CARGO_FEATURE_ARGS=(--features cuda)
elif [[ "$RUSTY_GPT_BACKEND" != "cpu" ]]; then
  echo "Unsupported RUSTY_GPT_BACKEND '$RUSTY_GPT_BACKEND'; expected cpu or cuda." >&2
  exit 2
fi

cd "$ROOT_DIR"
mkdir -p "$(dirname "$RUSTY_GPT_MINIGPT_CHECKPOINT")"

cargo run "${CARGO_FEATURE_ARGS[@]}" -- \
  --input "$TRAINING_FILE" \
  --model "$RUSTY_GPT_MODEL" \
  --checkpoint "$RUSTY_GPT_MINIGPT_CHECKPOINT" \
  "${TRAINING_BACKEND_ARGS[@]}"
