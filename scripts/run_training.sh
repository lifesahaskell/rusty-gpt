#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_training.sh [--backend cpu|cuda] [--checkpoint path] [--log-format plain|json] [--benchmark] [benchmark options] <training-data-file>

Runs rusty-gpt training against the provided UTF-8 text file.

Options:
  --backend cpu|cuda                     Overrides RUSTY_GPT_BACKEND
  --checkpoint path                      Overrides RUSTY_GPT_MINIGPT_CHECKPOINT
  --log-format plain|json                Overrides RUSTY_GPT_LOG_FORMAT
  --benchmark                            Run MiniGpt generation benchmarks after training
  --benchmark-prompt-lens list           Comma-separated prompt lengths, e.g. 10,50,100
  --benchmark-gen-lens list              Comma-separated generation lengths, e.g. 50,100,200
  --benchmark-warmups n                  Warmup iterations per benchmark case
  --benchmark-iterations n               Measured iterations per benchmark case
  -h, --help                             Show this help

Environment overrides:
  RUSTY_GPT_BACKEND=cpu|cuda              Default: cpu
  RUSTY_GPT_LOG_FORMAT=plain|json         Default: app default
  RUSTY_GPT_MODEL=trivial|single-attention|multi-attention|minigpt|compare
                                         Default: minigpt
  RUSTY_GPT_MINIGPT_CHECKPOINT=<path>     Default: checkpoints/mini_gpt
  RUSTY_GPT_TRAIN_STEPS=<int>             Default: app default
  RUSTY_GPT_EVAL_INTERVAL=<int>           Default: app default
  RUSTY_GPT_PREFETCH_BATCHES=<int>         Default: app default
  RUSTY_GPT_BENCHMARK_PROMPT_LENS=list    Default: app default
  RUSTY_GPT_BENCHMARK_GEN_LENS=list       Default: app default
  RUSTY_GPT_BENCHMARK_WARMUPS=<int>       Default: app default
  RUSTY_GPT_BENCHMARK_ITERATIONS=<int>    Default: app default
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BACKEND_ARG=""
CHECKPOINT_ARG=""
LOG_FORMAT_ARGS=()
BENCHMARK_ARGS=()
TRAINING_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      if [[ $# -lt 2 ]]; then
        echo "--backend requires a value: cpu or cuda" >&2
        exit 2
      fi
      BACKEND_ARG="$2"
      shift 2
      ;;
    --backend=*)
      BACKEND_ARG="${1#--backend=}"
      shift
      ;;
    --checkpoint)
      if [[ $# -lt 2 ]]; then
        echo "--checkpoint requires a value: path to a saved .mpk checkpoint without the extension" >&2
        exit 2
      fi
      CHECKPOINT_ARG="$2"
      shift 2
      ;;
    --checkpoint=*)
      CHECKPOINT_ARG="${1#--checkpoint=}"
      shift
      ;;
    --log-format)
      if [[ $# -lt 2 ]]; then
        echo "--log-format requires a value: plain or json" >&2
        exit 2
      fi
      LOG_FORMAT_ARGS=(--log-format "$2")
      shift 2
      ;;
    --log-format=*)
      LOG_FORMAT_ARGS=(--log-format "${1#--log-format=}")
      shift
      ;;
    --benchmark)
      BENCHMARK_ARGS+=(--benchmark-generation)
      shift
      ;;
    --benchmark-prompt-lens|--benchmark-gen-lens|--benchmark-warmups|--benchmark-iterations)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a value" >&2
        exit 2
      fi
      BENCHMARK_ARGS+=("$1" "$2")
      shift 2
      ;;
    --benchmark-prompt-lens=*|--benchmark-gen-lens=*|--benchmark-warmups=*|--benchmark-iterations=*)
      BENCHMARK_ARGS+=("${1%%=*}" "${1#*=}")
      shift
      ;;
    -*)
      echo "Unsupported argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$TRAINING_FILE" ]]; then
        echo "Only one training data file may be provided." >&2
        usage >&2
        exit 2
      fi
      TRAINING_FILE="$1"
      shift
      ;;
  esac
done

if [[ -z "$TRAINING_FILE" ]]; then
  usage >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUSTY_GPT_BACKEND="${BACKEND_ARG:-${RUSTY_GPT_BACKEND:-cpu}}"
RUSTY_GPT_MODEL="${RUSTY_GPT_MODEL:-minigpt}"
RUSTY_GPT_MINIGPT_CHECKPOINT="${CHECKPOINT_ARG:-${RUSTY_GPT_MINIGPT_CHECKPOINT:-checkpoints/mini_gpt}}"

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

cargo run "${CARGO_FEATURE_ARGS[@]}" --bin rusty-gpt -- \
  --input "$TRAINING_FILE" \
  --model "$RUSTY_GPT_MODEL" \
  --checkpoint "$RUSTY_GPT_MINIGPT_CHECKPOINT" \
  "${TRAINING_BACKEND_ARGS[@]}" \
  "${LOG_FORMAT_ARGS[@]}" \
  "${BENCHMARK_ARGS[@]}"
