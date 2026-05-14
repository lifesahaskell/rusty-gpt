#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_training.sh [--backend cpu|cuda] [--checkpoint path] [--log-format plain|json] [--benchmark] [benchmark options] <training-data-file|hf-uri>

Runs rusty-gpt training against the provided UTF-8 text file or Hugging Face dataset URI.
Runs cargo in release mode by default. Set RUSTY_GPT_CARGO_PROFILE=dev for faster debug builds.

Options:
  --backend cpu|cuda                     Overrides RUSTY_GPT_BACKEND
  --checkpoint path                      Overrides RUSTY_GPT_MINIGPT_CHECKPOINT
  --tokenizer path                       Overrides RUSTY_GPT_BPE_TOKENIZER
  --train-tokenizer                      Train BPE tokenizer from the training input before model training
  --vocab-size n                         Vocab size for --train-tokenizer (default: 2048)
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
  RUSTY_GPT_CARGO_PROFILE=release|dev     Default: release
  RUSTY_GPT_TRAIN_STEPS=<int>             Default: app default
  RUSTY_GPT_EVAL_INTERVAL=<int>           Default: 500 on CUDA, app default otherwise
  RUSTY_GPT_PREFETCH_BATCHES=<int>         Default: 2 on CUDA, app default otherwise
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
TOKENIZER_ARG=""
TRAIN_TOKENIZER=0
TOKENIZER_VOCAB_SIZE="${RUSTY_GPT_TOKENIZER_VOCAB_SIZE:-2048}"
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
    --tokenizer)
      if [[ $# -lt 2 ]]; then
        echo "--tokenizer requires a path" >&2
        exit 2
      fi
      TOKENIZER_ARG="$2"
      shift 2
      ;;
    --tokenizer=*)
      TOKENIZER_ARG="${1#--tokenizer=}"
      shift
      ;;
    --train-tokenizer)
      TRAIN_TOKENIZER=1
      shift
      ;;
    --vocab-size)
      if [[ $# -lt 2 ]]; then
        echo "--vocab-size requires an integer" >&2
        exit 2
      fi
      TOKENIZER_VOCAB_SIZE="$2"
      shift 2
      ;;
    --vocab-size=*)
      TOKENIZER_VOCAB_SIZE="${1#--vocab-size=}"
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
RUSTY_GPT_CARGO_PROFILE="${RUSTY_GPT_CARGO_PROFILE:-release}"
RUSTY_GPT_BPE_TOKENIZER="${TOKENIZER_ARG:-${RUSTY_GPT_BPE_TOKENIZER:-checkpoints/tokenizer.json}}"

if [[ "$TRAINING_FILE" == hf://* ]]; then
  :
elif [[ ! -f "$TRAINING_FILE" ]]; then
  echo "Training data file not found: $TRAINING_FILE" >&2
  exit 1
else
  TRAINING_FILE="$(realpath "$TRAINING_FILE")"
fi

CARGO_FEATURE_ARGS=()
TRAINING_BACKEND_ARGS=(--backend "$RUSTY_GPT_BACKEND")
if [[ "$RUSTY_GPT_BACKEND" == "cuda" ]]; then
  CARGO_FEATURE_ARGS=(--features cuda)
  export RUSTY_GPT_EVAL_INTERVAL="${RUSTY_GPT_EVAL_INTERVAL:-500}"
  export RUSTY_GPT_PREFETCH_BATCHES="${RUSTY_GPT_PREFETCH_BATCHES:-2}"
  if [[ ${#LOG_FORMAT_ARGS[@]} -eq 0 && -z "${RUSTY_GPT_LOG_FORMAT:-}" ]]; then
    LOG_FORMAT_ARGS=(--log-format json)
  fi
elif [[ "$RUSTY_GPT_BACKEND" != "cpu" ]]; then
  echo "Unsupported RUSTY_GPT_BACKEND '$RUSTY_GPT_BACKEND'; expected cpu or cuda." >&2
  exit 2
fi

CARGO_PROFILE_ARGS=()
case "$RUSTY_GPT_CARGO_PROFILE" in
  release)
    CARGO_PROFILE_ARGS=(--release)
    ;;
  dev)
    CARGO_PROFILE_ARGS=()
    ;;
  *)
    echo "Unsupported RUSTY_GPT_CARGO_PROFILE '$RUSTY_GPT_CARGO_PROFILE'; expected release or dev." >&2
    exit 2
    ;;
esac

cd "$ROOT_DIR"
mkdir -p "$(dirname "$RUSTY_GPT_MINIGPT_CHECKPOINT")"
mkdir -p "$(dirname "$RUSTY_GPT_BPE_TOKENIZER")"
export RUSTY_GPT_BPE_TOKENIZER

if [[ "$TRAIN_TOKENIZER" == "1" ]]; then
  cargo run "${CARGO_PROFILE_ARGS[@]}" --bin train-tokenizer -- \
    --corpus "$TRAINING_FILE" \
    --vocab-size "$TOKENIZER_VOCAB_SIZE" \
    --output "$RUSTY_GPT_BPE_TOKENIZER"
fi

cargo run "${CARGO_PROFILE_ARGS[@]}" "${CARGO_FEATURE_ARGS[@]}" --bin rusty-gpt -- \
  --input "$TRAINING_FILE" \
  --model "$RUSTY_GPT_MODEL" \
  --checkpoint "$RUSTY_GPT_MINIGPT_CHECKPOINT" \
  "${TRAINING_BACKEND_ARGS[@]}" \
  "${LOG_FORMAT_ARGS[@]}" \
  "${BENCHMARK_ARGS[@]}"
