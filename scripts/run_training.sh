#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_training.sh [options] <training-data-file|hf-uri>

Runs rusty-gpt training against the provided UTF-8 text file or Hugging Face dataset URI.
Runs cargo in release mode by default; pass --cargo-profile dev for faster debug builds.

Options:
  --backend cpu|cuda                     Backend to train on (default: cpu)
  --model name                           trivial|single-attention|multi-attention|minigpt|compare (default: minigpt)
  --checkpoint path                      MiniGPT checkpoint path without extension (default: checkpoints/mini_gpt)
  --tokenizer path                       BPE tokenizer path (default: checkpoints/tokenizer.json)
  --train-tokenizer                      Train BPE tokenizer from the training input before model training
  --vocab-size n                         Vocab size for --train-tokenizer (default: 2048)
  --cargo-profile release|dev            Cargo build profile (default: release)
  --train-steps n                        Number of training steps (default: app default)
  --eval-interval n                      Steps between eval logs (default: 500 on cuda, app default otherwise)
  --prefetch-batches n                   CPU prefetch queue depth (default: 2 on cuda, app default otherwise)
  --log-format plain|json                Log format (default: app default, json on cuda)
  --benchmark                            Run MiniGpt generation benchmarks after training
  --benchmark-prompt-lens list           Comma-separated prompt lengths, e.g. 10,50,100
  --benchmark-gen-lens list              Comma-separated generation lengths, e.g. 50,100,200
  --benchmark-warmups n                  Warmup iterations per benchmark case
  --benchmark-iterations n               Measured iterations per benchmark case
  --artifacts-dir path                   Save run manifest and combined training/benchmark log
  -h, --help                             Show this help

Deprecated environment overrides (still honored, but emit a warning — use the flag instead):
  RUSTY_GPT_BACKEND                -> --backend
  RUSTY_GPT_MODEL                  -> --model
  RUSTY_GPT_MINIGPT_CHECKPOINT     -> --checkpoint
  RUSTY_GPT_BPE_TOKENIZER          -> --tokenizer
  RUSTY_GPT_TOKENIZER_VOCAB_SIZE   -> --vocab-size
  RUSTY_GPT_CARGO_PROFILE          -> --cargo-profile
  RUSTY_GPT_TRAIN_STEPS            -> --train-steps
  RUSTY_GPT_EVAL_INTERVAL          -> --eval-interval
  RUSTY_GPT_PREFETCH_BATCHES       -> --prefetch-batches
  RUSTY_GPT_LOG_FORMAT             -> --log-format
  RUSTY_GPT_BENCHMARK_PROMPT_LENS  -> --benchmark-prompt-lens
  RUSTY_GPT_BENCHMARK_GEN_LENS     -> --benchmark-gen-lens
  RUSTY_GPT_BENCHMARK_WARMUPS      -> --benchmark-warmups
  RUSTY_GPT_BENCHMARK_ITERATIONS   -> --benchmark-iterations
  RUSTY_GPT_RUN_ARTIFACT_DIR       -> --artifacts-dir
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_value() {
  if [[ "$2" -lt 2 ]]; then
    echo "$1 requires a value" >&2
    exit 2
  fi
}

BACKEND_ARG=""
MODEL_ARG=""
CHECKPOINT_ARG=""
TOKENIZER_ARG=""
TRAIN_TOKENIZER=0
VOCAB_SIZE_ARG=""
CARGO_PROFILE_ARG=""
TRAIN_STEPS_ARG=""
EVAL_INTERVAL_ARG=""
PREFETCH_BATCHES_ARG=""
LOG_FORMAT_ARG=""
BENCHMARK_ARGS=()
ARTIFACTS_DIR_ARG=""
TRAINING_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      require_value "$1" "$#"
      BACKEND_ARG="$2"
      shift 2
      ;;
    --backend=*)
      BACKEND_ARG="${1#--backend=}"
      shift
      ;;
    --model)
      require_value "$1" "$#"
      MODEL_ARG="$2"
      shift 2
      ;;
    --model=*)
      MODEL_ARG="${1#--model=}"
      shift
      ;;
    --checkpoint)
      require_value "$1" "$#"
      CHECKPOINT_ARG="$2"
      shift 2
      ;;
    --checkpoint=*)
      CHECKPOINT_ARG="${1#--checkpoint=}"
      shift
      ;;
    --tokenizer)
      require_value "$1" "$#"
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
      require_value "$1" "$#"
      VOCAB_SIZE_ARG="$2"
      shift 2
      ;;
    --vocab-size=*)
      VOCAB_SIZE_ARG="${1#--vocab-size=}"
      shift
      ;;
    --cargo-profile)
      require_value "$1" "$#"
      CARGO_PROFILE_ARG="$2"
      shift 2
      ;;
    --cargo-profile=*)
      CARGO_PROFILE_ARG="${1#--cargo-profile=}"
      shift
      ;;
    --train-steps)
      require_value "$1" "$#"
      TRAIN_STEPS_ARG="$2"
      shift 2
      ;;
    --train-steps=*)
      TRAIN_STEPS_ARG="${1#--train-steps=}"
      shift
      ;;
    --eval-interval)
      require_value "$1" "$#"
      EVAL_INTERVAL_ARG="$2"
      shift 2
      ;;
    --eval-interval=*)
      EVAL_INTERVAL_ARG="${1#--eval-interval=}"
      shift
      ;;
    --prefetch-batches)
      require_value "$1" "$#"
      PREFETCH_BATCHES_ARG="$2"
      shift 2
      ;;
    --prefetch-batches=*)
      PREFETCH_BATCHES_ARG="${1#--prefetch-batches=}"
      shift
      ;;
    --log-format)
      require_value "$1" "$#"
      LOG_FORMAT_ARG="$2"
      shift 2
      ;;
    --log-format=*)
      LOG_FORMAT_ARG="${1#--log-format=}"
      shift
      ;;
    --benchmark)
      BENCHMARK_ARGS+=(--benchmark-generation)
      shift
      ;;
    --benchmark-prompt-lens|--benchmark-gen-lens|--benchmark-warmups|--benchmark-iterations)
      require_value "$1" "$#"
      BENCHMARK_ARGS+=("$1" "$2")
      shift 2
      ;;
    --benchmark-prompt-lens=*|--benchmark-gen-lens=*|--benchmark-warmups=*|--benchmark-iterations=*)
      BENCHMARK_ARGS+=("${1%%=*}" "${1#*=}")
      shift
      ;;
    --artifacts-dir)
      require_value "$1" "$#"
      ARTIFACTS_DIR_ARG="$2"
      shift 2
      ;;
    --artifacts-dir=*)
      ARTIFACTS_DIR_ARG="${1#--artifacts-dir=}"
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

# Deprecated environment overrides are still honored (CLI flags win) but warn on use.
warn_deprecated_env() {
  if [[ -n "${!1:-}" ]]; then
    echo "warning: \$$1 is deprecated; use $2 instead" >&2
  fi
}
warn_deprecated_env RUSTY_GPT_BACKEND --backend
warn_deprecated_env RUSTY_GPT_MODEL --model
warn_deprecated_env RUSTY_GPT_MINIGPT_CHECKPOINT --checkpoint
warn_deprecated_env RUSTY_GPT_BPE_TOKENIZER --tokenizer
warn_deprecated_env RUSTY_GPT_TOKENIZER_VOCAB_SIZE --vocab-size
warn_deprecated_env RUSTY_GPT_CARGO_PROFILE --cargo-profile
warn_deprecated_env RUSTY_GPT_TRAIN_STEPS --train-steps
warn_deprecated_env RUSTY_GPT_EVAL_INTERVAL --eval-interval
warn_deprecated_env RUSTY_GPT_PREFETCH_BATCHES --prefetch-batches
warn_deprecated_env RUSTY_GPT_LOG_FORMAT --log-format
warn_deprecated_env RUSTY_GPT_BENCHMARK_PROMPT_LENS --benchmark-prompt-lens
warn_deprecated_env RUSTY_GPT_BENCHMARK_GEN_LENS --benchmark-gen-lens
warn_deprecated_env RUSTY_GPT_BENCHMARK_WARMUPS --benchmark-warmups
warn_deprecated_env RUSTY_GPT_BENCHMARK_ITERATIONS --benchmark-iterations
warn_deprecated_env RUSTY_GPT_RUN_ARTIFACT_DIR --artifacts-dir

RUSTY_GPT_BACKEND="${BACKEND_ARG:-${RUSTY_GPT_BACKEND:-cpu}}"
RUSTY_GPT_MODEL="${MODEL_ARG:-${RUSTY_GPT_MODEL:-minigpt}}"
RUSTY_GPT_MINIGPT_CHECKPOINT="${CHECKPOINT_ARG:-${RUSTY_GPT_MINIGPT_CHECKPOINT:-checkpoints/mini_gpt}}"
RUSTY_GPT_CARGO_PROFILE="${CARGO_PROFILE_ARG:-${RUSTY_GPT_CARGO_PROFILE:-release}}"
RUSTY_GPT_BPE_TOKENIZER="${TOKENIZER_ARG:-${RUSTY_GPT_BPE_TOKENIZER:-checkpoints/tokenizer.json}}"
RUSTY_GPT_RUN_ARTIFACT_DIR="${ARTIFACTS_DIR_ARG:-${RUSTY_GPT_RUN_ARTIFACT_DIR:-}}"
TOKENIZER_VOCAB_SIZE="${VOCAB_SIZE_ARG:-${RUSTY_GPT_TOKENIZER_VOCAB_SIZE:-2048}}"
TRAIN_STEPS="${TRAIN_STEPS_ARG:-${RUSTY_GPT_TRAIN_STEPS:-}}"
EVAL_INTERVAL="${EVAL_INTERVAL_ARG:-${RUSTY_GPT_EVAL_INTERVAL:-}}"
PREFETCH_BATCHES="${PREFETCH_BATCHES_ARG:-${RUSTY_GPT_PREFETCH_BATCHES:-}}"
LOG_FORMAT="${LOG_FORMAT_ARG:-${RUSTY_GPT_LOG_FORMAT:-}}"

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
  EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
  PREFETCH_BATCHES="${PREFETCH_BATCHES:-2}"
  LOG_FORMAT="${LOG_FORMAT:-json}"
elif [[ "$RUSTY_GPT_BACKEND" != "cpu" ]]; then
  echo "Unsupported backend '$RUSTY_GPT_BACKEND'; expected cpu or cuda." >&2
  exit 2
fi

LOG_FORMAT_ARGS=()
if [[ -n "$LOG_FORMAT" ]]; then
  LOG_FORMAT_ARGS=(--log-format "$LOG_FORMAT")
fi

TRAINING_TUNING_ARGS=()
if [[ -n "$TRAIN_STEPS" ]]; then
  TRAINING_TUNING_ARGS+=(--train-steps "$TRAIN_STEPS")
fi
if [[ -n "$EVAL_INTERVAL" ]]; then
  TRAINING_TUNING_ARGS+=(--eval-interval "$EVAL_INTERVAL")
fi
if [[ -n "$PREFETCH_BATCHES" ]]; then
  TRAINING_TUNING_ARGS+=(--prefetch-batches "$PREFETCH_BATCHES")
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
    echo "Unsupported cargo profile '$RUSTY_GPT_CARGO_PROFILE'; expected release or dev." >&2
    exit 2
    ;;
esac

cd "$ROOT_DIR"
mkdir -p "$(dirname "$RUSTY_GPT_MINIGPT_CHECKPOINT")"
mkdir -p "$(dirname "$RUSTY_GPT_BPE_TOKENIZER")"
export RUSTY_GPT_BPE_TOKENIZER

ARTIFACT_LOG=""
if [[ -n "$RUSTY_GPT_RUN_ARTIFACT_DIR" ]]; then
  mkdir -p "$RUSTY_GPT_RUN_ARTIFACT_DIR"
  RUSTY_GPT_RUN_ARTIFACT_DIR="$(cd "$RUSTY_GPT_RUN_ARTIFACT_DIR" && pwd)"
  ARTIFACT_LOG="$RUSTY_GPT_RUN_ARTIFACT_DIR/training.log"
  : > "$ARTIFACT_LOG"
  cat > "$RUSTY_GPT_RUN_ARTIFACT_DIR/manifest.txt" <<MANIFEST
created_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
backend=$RUSTY_GPT_BACKEND
model=$RUSTY_GPT_MODEL
input=$TRAINING_FILE
checkpoint=$RUSTY_GPT_MINIGPT_CHECKPOINT
tokenizer=$RUSTY_GPT_BPE_TOKENIZER
cargo_profile=$RUSTY_GPT_CARGO_PROFILE
train_tokenizer=$TRAIN_TOKENIZER
tokenizer_vocab_size=$TOKENIZER_VOCAB_SIZE
training_tuning_args=${TRAINING_TUNING_ARGS[*]-}
benchmark_args=${BENCHMARK_ARGS[*]-}
log_format_args=${LOG_FORMAT_ARGS[*]-}
MANIFEST
fi

run_logged() {
  local label="$1"
  shift

  if [[ -n "$ARTIFACT_LOG" ]]; then
    {
      printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label"
      printf '$'
      printf ' %q' "$@"
      printf '\n'
    } | tee -a "$ARTIFACT_LOG"
    "$@" 2>&1 | tee -a "$ARTIFACT_LOG"
  else
    "$@"
  fi
}

if [[ "$TRAIN_TOKENIZER" == "1" ]]; then
  run_logged "train tokenizer" cargo run "${CARGO_PROFILE_ARGS[@]}" --bin train-tokenizer -- \
    --corpus "$TRAINING_FILE" \
    --vocab-size "$TOKENIZER_VOCAB_SIZE" \
    --output "$RUSTY_GPT_BPE_TOKENIZER"
fi

run_logged "train and evaluate" cargo run "${CARGO_PROFILE_ARGS[@]}" "${CARGO_FEATURE_ARGS[@]}" --bin rusty-gpt -- \
  --input "$TRAINING_FILE" \
  --model "$RUSTY_GPT_MODEL" \
  --checkpoint "$RUSTY_GPT_MINIGPT_CHECKPOINT" \
  "${TRAINING_BACKEND_ARGS[@]}" \
  "${TRAINING_TUNING_ARGS[@]}" \
  "${LOG_FORMAT_ARGS[@]}" \
  "${BENCHMARK_ARGS[@]}"
