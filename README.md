# rusty-gpt

A GPT playground built from scratch in Rust on top of the [Burn](https://burn.dev) deep-learning framework. The crate trains four progressively richer models — from a plain embedding-plus-linear baseline up to a multi-block transformer — and can serve MiniGPT interactively at the terminal or over HTTP for the React UI. MiniGPT uses the saved BPE tokenizer at `checkpoints/tokenizer.json` by default; the smaller baseline models still use the character tokenizer.

## Quick start

```bash
# Default: CPU, trivial model, data/input.txt
cargo run --bin rusty-gpt

# Train the full MiniGPT (CPU)
cargo run --release --bin rusty-gpt -- --model minigpt

# Train on CUDA (requires the CUDA toolkit installed; opt in via the `cuda` Cargo feature)
cargo run --release --features cuda --bin rusty-gpt -- --backend cuda --model minigpt

# Train from a file on CUDA via the helper script
./scripts/run_training.sh --backend cuda --checkpoint checkpoints/mini_gpt data/input.txt

# Train a BPE tokenizer from a corpus
cargo run --bin train-tokenizer -- --corpus data/repo-source.txt --vocab-size 2048 --output checkpoints/tokenizer.json

# Train the full MiniGPT against checkpoints/tokenizer.json
cargo run --release --bin rusty-gpt -- --model minigpt --input data/repo-source.txt

# Collect source files from another repository into data/<name>.txt
cargo run --bin collect-source -- --repo /path/to/repo --output repo-source.txt

# Compare all four model variants on the same batch
cargo run --bin rusty-gpt -- --model compare

# Chat with a saved MiniGPT checkpoint
cargo run --bin rusty-gpt -- --model minigpt --interactive-generate --checkpoint checkpoints/mini_gpt

# Serve the GPT HTTP API on http://127.0.0.1:8787/api
cargo run --bin rusty-gpt -- --serve --input data/input.txt

# Serve the API with the newest trained MiniGPT checkpoint from checkpoints/*.mpk
cargo run --bin rusty-gpt -- --serve --input data/input.txt --load-latest-checkpoint

# Serve the API with a specific pretrained MiniGPT checkpoint
cargo run --bin rusty-gpt -- --serve --input data/input.txt --checkpoint checkpoints/mini_gpt --load-checkpoint

# Serve the GPT HTTP API on CUDA
cargo run --features cuda --bin rusty-gpt -- --serve --backend cuda --input data/input.txt

# Start the local React UI dev server
./scripts/run_local.sh

# Start the local API + UI with a CUDA API backend
RUSTY_GPT_BACKEND=cuda ./scripts/run_local.sh
```

Run the test suite (unit tests plus the binary-level smoke test) with:

```bash
cargo test
```

Run the full-stack E2E suite, which starts the Rust API and Vite UI and sends a generation request through the UI server:

```bash
./scripts/run_e2e_tests.sh
```

Build a release-candidate artifact containing the release API binary, static UI bundle, launcher script, README, and manifest:

```bash
./scripts/build_release_candidate.sh

# Optional stable identifier for repeatable package names
RC_ID=rc1 ./scripts/build_release_candidate.sh

# Build a CUDA-capable artifact
RUSTY_GPT_BACKEND=cuda ./scripts/build_release_candidate.sh
```

Artifacts are written to `target/release-candidates/`.

## Configuration

CLI flags and `RUSTY_GPT_*` environment variables can both drive runtime behavior; the CLI wins when both are set.

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--backend cpu\|cuda` | `RUSTY_GPT_BACKEND` | `cpu` | `cuda` is only available when the crate is built with `--features cuda` (requires the CUDA toolkit). |
| `--input <path>` | `RUSTY_GPT_INPUT` | `data/input.txt` | Plain UTF-8 text. |
| `--model <name>` | `RUSTY_GPT_MODEL` | `trivial` | `trivial`, `single-attention`, `multi-attention`, `minigpt` (alias `mini-gpt`), `compare`. |
| `--checkpoint <path>` | `RUSTY_GPT_MINIGPT_CHECKPOINT` | `checkpoints/mini_gpt` | Path without `.mpk` — Burn appends it. |
| `--interactive-generate` | — | off | Requires `--backend cpu` and `--model minigpt`. |
| `--serve` | — | off | Starts the HTTP API under `/api`; supports `cpu` and compiled-in `cuda` backends. |
| `--load-checkpoint` | — | off | With `--serve`, loads MiniGPT API weights from `--checkpoint`. The checkpoint must match the model shape and tokenizer vocabulary from `--input`. |
| `--load-latest-checkpoint` | — | off | With `--serve`, loads the newest `.mpk` file in `checkpoints/` as the MiniGPT API weights. |
| `--server-addr <host:port>` | `RUSTY_GPT_SERVER_ADDR` | `127.0.0.1:8787` | Address used by `--serve`. |
| — | `RUSTY_GPT_TRAIN_STEPS` | `1000` | |
| — | `RUSTY_GPT_EVAL_INTERVAL` | `100` | `0` ⇒ log only the final step. |
| — | `RUSTY_GPT_GENERATE_TOKENS` | `80` | |
| — | `RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM` | `1.0` | Must be > 0. |

Model-shape hyperparameters (`BLOCK_SIZE`, `BATCH_SIZE`, `EMBED_DIM`, `NUM_HEADS`, `NUM_LAYERS`, `DROPOUT`, `LEARNING_RATE`) are compile-time constants at the top of `src/main.rs`.

MiniGPT and `compare` runs load the BPE tokenizer from `checkpoints/tokenizer.json`. Train or replace that file before training/loading MiniGPT checkpoints if the corpus vocabulary changes, because checkpoint tensor shapes must match the tokenizer vocabulary size.

## Tools

The package includes small utility binaries under `src/bin/`.

### BPE tokenizer training

Train and save a byte-pair encoding tokenizer as JSON:

```bash
cargo run --bin train-tokenizer -- --corpus data/fafolang.txt --vocab-size 2048 --output checkpoints/tokenizer.json
```

Flags:

| Flag | Required | Notes |
|---|---:|---|
| `--corpus <path>` | yes | UTF-8 text corpus used to learn BPE merges. |
| `--vocab-size <n>` | yes | Target vocabulary size. Must be at least `256` because byte tokens are always present. |
| `--output <path>` | yes | Tokenizer JSON output path. Parent directories are created. |

### Source corpus collection

Concatenate source files from a repository into a text file under `data/`:

```bash
cargo run --bin collect-source -- --repo /path/to/repo --output repo-source.txt
```

If `--output` is omitted, the tool writes `data/<repo-folder-name>.txt`. If `--output` includes a path, only the file name is used so output stays in `data/`.

The collector includes common source/config/documentation extensions such as `rs`, `toml`, `ts`, `tsx`, `js`, `py`, `go`, `java`, `c`, `cpp`, `sh`, `html`, `css`, `json`, `yaml`, `sql`, and `md`. It skips generated or heavy directories including `.git`, `target`, `node_modules`, `dist`, `build`, `.next`, and `coverage`.

## CUDA troubleshooting

The CUDA backend is compiled into the binary with `--features cuda`, but it still needs a working NVIDIA driver visible to the process. If `/api/info` works and `/api/generate` crashes with a `cudarc` error like `undefined symbol: cuCoredumpDeregisterCompleteCallback`, the loaded `libcuda.so.1` is older than the CUDA driver API expected by Burn/CubeCL. On WSL this usually means the Windows NVIDIA driver or WSL GPU integration needs to be updated; `nvidia-smi` should work from the same shell before using `--backend cuda`.

## What's in the box

- `src/lib.rs` — shared library entry point used by the main app and utility binaries.
- `src/tokenizer/char.rs` — deterministic character-level tokenizer.
- `src/tokenizer/bpe.rs` — byte-pair encoding tokenizer trainer, JSON save/load, and shared tokenizer trait implementation.
- `src/loader/data.rs` — random-window batch sampler returning `(x, y)` where `y` is `x` shifted by one token.
- `src/model/mod.rs` — four models built up step by step:
  1. `TrivialModel` — embedding → linear head.
  2. `SingleAttentionModel` — single causal-masked attention head.
  3. `MultiAttentionModel` — fused-QKV multi-head attention.
  4. `MiniGpt` — token + position embeddings, a stack of pre-norm transformer blocks, final layer norm, LM head; supports greedy generation and attention introspection.
- `src/model/persistence.rs` — `.mpk` checkpoint save/load via Burn's `NamedMpkFileRecorder`.
- `src/server/mod.rs` — Axum router exposing `POST /api/generate` and `GET /api/info` with per-layer/per-head attention matrices.
- `src/bin/train-tokenizer.rs` — CLI for training and saving BPE tokenizers.
- `src/bin/collect-source.rs` — CLI for building code corpora from repositories.
- `mini-gpt-ui/` — React/Vite UI that calls the GPT API.
- `scripts/run_local.sh` — starts the local UI dev server.
- `scripts/run_e2e_tests.sh` — runs the full-stack E2E suite.
- `scripts/build_release_candidate.sh` — packages a deployable release-candidate tarball.
- `tests/default_runtime.rs` — runs the binary and asserts the CPU default path never loads `libcuda`.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
