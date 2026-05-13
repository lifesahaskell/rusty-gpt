# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-binary Rust crate that trains and runs a GPT from scratch on top of the [Burn](https://burn.dev) 0.21 deep-learning framework. The Shakespeare-style corpus at `data/input.txt` is the default training input. Edition 2024. The smaller teaching variants (`trivial`, `single-attention`, `multi-attention`) use a char-level tokenizer; `MiniGpt` uses a BPE tokenizer loaded from `checkpoints/tokenizer.json`.

Published at https://github.com/lifesahaskell/rusty-gpt. See `README.md` for the user-facing quick start; this file is the architectural memo for Claude Code.

## Common commands

Run all commands from the crate root.

```bash
# Build / lint / format (CPU build; cuda is opt-in via --features cuda)
cargo build
cargo build --release
cargo clippy --all-targets          # has known pre-existing warnings, see Gotchas
cargo fmt --all -- --check
cargo check --features cuda         # verify the cuda gate still compiles

# Run the unit test suite + integration test (107 unit tests + 1 integration test)
cargo test

# Run a single unit test by path (filter is a substring match)
cargo test multi_head_attention_returns_model_dim_for_each_token_position

# Run the integration test only
cargo test --test default_runtime

# Default demo: CPU, trivial model, data/input.txt
cargo run

# Quick smoke run (mirrors the integration test)
RUSTY_GPT_TRAIN_STEPS=1 cargo run -- --input tests/fixtures/input.txt

# Train a BPE tokenizer (required before MiniGPT will run)
cargo run --bin train-tokenizer -- --corpus data/input.txt --vocab-size 2048 --output checkpoints/tokenizer.json

# Train MiniGPT on CUDA (opt-in feature; requires the CUDA toolkit)
cargo run --release --features cuda -- --backend cuda --model minigpt

# Compare all four model variants on the same batch
cargo run -- --model compare

# Load a saved MiniGPT checkpoint and chat with it
cargo run -- --model minigpt --interactive-generate --checkpoint checkpoints/mini_gpt

# Serve the GPT HTTP API on http://127.0.0.1:8787/api with the newest trained checkpoint
cargo run -- --serve --load-latest-checkpoint

# Collect another repo's source files into data/<name>.txt for use as a corpus
cargo run --bin collect-source -- --repo /path/to/repo
```

### Runtime configuration

CLI flags (in `src/main.rs`) and environment variables both work; CLI takes precedence.

| Flag | Env var | Values / default |
|---|---|---|
| `--backend` | `RUSTY_GPT_BACKEND` | `cpu` (default) \| `cuda` — `cuda` requires building with `--features cuda` |
| `--input` | `RUSTY_GPT_INPUT` | path, default `data/input.txt` |
| `--model` | `RUSTY_GPT_MODEL` | `trivial` (default) \| `single-attention` \| `multi-attention` \| `minigpt` (alias `mini-gpt`) \| `compare` |
| `--checkpoint` | `RUSTY_GPT_MINIGPT_CHECKPOINT` | path without `.mpk`, default `checkpoints/mini_gpt` |
| `--interactive-generate` | — | flag; requires `--backend cpu` and `--model minigpt` |
| `--serve` | — | flag; starts the Axum HTTP server instead of the demo/training run |
| `--load-checkpoint` | — | flag; load MiniGPT weights from `--checkpoint` before serving / interactive |
| `--load-latest-checkpoint` | — | flag; load the newest `*.mpk` in `checkpoints/` (mutually exclusive with `--load-checkpoint`) |
| `--server-addr` | `RUSTY_GPT_SERVER_ADDR` | `host:port`, default `127.0.0.1:8787` |
| `--benchmark-generation` | — | flag; runs naive-vs-cached generation benchmarks (requires `--model minigpt` or `compare`) |
| — | `RUSTY_GPT_BPE_TOKENIZER` | path, default `checkpoints/tokenizer.json` |
| — | `RUSTY_GPT_TRAIN_STEPS` | int |
| — | `RUSTY_GPT_EVAL_INTERVAL` | int (0 ⇒ log only final step) |
| — | `RUSTY_GPT_GENERATE_TOKENS` | int |
| — | `RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM` | f32, must be > 0 |
| — | `RUSTY_GPT_PREFETCH_BATCHES` | int, default `2` (CPU prefetch queue depth) |

The model-shape and training hyperparameters (`BLOCK_SIZE`, `BATCH_SIZE`, `EMBED_DIM`, `NUM_HEADS`, `NUM_LAYERS`, `DROPOUT`, `LEARNING_RATE`, `TRAIN_STEPS`, `EVAL_INTERVAL`, `GENERATE_TOKENS`, `MINIGPT_GRAD_CLIP_NORM`, `PREFETCH_BATCHES`) are compile-time constants at the top of `src/main.rs` (lines 34–46) — edit and rebuild to change them. The trailing five also accept env-var overrides at runtime.

## Architecture

```
src/
  lib.rs                        re-exports loader / model / server / tokenizer / utils for the bins + tests
  main.rs                       entry point: CLI/env parsing, demo / training / interactive / server / benchmark dispatch
  bin/
    train-tokenizer.rs          CLI to train a BPE tokenizer and write checkpoints/tokenizer.json
    collect-source.rs           CLI to concatenate a repo's source files into data/<name>.txt
  tokenizer/
    mod.rs                      `RuntimeTokenizer` enum dispatching to Char or Bpe; `Tokenizer` trait
    char.rs                     deterministic char-level tokenizer (sorted-unique chars ⇒ id)
    bpe.rs                      BPE tokenizer + `BpeTrainer`; loads/saves JSON
  loader/{mod,data}.rs          random-window batch sampler producing (x, y) shifted by 1; `BatchPrefetcher` for CPU prefetch
  model/mod.rs                  four model variants + shared loss/log helpers; `MiniGptConfig`, `TrainingLogFormat`, `TrainingLogContext`
  model/persistence.rs          save/load wrappers around burn's NamedMpkFileRecorder (.mpk files)
  server/mod.rs                 Axum router exposing POST /generate and GET /info (nested under /api by main.rs)
  utils/mod.rs                  generation benchmarking helpers (`benchmark_generation`, `benchmark_generation_cases`)
tests/
  default_runtime.rs            binary-level smoke test asserting CPU default does not load libcuda
  fixtures/input.txt            small corpus used by the integration test
  fixtures/tokenizer.json       BPE tokenizer fixture used by checkpoint-save unit tests
scripts/                        helper shell scripts: run_training.sh, run_local.sh, run_e2e_tests.sh, build_release_candidate.sh
mini-gpt-ui/                    separate React frontend that calls the /api server — see its own README
data/input.txt                  ~1 MB Shakespeare-style training corpus (committed)
data-secret/                    gitignored corpora directory (e.g. fafolang.txt, claude_src.txt)
checkpoints/                    gitignored .mpk model files + tokenizer.json (BPE artifact for MiniGPT)
.github/workflows/ci.yml        GitHub Actions: cargo fmt --check, cargo clippy, cargo test on push/PR
.gitignore                      excludes /target/, /checkpoints/, /data-secret/
```

### The model progression in `src/model/mod.rs`

The file builds up a GPT one layer of complexity at a time, and the four `ModelChoice` variants are deliberate teaching steps:

1. **`TrivialModel`** — embedding → linear head. No attention.
2. **`SingleAttentionModel`** — embedding → `SingleHeadAttention` (with causal mask) → linear head.
3. **`MultiAttentionModel`** — embedding → `MultiHeadAttention` (fused QKV, causal mask) → linear head.
4. **`MiniGpt`** — token + position embeddings → stack of pre-norm `Block`s (`LayerNorm → MultiHeadAttention → residual → LayerNorm → MLP → residual`) → final `LayerNorm` → linear LM head. Supports `generate()` / `generate_cached()` (greedy argmax, context cropped to `block_size`) and `forward_tokens_with_attention()` for attention introspection.

`ModelChoice::Compare` expands to the full list and trains/evals all four in sequence — `unreachable!` arms in the match dispatch enforce that expansion happens upstream of forward/training dispatch.

### Backend & autodiff plumbing

Every model is generic over `B: Backend`; training impls additionally require `B: AutodiffBackend`. `main.rs` picks the concrete backend at runtime:

- `BackendChoice::Cpu` ⇒ `NdArray<f32, i64>` for inference demo and HTTP serving, `Autodiff<NdArray<f32, i64>>` for training and interactive generation.
- `BackendChoice::Cuda` ⇒ `Cuda` backend, training only (interactive generation rejects CUDA). Only present when built with `--features cuda`.

`run_http_server` is generic over `B: Backend + Send + Sync + 'static` and is dispatched off `BackendChoice` the same way demo/training is, so the API works on either backend.

The `cuda` Cargo feature (`rusty-gpt`'s `cuda` ⇒ `burn/cuda`) is **off by default**. All cuda-touching code in `main.rs` — the `Cuda`/`CudaDevice` imports, the `BackendChoice::Cuda` enum variant, the cuda branch in `main`, the `"cuda"` arm in `parse_backend_name`, and the `backend_can_be_selected_from_args` test — is gated on `#[cfg(feature = "cuda")]`. A complementary `backend_cuda_arg_requires_feature` test under `#[cfg(not(feature = "cuda"))]` locks in the "rebuild with `--features cuda`" error message. Add new cuda references behind the same gate or CI (which builds CPU-only) will fail.

The integration test `default_runtime.rs` additionally asserts that a default-config run does **not** load `libcuda` — keep the CPU codepath free of CUDA backend instantiation even when the feature is enabled.

### Training loop pattern

Every `train(...)` impl follows the same shape (see `TrivialModel::train` for the canonical version):
- Build model + `AdamWConfig` optimizer + `CrossEntropyLoss`.
- For each step: `loader.next_batch` ⇒ forward ⇒ `language_model_loss` (reshape `[B*T, V]` against `[B*T]`) ⇒ `loss.backward()` ⇒ `GradientsParams::from_grads` ⇒ `optimizer.step(lr, model, grads)`.
- Log on `should_log_training_step(step, steps, eval_interval)`: training loss + a value-loss probe via `value_loss(...)` against the held-out 10% tail.
- `MiniGpt::train` additionally wires gradient clipping (`GradientClippingConfig::Norm(grad_clip_norm)`).

Train/value split: `split_training_and_value_tokens` reserves the **last 10%** for value loss and shrinks the value-loader's block size if the tail is shorter than `block_size`.

### Tokenizer & batching invariants

The tokenizer is chosen by model:

- `RuntimeTokenizer::Char(CharTokenizer)` — used by `Trivial`, `SingleAttention`, `MultiAttention`. Built from the corpus via `CharTokenizer::from_text` (sorted-unique chars ⇒ id, so the same corpus always produces the same vocab).
- `RuntimeTokenizer::Bpe(BpeTokenizer)` — used by `MiniGpt`. Loaded from `checkpoints/tokenizer.json` (or `RUSTY_GPT_BPE_TOKENIZER`) — **not derived from the corpus**. If the file is missing, `main.rs::load_minigpt_tokenizer` errors with the exact `cargo run --bin train-tokenizer ...` command to run.

Other invariants:
- `CharTokenizer::encode` panics on unknown chars; **use `try_encode` for any user-supplied input** (the interactive loop and the HTTP `/generate` endpoint do this). The BPE path is byte-based and never panics.
- `DataLoader::next_batch` samples `batch_size` random start positions in `0..tokens.len()-block_size`, returns `(x, y)` where `y` is `x` shifted right by one token — both shaped `[batch_size, block_size]` as `Int` tensors.

### MultiHeadAttention shape rule

`MultiHeadAttention::new(d_model, num_heads)` **panics** if `d_model % num_heads != 0` (also asserts `num_heads > 0`). The defaults (`EMBED_DIM=128`, `NUM_HEADS=4`, `HEAD_DIM=32`) satisfy this; preserve divisibility when changing constants.

### HTTP server module

`src/server/mod.rs` exposes an Axum `Router<SharedServerState<B>>` with `POST /generate` and `GET /info`. `main.rs::run_http_server` nests that router under `/api` (final routes: `/api/generate`, `/api/info`), binds to `--server-addr` (default `127.0.0.1:8787`), and serves with `axum::serve`.

`ServerState` holds a `MiniGpt`, a `RuntimeTokenizer`, and a `B::Device`. The model can come from one of three places at startup: a fresh template (default), `--load-checkpoint <path>`, or `--load-latest-checkpoint` (newest `*.mpk` in `checkpoints/`). The two `--load-*` flags are mutually exclusive — `main.rs` enforces this at parse time.

`GenerateResponse` includes per-layer/per-head attention matrices (`AttentionData { layer, head, weights }`) for visualization; the React UI in `mini-gpt-ui/` is the primary consumer.

### Sibling tooling

- `src/bin/train-tokenizer.rs` — `cargo run --bin train-tokenizer -- --corpus <path> --vocab-size <n> --output <path.json>`. Emits JSON progress events to stdout. Required to produce the BPE tokenizer MiniGPT depends on.
- `src/bin/collect-source.rs` — `cargo run --bin collect-source -- --repo <path> [--output <path>]`. Concatenates source files from a repo into `data/<name>.txt` for use as a training corpus.
- `scripts/run_training.sh`, `scripts/run_local.sh`, `scripts/run_e2e_tests.sh`, `scripts/build_release_candidate.sh` — convenience wrappers; see the README for details.
- `mini-gpt-ui/` — separate React frontend with its own README and toolchain. Calls the Rust server's `/api` routes. **Out of scope** for this memo; treat it as a black-box consumer.

## Gotchas

- **Checkpoints**: Burn's `NamedMpkFileRecorder` appends `.mpk` automatically. Pass the path **without** the extension (e.g. `checkpoints/mini_gpt`); the actual file is `checkpoints/mini_gpt.mpk`. `--checkpoint` and `RUSTY_GPT_MINIGPT_CHECKPOINT` follow the same convention. The `checkpoints/` directory is gitignored.
- **MiniGPT needs `checkpoints/tokenizer.json` to exist** — there is no auto-train fallback. If absent, `main.rs::load_minigpt_tokenizer` returns an error containing the exact `train-tokenizer` command to run. `RUSTY_GPT_BPE_TOKENIZER` overrides the path.
- **Server routes live under `/api`** — `src/server/mod.rs` defines `/generate` and `/info`, but `run_http_server` nests them under `/api`. `curl http://localhost:8787/generate` returns 404; the right path is `/api/generate`.
- **`--serve` only hosts MiniGPT** — the other three model variants are training-only and cannot be served. `--load-checkpoint` and `--load-latest-checkpoint` are mutually exclusive (parse-time error).
- **Checkpoint dir scan** — `--load-latest-checkpoint` reads from the hardcoded `checkpoints/` directory (`DEFAULT_CHECKPOINT_DIR` in `src/main.rs:31`), regardless of `--checkpoint`. The `--checkpoint` flag only matters for explicit save/load paths.
- **Interactive mode constraints**: `--interactive-generate` requires both `--backend cpu` and `--model minigpt`; any other combination errors out in `run_cpu_demo` / `main`.
- **CPU default must stay CUDA-free**: `tests/default_runtime.rs` greps stderr/stdout for `libcuda` and fails if it appears. Keep `BackendChoice::Cpu` away from CUDA types.
- **`compare` is a pseudo-variant**: it is expanded to the four real models via `ModelChoice::comparison_models()` before forward/training dispatch. The forward and training match statements on `ModelChoice::Compare` are `unreachable!()` and must stay that way.
- **Env-mutating tests use `unsafe`**: Rust 2024 makes `env::set_var` / `env::remove_var` unsafe. Existing tests in `main.rs` wrap them in `unsafe { ... }` blocks with a SAFETY comment — follow the same pattern when adding more.
- **`data-secret/` is gitignored**: anything you drop there (e.g. `fafolang.txt`, `claude_src.txt`) won't be committed. Use it for corpora you don't want in the repo.
- **Clippy has known pre-existing warnings**: `cargo clippy --all-targets` currently reports a handful of style lints in `src/model/mod.rs` (too-many-args, complex-type, etc.). The CI workflow runs clippy without `-D warnings` for that reason. If you tighten CI, fix those lints first.
