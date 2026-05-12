# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-binary Rust crate that trains and runs a character-level GPT from scratch on top of the [Burn](https://burn.dev) deep-learning framework. The Shakespeare-style corpus at `data/input.txt` is the default training input. Edition 2024.

## Common commands

Run all commands from the crate root (`/home/rakdos/repos/rusty-gpt`).

```bash
# Build / lint / format
cargo build
cargo build --release
cargo clippy --all-targets
cargo fmt

# Run the unit test suite + integration test
cargo test

# Run a single unit test by path (filter is a substring match)
cargo test multi_head_attention_returns_model_dim_for_each_token_position

# Run the integration test only
cargo test --test default_runtime

# Default demo: CPU, trivial model, data/input.txt
cargo run

# Quick smoke run (mirrors the integration test)
RUSTY_GPT_TRAIN_STEPS=1 cargo run -- --input tests/fixtures/input.txt

# Train MiniGPT on CUDA (opt-in feature; requires the CUDA toolkit)
cargo run --release --features cuda -- --backend cuda --model minigpt

# Compare all four model variants on the same batch
cargo run -- --model compare

# Load a saved MiniGPT checkpoint and chat with it
cargo run -- --model minigpt --interactive-generate --checkpoint checkpoints/mini_gpt
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
| — | `RUSTY_GPT_TRAIN_STEPS` | int |
| — | `RUSTY_GPT_EVAL_INTERVAL` | int (0 ⇒ log only final step) |
| — | `RUSTY_GPT_GENERATE_TOKENS` | int |
| — | `RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM` | f32, must be > 0 |

The model-shape hyperparameters (`BLOCK_SIZE`, `BATCH_SIZE`, `EMBED_DIM`, `NUM_HEADS`, `NUM_LAYERS`, `DROPOUT`, `LEARNING_RATE`) are compile-time constants at the top of `src/main.rs` — edit and rebuild to change them.

## Architecture

```
src/
  main.rs              entry point, CLI/env parsing, demo + training + interactive dispatch
  tokenizer/char.rs    deterministic char-level tokenizer (sorted-unique chars ⇒ id)
  loader/data.rs       random-window batch sampler producing (x, y) where y is x shifted by 1
  model/mod.rs         all four model variants + shared loss/logging helpers
  model/persistence.rs save/load wrappers around burn's NamedMpkFileRecorder (.mpk files)
  server/mod.rs        Axum HTTP router (POST /generate, GET /info) — defined but NOT wired up from main yet
tests/
  default_runtime.rs   binary-level smoke test asserting CPU default does not load libcuda
  fixtures/input.txt   small corpus used by the integration test
```

### The model progression in `src/model/mod.rs`

The file builds up a GPT one layer of complexity at a time, and the four `ModelChoice` variants are deliberate teaching steps:

1. **`TrivialModel`** — embedding → linear head. No attention.
2. **`SingleAttentionModel`** — embedding → `SingleHeadAttention` (with causal mask) → linear head.
3. **`MultiAttentionModel`** — embedding → `MultiHeadAttention` (fused QKV, causal mask) → linear head.
4. **`MiniGpt`** — token + position embeddings → stack of pre-norm `Block`s (`LayerNorm → MultiHeadAttention → residual → LayerNorm → MLP → residual`) → final `LayerNorm` → linear LM head. Supports `generate()` (greedy argmax, context cropped to `block_size`) and `forward_tokens_with_attention()` for attention introspection.

`ModelChoice::Compare` expands to the full list and trains/evals all four in sequence — `unreachable!` arms in the match dispatch enforce that expansion happens upstream of forward/training dispatch.

### Backend & autodiff plumbing

Every model is generic over `B: Backend`; training impls additionally require `B: AutodiffBackend`. `main.rs` picks the concrete backend at runtime:

- `BackendChoice::Cpu` ⇒ `NdArray<f32, i64>` for inference demo, `Autodiff<NdArray<f32, i64>>` for training and interactive generation.
- `BackendChoice::Cuda` ⇒ `Cuda` backend, training only (interactive generation rejects CUDA). Only present when built with `--features cuda`.

The `cuda` Cargo feature (`rusty-gpt`'s `cuda` ⇒ `burn/cuda`) is **off by default**. All cuda-touching code in `main.rs` — the `Cuda`/`CudaDevice` imports, the `BackendChoice::Cuda` enum variant, the cuda branch in `main`, the `"cuda"` arm in `parse_backend_name`, and the cuda-specific unit test — is gated on `#[cfg(feature = "cuda")]`. Add new cuda references behind the same gate or CI (which builds CPU-only) will fail.

The integration test `default_runtime.rs` additionally asserts that a default-config run does **not** load `libcuda` — keep the CPU codepath free of CUDA backend instantiation even when the feature is enabled.

### Training loop pattern

Every `train(...)` impl follows the same shape (see `TrivialModel::train` for the canonical version):
- Build model + `AdamWConfig` optimizer + `CrossEntropyLoss`.
- For each step: `loader.next_batch` ⇒ forward ⇒ `language_model_loss` (reshape `[B*T, V]` against `[B*T]`) ⇒ `loss.backward()` ⇒ `GradientsParams::from_grads` ⇒ `optimizer.step(lr, model, grads)`.
- Log on `should_log_training_step(step, steps, eval_interval)`: training loss + a value-loss probe via `value_loss(...)` against the held-out 10% tail.
- `MiniGpt::train` additionally wires gradient clipping (`GradientClippingConfig::Norm(grad_clip_norm)`).

Train/value split: `split_training_and_value_tokens` reserves the **last 10%** for value loss and shrinks the value-loader's block size if the tail is shorter than `block_size`.

### Tokenizer & batching invariants

- `CharTokenizer::from_text` sorts and dedups the input characters; ids are assigned in sorted order so the same corpus always produces the same vocab.
- `encode` panics on unknown chars; **use `try_encode` for any user-supplied input** (the interactive loop and the HTTP `/generate` endpoint do this).
- `DataLoader::next_batch` samples `batch_size` random start positions in `0..tokens.len()-block_size`, returns `(x, y)` where `y` is `x` shifted right by one token — both shaped `[batch_size, block_size]` as `Int` tensors.

### MultiHeadAttention shape rule

`MultiHeadAttention::new(d_model, num_heads)` **panics** if `d_model % num_heads != 0` (also asserts `num_heads > 0`). The defaults (`EMBED_DIM=128`, `NUM_HEADS=4`, `HEAD_DIM=32`) satisfy this; preserve divisibility when changing constants.

### HTTP server module

`src/server/mod.rs` exposes an Axum `Router<SharedServerState<B>>` with `POST /generate` and `GET /info`, returning generated text plus per-layer/per-head attention matrices for visualization. **It is implemented and unit-tested but `main.rs` does not start a server today** — wiring it up (binding a port, building `ServerState` from a loaded checkpoint) is the next integration step if you need an HTTP-served model.

## Gotchas

- **Checkpoints**: Burn's `NamedMpkFileRecorder` appends `.mpk` automatically. Pass the path **without** the extension (e.g. `checkpoints/mini_gpt`); the actual file is `checkpoints/mini_gpt.mpk`. `--checkpoint` and `RUSTY_GPT_MINIGPT_CHECKPOINT` follow the same convention.
- **Interactive mode constraints**: `--interactive-generate` requires both `--backend cpu` and `--model minigpt`; any other combination errors out in `run_cpu_demo` / `main`.
- **CPU default must stay CUDA-free**: `tests/default_runtime.rs` greps stderr/stdout for `libcuda` and fails if it appears. Keep `BackendChoice::Cpu` away from CUDA types.
- **`compare` is a pseudo-variant**: it is expanded to the four real models via `ModelChoice::comparison_models()` before forward/training dispatch. The forward and training match statements on `ModelChoice::Compare` are `unreachable!()` and must stay that way.
- **Env-mutating tests use `unsafe`**: Rust 2024 makes `env::set_var` / `env::remove_var` unsafe. Existing tests in `main.rs` wrap them in `unsafe { ... }` blocks with a SAFETY comment — follow the same pattern when adding more.
- **No README, no CI config**: there is currently no README.md, no `.github/`, and no top-level docs other than this file. Document new conventions here.
