# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-binary Rust crate that trains and runs a GPT from scratch on top of the [Burn](https://burn.dev) 0.21 deep-learning framework. The Shakespeare-style corpus at `data/input.txt` is the default training input. Edition 2024. The smaller teaching variants (`trivial`, `single-attention`, `multi-attention`) use a char-level tokenizer; `MiniGpt` and `MoeGpt` use a BPE tokenizer loaded from `checkpoints/tokenizer.json`.

Published at https://github.com/lifesahaskell/rusty-gpt. See `README.md` for the user-facing quick start; this file is the architectural memo for Claude Code.

## Common commands

Run all commands from the crate root.

```bash
# Build / lint / format (CPU build; cuda is opt-in via --features cuda)
cargo build
cargo build --release
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features cuda -- -D warnings
cargo fmt --all -- --check
cargo check --features cuda         # verify the cuda gate still compiles

# Run the full test suite (lib + main + two bin targets + three integration suites)
cargo test

# Run a single unit test by path (filter is a substring match)
cargo test multi_head_attention_returns_model_dim_for_each_token_position

# Run a specific integration test target only
cargo test --test default_runtime
cargo test --test graceful_shutdown      # SIGINT-during-training, Unix-only (#![cfg(unix)])
cargo test --test periodic_checkpoints   # --checkpoint-interval / --checkpoint-keep behavior

# Default demo: CPU, MiniGPT model, data/input.txt (needs checkpoints/tokenizer.json)
cargo run

# Quick smoke run (mirrors the integration test)
RUSTY_GPT_TRAIN_STEPS=1 cargo run -- --input tests/fixtures/input.txt

# Train a BPE tokenizer (required before MiniGPT will run; --corpus also accepts hf://...)
cargo run --bin train-tokenizer -- --corpus data/input.txt --vocab-size 2048 --output checkpoints/tokenizer.json

# Train MiniGPT on CUDA (opt-in feature; requires the CUDA toolkit)
cargo run --release --features cuda -- --backend cuda --model minigpt

# Train from a HuggingFace dataset URI (loader fetches + caches under data/huggingface-cache/)
cargo run -- --input "hf://Salesforce/wikitext?config=wikitext-2-raw-v1&split=train&rows=1000"

# Compare all five model variants on the same batch
cargo run -- --model compare

# Load a saved MiniGPT or MoeGPT checkpoint and chat with it
cargo run -- --model minigpt --interactive-generate --checkpoint checkpoints/mini_gpt

# Serve the GPT HTTP API on http://127.0.0.1:8787/api with the newest trained checkpoint
cargo run -- --serve --load-latest-checkpoint

# Emit structured observability events as JSON instead of the default plain text
cargo run -- --log-format json

# Collect another repo's source files into data/<name>.txt for use as a corpus
cargo run --bin collect-source -- --repo /path/to/repo
```

### Runtime configuration

Defaults live in `src/runtime_config.rs`; validation rules in `Hyperparameters::validate`. CLI flags and `RUSTY_GPT_*` env vars both work; CLI wins.

**Full flag/env reference: `docs/configuration.md`** — every flag, env var, default, and constraint. That file is authoritative; do not mirror it here.

`Hyperparameters::from_env_and_overrides` (in `src/runtime_config.rs`) resolves env → CLI overrides → `validate()`, which enforces the positivity/divisibility rules and recomputes `head_dim = embed_dim / num_heads`. Invalid combinations (e.g. `embed_dim` not divisible by `num_heads`) fail at config-parse time.

## Architecture

What the file tree won't tell you:

- `main.rs` is a thin entry point. The real dispatch is split across `runtime_config.rs` (parse + validate) → `runtime_assets.rs` (corpus / tokenizer / checkpoint resolution, including `hf://`) → `runtime_orchestration.rs` (demo / training / interactive / server / benchmark) → `runtime_training.rs`. Start there, not in `main.rs`.
- All five model variants live in `src/model/mod.rs`; the per-variant `train(...)` impls were lifted out into `model/training.rs`.
- `mini-gpt-ui/` is a separate React app with its own toolchain — **out of scope** for this memo; treat it as a black-box consumer of `/api`.

### The model progression in `src/model/mod.rs`

The file builds up a GPT one layer of complexity at a time, and the five trainable `ModelChoice` variants are deliberate teaching steps:

1. **`TrivialModel`** — embedding → linear head. No attention.
2. **`SingleAttentionModel`** — embedding → `SingleHeadAttention` (with causal mask) → linear head.
3. **`MultiAttentionModel`** — embedding → `MultiHeadAttention` (fused QKV, causal mask) → linear head.
4. **`MiniGpt`** — token + position embeddings → stack of pre-norm `Block`s (`LayerNorm → MultiHeadAttention → residual → LayerNorm → dense MLP → residual`) → final `LayerNorm` → linear LM head. Supports `generate()` / `generate_cached()` (greedy argmax, context cropped to `block_size`) and `forward_tokens_with_attention()` for attention introspection.
5. **`MoeGpt`** — MiniGPT shape with each block's feed-forward slot replaced by `MoeFeedForward` (`Router` + expert MLP pool). Training adds `moe_aux_loss_weight * load_balancing_loss` to the language-model loss; HTTP generation returns optional `routing[]` data for the UI.

`ModelChoice::Compare` expands to the full list and trains/evals all five in sequence — `unreachable!` arms in the match dispatch enforce that expansion happens upstream of forward/training dispatch.

### Backend & autodiff plumbing

Every model is generic over `B: Backend`; training impls additionally require `B: AutodiffBackend`. The runtime modules pick the concrete backend off `BackendChoice`:

- `BackendChoice::Cpu` ⇒ `NdArray<f32, i64>` for inference demo and HTTP serving, `Autodiff<NdArray<f32, i64>>` for training and interactive generation.
- `BackendChoice::Cuda` ⇒ `Cuda` backend, training only (interactive generation rejects CUDA). Only present when built with `--features cuda`.

`run_http_server` (in `runtime_orchestration.rs`) is generic over `B: Backend + Send + Sync + 'static` and is dispatched off `BackendChoice` the same way demo/training is, so the API works on either backend.

The `cuda` Cargo feature (`rusty-gpt`'s `cuda` ⇒ `burn/cuda`) is **off by default**. All cuda-touching code — the `Cuda`/`CudaDevice` imports in `main.rs`, the `BackendChoice::Cuda` enum variant in `runtime_config.rs`, the cuda branch in `main`, the `"cuda"` arm in `parse_backend_name`, and the `backend_can_be_selected_from_args` test — is gated on `#[cfg(feature = "cuda")]`. A complementary `backend_cuda_arg_requires_feature` test under `#[cfg(not(feature = "cuda"))]` locks in the "rebuild with `--features cuda`" error message. Add new cuda references behind the same gate or CI (which builds CPU-only) will fail.

The integration test `default_runtime.rs` additionally asserts that a default-config run does **not** load `libcuda` — keep the CPU codepath free of CUDA backend instantiation even when the feature is enabled.

### Training loop pattern

Every `train(...)` impl follows the same shape (see `TrivialModel::train` for the canonical version):
- Build model + `AdamWConfig` optimizer + `CrossEntropyLoss`.
- For each step: `loader.next_batch` ⇒ forward ⇒ `language_model_loss` (reshape `[B*T, V]` against `[B*T]`) ⇒ `loss.backward()` ⇒ `GradientsParams::from_grads` ⇒ `optimizer.step(lr, model, grads)`.
- Log on `should_log_training_step(step, steps, eval_interval)`: training loss + a value-loss probe via `value_loss(...)` against the held-out 10% tail.
- `MiniGpt::train` additionally wires gradient clipping (`GradientClippingConfig::Norm(grad_clip_norm)`).

Train/value split: `split_training_and_value_tokens` reserves the **last 10%** for value loss and shrinks the value-loader's block size if the tail is shorter than `block_size`.

Each `train(...)` returns `TrainingOutcome<M> { model, metrics }` where `TrainingMetrics` carries `final_value_loss` and `final_perplexity`. The `TrainingProgress` / `TrainingCompleted` observability events were extended to log perplexity.

### Tokenizer & batching invariants

The tokenizer is chosen by model:

- `RuntimeTokenizer::Char(CharTokenizer)` — used by `Trivial`, `SingleAttention`, `MultiAttention`. Built from the corpus via `CharTokenizer::from_text` (sorted-unique chars ⇒ id, so the same corpus always produces the same vocab).
- `RuntimeTokenizer::Bpe(BpeTokenizer)` — used by `MiniGpt` and `MoeGpt`. Loaded from `checkpoints/tokenizer.json` (or `RUSTY_GPT_BPE_TOKENIZER`) — **not derived from the corpus**. If the file is missing, `runtime_assets::load_minigpt_tokenizer` errors with the exact `cargo run --bin train-tokenizer ...` command to run.

Other invariants:
- `CharTokenizer::encode` panics on unknown chars; **use `try_encode` for any user-supplied input** (the interactive loop and the HTTP `/generate` endpoint do this). The BPE path is byte-based and never panics.
- `DataLoader::next_batch` samples `batch_size` random start positions in `0..tokens.len()-block_size`, returns `(x, y)` where `y` is `x` shifted right by one token — both shaped `[batch_size, block_size]` as `Int` tensors.

### MultiHeadAttention shape rule

`MultiHeadAttention::new(d_model, num_heads)` **panics** if `d_model % num_heads != 0` (also asserts `num_heads > 0`). The defaults (`EMBED_DIM=128`, `NUM_HEADS=4`, `HEAD_DIM=32`) satisfy this; preserve divisibility when changing constants.

### HTTP server module

The served model comes from one of three places at startup: a fresh template (default), `--load-checkpoint <path>`, or `--load-latest-checkpoint` (newest `*.mpk` in `checkpoints/`). The two `--load-*` flags are mutually exclusive — `runtime_config.rs` enforces this at parse time.

`GET /api/health` is the liveness probe: returns status, uptime, model shape, checkpoint source (`"none" | "explicit" | "latest"`), checkpoint+tokenizer sha256, and **only file basenames — never absolute paths** (information-disclosure boundary, enforced by `health_never_exposes_absolute_path` test). It is intentionally outside any future rate limiter so monitoring probes don't get 429'd.

`GenerateResponse` includes per-layer/per-head attention matrices (`AttentionData { layer, head, weights }`) and, for MoeGPT, optional `routing[]` expert assignments for visualization; the React UI in `mini-gpt-ui/` is the primary consumer.

`POST /api/generate` accepts an optional `top_k` alongside `prompt`, `max_tokens`, and `temperature`. API requests require `temperature > 0` (sampling); `top_k == 0` is rejected. Greedy decoding (`GenerationOptions::greedy()`, temperature 0) stays available internally for benchmarks/tests. Generation entry points have `_with_options` variants in `src/model/mod.rs`.

`POST /api/generate` validates `prompt` byte length and `max_tokens` before tokenizer/model work, applies a route-local body-size limit of `max_prompt_bytes + 4096`, then consumes one token from the in-process rate limiter. `GET /api/info` and `GET /api/health` stay exempt. The limiter is per-process; scaled replicas multiply the effective limit.

### Sibling tooling

- `--corpus` on `train-tokenizer` accepts a local path or an `hf://` dataset URI (same loader as `--input`). Running it is a prerequisite for MiniGPT — there is no auto-train fallback.
- `scripts/` holds convenience wrappers; `run_training.sh` is fully flag-driven (`--help` lists them). See the README.

## Gotchas

- **Checkpoints**: Burn's `NamedMpkFileRecorder` appends `.mpk` automatically. Pass the path **without** the extension (e.g. `checkpoints/mini_gpt`); the actual file is `checkpoints/mini_gpt.mpk`. `--checkpoint` and `RUSTY_GPT_MINIGPT_CHECKPOINT` follow the same convention and are confined to `checkpoints/`; bare names such as `mini_gpt` resolve under that directory. The `checkpoints/` directory is gitignored.
- **Checkpoint metadata sidecar**: MiniGPT and MoeGPT saves also write `<checkpoint>.metadata.json` next to the `.mpk` weights (`src/model/persistence.rs`). It records model kind/shape, MoE expert shape when present, tokenizer path + sha256, input source, training hyperparameters, final value loss/perplexity, optional aux loss, and git commit. Two loaders exist: `load_model_with_metadata_validation` fails on model-shape mismatch but tolerates a missing sidecar (legacy `.mpk` files still load); `load_model_with_strict_metadata_validation` (the production path) additionally rejects a missing sidecar and any tokenizer-path/hash mismatch with a diff-style error.
- **MiniGPT and MoeGPT need `checkpoints/tokenizer.json` to exist** — there is no auto-train fallback. If absent, `runtime_assets::load_minigpt_tokenizer` returns an error containing the exact `train-tokenizer` command to run. `RUSTY_GPT_BPE_TOKENIZER` overrides the path.
- **Strict checkpoint loading rejects tokenizer-hash mismatches**: if you trained against tokenizer A and try to load that checkpoint against a differently-trained `tokenizer.json`, the strict loader bails with the expected vs. actual sha256. The lenient loader exists for tests/back-compat — don't use it in production codepaths.
- **Server routes live under `/api`** — `src/server/mod.rs` defines `/generate`, `/info`, and `/health`, but `run_http_server` nests them under `/api`. `curl http://localhost:8787/generate` returns 404; the right paths are `/api/generate`, `/api/info`, `/api/health`.
- **New API endpoints must opt into request body limits intentionally** — `POST /api/generate` attaches its body-size limit directly to that route so health/info are not constrained. Any new body-bearing `/api/*` endpoint should add an explicit route-local limit instead of relying on a global default.
- **`--serve` only hosts MiniGPT and MoeGPT** — the three char-level variants are training-only and cannot be served. `--load-checkpoint` and `--load-latest-checkpoint` are mutually exclusive (parse-time error).
- **Checkpoint dir scan** — `--load-latest-checkpoint` reads from the hardcoded `checkpoints/` directory (`DEFAULT_CHECKPOINT_DIR` in `src/runtime_assets.rs`), regardless of `--checkpoint`. The `--checkpoint` flag only matters for explicit save/load paths.
- **Interactive mode constraints**: `--interactive-generate` requires `--backend cpu` and `--model minigpt` or `--model moe-gpt`; any other combination errors out in `run_cpu_demo` / `main`.
- **CPU default must stay CUDA-free**: `tests/default_runtime.rs` greps stderr/stdout for `libcuda` and fails if it appears. Keep `BackendChoice::Cpu` away from CUDA types.
- **`compare` is a pseudo-variant**: it is expanded to the five real models via `ModelChoice::comparison_models()` before forward/training dispatch. The forward and training match statements on `ModelChoice::Compare` are `unreachable!()` and must stay that way.
- **Env-mutating tests use `unsafe`**: Rust 2024 makes `env::set_var` / `env::remove_var` unsafe. Existing tests in `main.rs` wrap them in `unsafe { ... }` blocks with a SAFETY comment — follow the same pattern when adding more.
- **Burn features**: `burn` is pulled in with `["train", "ndarray", "wgpu"]` in `Cargo.toml`. The `wgpu` feature isn't exercised at runtime but kept so the CPU build stays portable; `cuda` is the only opt-in feature.
- **`data-secret/` is gitignored**: anything you drop there (e.g. `fafolang.txt`, `claude_src.txt`) won't be committed. Use it for corpora you don't want in the repo.
- **Graceful shutdown is training-only**: `runtime_signals` only installs the SIGINT/SIGTERM handler on the training path. First interrupt breaks at the next step boundary so `runtime_training` can save `<checkpoint>.interrupted-step-<N>.mpk` and exit with `INTERRUPTED_EXIT_CODE` (130); a second interrupt within ~2s aborts immediately. **Do not install the handler on the serve/interactive paths** — they rely on the default Ctrl-C abort. The `.interrupted-step-*` save is never pruned by the `--checkpoint-keep` retention window.
- **`--input` / `--corpus` are validated before any I/O**: `loader::input_source::InputSource::parse` is purely syntactic (scheme, charset, dataset/revision shape); local files additionally pass `validate_local_size` against `DEFAULT_MAX_LOCAL_INPUT_BYTES` (1 GiB) using `fs::metadata().len()` only. Consumers must branch on the parsed enum — never re-parse the raw string.
- **Authoritative config reference lives in `docs/configuration.md`**: consult it for any flag, env var, default, or constraint. Do not copy its table back into this file — a second copy only drifts.
