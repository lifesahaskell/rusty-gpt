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

# Compare all four model variants on the same batch
cargo run -- --model compare

# Load a saved MiniGPT checkpoint and chat with it
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

| Flag | Env var | Values / default |
|---|---|---|
| `--backend` | `RUSTY_GPT_BACKEND` | `cpu` (default) \| `cuda` — `cuda` requires building with `--features cuda` |
| `--input` | `RUSTY_GPT_INPUT` | path or `hf://…` URI, default `data/input.txt` |
| `--model` | `RUSTY_GPT_MODEL` | `minigpt` (default, alias `mini-gpt`) \| `trivial` \| `single-attention` \| `multi-attention` \| `compare` |
| `--checkpoint` | `RUSTY_GPT_MINIGPT_CHECKPOINT` | path without `.mpk`, default `checkpoints/mini_gpt` |
| `--interactive-generate` | — | flag; requires `--backend cpu` and `--model minigpt` |
| `--serve` | — | flag; starts the Axum HTTP server instead of the demo/training run |
| `--load-checkpoint` | — | flag; load MiniGPT weights from `--checkpoint` before serving / interactive |
| `--load-latest-checkpoint` | — | flag; load the newest `*.mpk` in `checkpoints/` (mutually exclusive with `--load-checkpoint`) |
| `--server-addr` | `RUSTY_GPT_SERVER_ADDR` | `host:port`, default `127.0.0.1:8787` |
| `--max-prompt-bytes` | `RUSTY_GPT_MAX_PROMPT_BYTES` | int, default `8192`; generate prompt byte cap |
| `--max-output-tokens` | `RUSTY_GPT_MAX_OUTPUT_TOKENS` | int, default `512`; generate `max_tokens` cap |
| `--rate-limit-rps` | `RUSTY_GPT_RATE_LIMIT_RPS` | int, default `5`; `0` disables generate rate limiting |
| `--rate-limit-burst` | `RUSTY_GPT_RATE_LIMIT_BURST` | int, default `10`; generate burst capacity |
| `--log-format` | `RUSTY_GPT_LOG_FORMAT` | `plain` \| `json` — default depends on backend; controls `observability::EventLogger` output |
| `--benchmark-generation` | — | flag; runs naive-vs-cached generation benchmarks (requires `--model minigpt` or `compare`) |
| — | `RUSTY_GPT_BPE_TOKENIZER` | path, default `checkpoints/tokenizer.json` |
| `--block-size` | `RUSTY_GPT_BLOCK_SIZE` | int, default `128`; must be > 0 |
| `--batch-size` | `RUSTY_GPT_BATCH_SIZE` | int, default `32`; must be > 0 |
| `--embed-dim` | `RUSTY_GPT_EMBED_DIM` | int, default `128`; must be divisible by `num_heads` |
| `--num-heads` | `RUSTY_GPT_NUM_HEADS` | int, default `4`; must be > 0 |
| `--num-layers` | `RUSTY_GPT_NUM_LAYERS` | int, default `4`; must be > 0 |
| `--dropout` | `RUSTY_GPT_DROPOUT` | f64, default `0.1`; must be `>= 0` and `< 1` |
| `--learning-rate` | `RUSTY_GPT_LEARNING_RATE` | f64, default `1e-4`; must be > 0 |
| `--lr-schedule` | `RUSTY_GPT_LR_SCHEDULE` | `constant` (default) \| `cosine`; `constant` is behaviour-neutral, `cosine` = warmup + cosine decay to `--min-learning-rate` |
| `--warmup-steps` | `RUSTY_GPT_WARMUP_STEPS` | int, default `0`; must be `< train_steps`; only used by the `cosine` schedule |
| `--min-learning-rate` | `RUSTY_GPT_MIN_LEARNING_RATE` | f64, default `0.0`; must be `>= 0` and `<= learning_rate`; only used by the `cosine` schedule |
| `--train-steps` | `RUSTY_GPT_TRAIN_STEPS` | int, must be > 0 |
| `--eval-interval` | `RUSTY_GPT_EVAL_INTERVAL` | int (0 ⇒ log only final step) |
| `--generate-tokens` | `RUSTY_GPT_GENERATE_TOKENS` | int, must be > 0 |
| `--grad-clip-norm` | `RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM` | f32, must be > 0 |
| `--prefetch-batches` | `RUSTY_GPT_PREFETCH_BATCHES` | int, default `2` (CPU prefetch queue depth) |
| `--checkpoint-interval` | `RUSTY_GPT_CHECKPOINT_INTERVAL` | int, default `0` (disabled); save `<checkpoint>.step-<N>.mpk` every N steps during MiniGPT training |
| `--checkpoint-keep` | `RUSTY_GPT_CHECKPOINT_KEEP` | int, default `3`; retention window for periodic snapshots — older `.step-N.` files are pruned, the final save and any `.interrupted-step-*` save are never pruned |

`Hyperparameters::from_env_and_overrides` (in `src/runtime_config.rs`) resolves env → CLI overrides → `validate()`, which enforces the positivity/divisibility rules and recomputes `head_dim = embed_dim / num_heads`. Invalid combinations (e.g. `embed_dim` not divisible by `num_heads`) fail at config-parse time.

## Architecture

```
src/
  lib.rs                        re-exports loader / model / observability / server / tokenizer / utils for the bins + tests
  main.rs                       thin entry point: parses RuntimeConfig, then hands off to the runtime_* modules below
  runtime_config.rs             `RuntimeConfig`, `BackendChoice`, `ModelChoice`, `Hyperparameters`, `RuntimeEnv`; parses + validates CLI args and `RUSTY_GPT_*` env vars
  runtime_assets.rs             corpus / tokenizer / checkpoint resolution (`load_input_text`, `load_minigpt_tokenizer`, `latest_checkpoint_path`); also dispatches `hf://` URIs to `loader::huggingface`
  runtime_orchestration.rs      picks demo / training / interactive / server / benchmark dispatch (`run_cpu_demo`, `run_http_server_with_runtime`)
  runtime_training.rs           training-demo orchestration: train/value split, periodic + interrupt-driven checkpoint saves with metadata sidecar, observability events
  runtime_signals.rs            process-wide SIGINT/SIGTERM flag for graceful training shutdown; `install_training_signal_handler`, `interrupt_requested`, `INTERRUPTED_EXIT_CODE = 130`. Unix-only; no-op on other targets. Only installed on the **training** path — serving and interactive inference keep default Ctrl-C abort.
  observability.rs              `EventLogger`, `RuntimeEvent`, `LogFormat`; emits structured stdout events consumed by `scripts/*` and downstream tools
  bin/
    train-tokenizer.rs          CLI to train a BPE tokenizer and write checkpoints/tokenizer.json
    collect-source.rs           CLI to concatenate a repo's source files into data/<name>.txt
  tokenizer/
    mod.rs                      `RuntimeTokenizer` enum dispatching to Char or Bpe; `Tokenizer` trait
    char.rs                     deterministic char-level tokenizer (sorted-unique chars ⇒ id)
    bpe.rs                      BPE tokenizer + `BpeTrainer`; loads/saves JSON
  loader/
    mod.rs / data.rs            random-window batch sampler producing (x, y) shifted by 1; `BatchPrefetcher` for CPU prefetch
    input_source.rs             strict `InputSource::parse` for `--input`/`--corpus` (local path vs `hf://org/dataset[@rev]`); purely syntactic — no I/O. `validate_local_size` enforces `DEFAULT_MAX_LOCAL_INPUT_BYTES = 1 GiB` via `fs::metadata` only. Every consumer must branch on the parsed enum, not re-parse the raw string.
    huggingface.rs              fetches `hf://dataset?config=…&split=…` URIs via the HF datasets-server API; caches under `data/huggingface-cache/`
  model/
    mod.rs                      four model variants + shared loss/log helpers; `MiniGptConfig`, `TrainingLogFormat`, `TrainingLogContext`
    generation.rs               `GenerationOptions` (temperature, top_k), `sample_from_logits`, `select_token_from_logits`
    training.rs                 per-variant `train(...)` impls (lifted out of `mod.rs`)
    persistence.rs              save/load wrappers around burn's NamedMpkFileRecorder (.mpk files) + the `.metadata.json` sidecar
  server/mod.rs                 Axum router exposing POST /generate, GET /info, GET /health (nested under /api by run_http_server)
  utils/mod.rs                  generation benchmarking helpers (`benchmark_generation`, `benchmark_generation_cases`)
tests/
  default_runtime.rs            binary-level smoke test asserting CPU default does not load libcuda
  graceful_shutdown.rs          Unix-only (`#![cfg(unix)]`): spawns `cargo run --`, sends SIGINT mid-training, asserts the partial `<checkpoint>.interrupted-step-<N>.mpk` save lands and the process exits with code 130
  periodic_checkpoints.rs       exercises `--checkpoint-interval` / `--checkpoint-keep`: numbered snapshots written, oldest `.step-N.` pruned, final + interrupted saves never pruned
  fixtures/input.txt            small corpus used by the integration tests
  fixtures/tokenizer.json       BPE tokenizer fixture used by checkpoint-save unit tests
scripts/                        helper shell scripts: run_training.sh, run_local.sh, run_e2e_tests.sh, build_release_candidate.sh, start_dev_server.sh, test_cuda_passthrough.sh, install_nvidia_container_toolkit.sh, and devcontainer e2e probes test_devcontainer_{generate,server,ui}.sh
docker/                         Dockerfile.cpu (debian:slim runtime) and Dockerfile.cuda; both multi-stage with cargo registry/target cache mounts
compose.yaml + compose.override.yaml  three Docker Compose profiles: `bootstrap` (one-shot train-tokenizer), `train` (one-shot CUDA MiniGPT training), `serve` (default — HTTP API + React UI). Binds ./checkpoints and ./data into containers.
docs/                           configuration.md (canonical flag/env reference), development-runbook.md (operational recipes), release-and-evaluation.md, sprints/ (sprint plans)
mini-gpt-ui/                    separate React frontend that calls the /api server — see its own README
data/input.txt                  ~1 MB Shakespeare-style training corpus (committed)
data-secret/                    gitignored corpora directory (e.g. fafolang.txt, claude_src.txt)
checkpoints/                    gitignored .mpk model files + tokenizer.json (BPE artifact for MiniGPT)
.github/workflows/ci.yml        GitHub Actions: cargo fmt --check, strict CPU/CUDA cargo clippy, cargo test on push/PR
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
- `RuntimeTokenizer::Bpe(BpeTokenizer)` — used by `MiniGpt`. Loaded from `checkpoints/tokenizer.json` (or `RUSTY_GPT_BPE_TOKENIZER`) — **not derived from the corpus**. If the file is missing, `runtime_assets::load_minigpt_tokenizer` errors with the exact `cargo run --bin train-tokenizer ...` command to run.

Other invariants:
- `CharTokenizer::encode` panics on unknown chars; **use `try_encode` for any user-supplied input** (the interactive loop and the HTTP `/generate` endpoint do this). The BPE path is byte-based and never panics.
- `DataLoader::next_batch` samples `batch_size` random start positions in `0..tokens.len()-block_size`, returns `(x, y)` where `y` is `x` shifted right by one token — both shaped `[batch_size, block_size]` as `Int` tensors.

### MultiHeadAttention shape rule

`MultiHeadAttention::new(d_model, num_heads)` **panics** if `d_model % num_heads != 0` (also asserts `num_heads > 0`). The defaults (`EMBED_DIM=128`, `NUM_HEADS=4`, `HEAD_DIM=32`) satisfy this; preserve divisibility when changing constants.

### HTTP server module

`src/server/mod.rs` exposes an Axum `Router<SharedServerState<B>>` with `POST /generate`, `GET /info`, and `GET /health`. `runtime_orchestration::run_http_server` nests that router under `/api` (final routes: `/api/generate`, `/api/info`, `/api/health`), binds to `--server-addr` (default `127.0.0.1:8787`), and serves with `axum::serve`.

`ServerState` holds a `MiniGpt`, a `RuntimeTokenizer`, a `B::Device`, an `EventLogger`, `ServerProvenance` (uptime + checkpoint source/basename/sha256 + tokenizer sha256, populated once at startup by `run_http_server`), generate request limits, and the in-process token bucket. The model can come from one of three places at startup: a fresh template (default), `--load-checkpoint <path>`, or `--load-latest-checkpoint` (newest `*.mpk` in `checkpoints/`). The two `--load-*` flags are mutually exclusive — `runtime_config.rs` enforces this at parse time.

`GET /api/health` is the liveness probe: returns status, uptime, model shape, checkpoint source (`"none" | "explicit" | "latest"`), checkpoint+tokenizer sha256, and **only file basenames — never absolute paths** (information-disclosure boundary, enforced by `health_never_exposes_absolute_path` test). It is intentionally outside any future rate limiter so monitoring probes don't get 429'd.

`GenerateResponse` includes per-layer/per-head attention matrices (`AttentionData { layer, head, weights }`) for visualization; the React UI in `mini-gpt-ui/` is the primary consumer.

`POST /api/generate` accepts an optional `top_k` alongside `prompt`, `max_tokens`, and `temperature`. API requests require `temperature > 0` (sampling); `top_k == 0` is rejected. Greedy decoding (`GenerationOptions::greedy()`, temperature 0) stays available internally for benchmarks/tests. Generation entry points have `_with_options` variants in `src/model/mod.rs`.

`POST /api/generate` validates `prompt` byte length and `max_tokens` before tokenizer/model work, applies a route-local body-size limit of `max_prompt_bytes + 4096`, then consumes one token from the in-process rate limiter. `GET /api/info` and `GET /api/health` stay exempt. The limiter is per-process; scaled replicas multiply the effective limit.

### Sibling tooling

- `src/bin/train-tokenizer.rs` — `cargo run --bin train-tokenizer -- --corpus <path-or-hf-uri> --vocab-size <n> --output <path.json>`. `--corpus` accepts a local path or an `hf://` dataset URI (same loader as `--input`). Emits JSON progress events to stdout. Required to produce the BPE tokenizer MiniGPT depends on.
- `src/bin/collect-source.rs` — `cargo run --bin collect-source -- --repo <path> [--output <path>]`. Concatenates source files from a repo into `data/<name>.txt` for use as a training corpus.
- `scripts/run_training.sh`, `scripts/run_local.sh`, `scripts/run_e2e_tests.sh`, `scripts/build_release_candidate.sh` — convenience wrappers; see the README for details. `run_training.sh` is fully flag-driven: `--backend`, `--model`, `--checkpoint`, `--tokenizer`, `--train-tokenizer`, `--vocab-size`, `--cargo-profile`, `--train-steps`, `--eval-interval`, `--prefetch-batches`, `--log-format`, `--benchmark*`, `--artifacts-dir`. The old `RUSTY_GPT_*` env overrides are **deprecated** — still honored (CLI flags win) but each emits a deprecation warning; use the equivalent flag instead.
- `mini-gpt-ui/` — separate React frontend with its own README and toolchain. Calls the Rust server's `/api` routes. **Out of scope** for this memo; treat it as a black-box consumer.

## Gotchas

- **Checkpoints**: Burn's `NamedMpkFileRecorder` appends `.mpk` automatically. Pass the path **without** the extension (e.g. `checkpoints/mini_gpt`); the actual file is `checkpoints/mini_gpt.mpk`. `--checkpoint` and `RUSTY_GPT_MINIGPT_CHECKPOINT` follow the same convention and are confined to `checkpoints/`; bare names such as `mini_gpt` resolve under that directory. The `checkpoints/` directory is gitignored.
- **Checkpoint metadata sidecar**: MiniGPT saves also write `<checkpoint>.metadata.json` next to the `.mpk` weights (`src/model/persistence.rs`). It records model shape, tokenizer path + sha256, input source, training hyperparameters, final value loss/perplexity, and git commit. Two loaders exist: `load_model_with_metadata_validation` fails on model-shape mismatch but tolerates a missing sidecar (legacy `.mpk` files still load); `load_model_with_strict_metadata_validation` (the production path) additionally rejects a missing sidecar and any tokenizer-path/hash mismatch with a diff-style error.
- **MiniGPT needs `checkpoints/tokenizer.json` to exist** — there is no auto-train fallback. If absent, `runtime_assets::load_minigpt_tokenizer` returns an error containing the exact `train-tokenizer` command to run. `RUSTY_GPT_BPE_TOKENIZER` overrides the path.
- **Strict checkpoint loading rejects tokenizer-hash mismatches**: if you trained against tokenizer A and try to load that checkpoint against a differently-trained `tokenizer.json`, the strict loader bails with the expected vs. actual sha256. The lenient loader exists for tests/back-compat — don't use it in production codepaths.
- **Server routes live under `/api`** — `src/server/mod.rs` defines `/generate`, `/info`, and `/health`, but `run_http_server` nests them under `/api`. `curl http://localhost:8787/generate` returns 404; the right paths are `/api/generate`, `/api/info`, `/api/health`.
- **New API endpoints must opt into request body limits intentionally** — `POST /api/generate` attaches its body-size limit directly to that route so health/info are not constrained. Any new body-bearing `/api/*` endpoint should add an explicit route-local limit instead of relying on a global default.
- **`--serve` only hosts MiniGPT** — the other three model variants are training-only and cannot be served. `--load-checkpoint` and `--load-latest-checkpoint` are mutually exclusive (parse-time error).
- **Checkpoint dir scan** — `--load-latest-checkpoint` reads from the hardcoded `checkpoints/` directory (`DEFAULT_CHECKPOINT_DIR` in `src/runtime_assets.rs`), regardless of `--checkpoint`. The `--checkpoint` flag only matters for explicit save/load paths.
- **Interactive mode constraints**: `--interactive-generate` requires both `--backend cpu` and `--model minigpt`; any other combination errors out in `run_cpu_demo` / `main`.
- **CPU default must stay CUDA-free**: `tests/default_runtime.rs` greps stderr/stdout for `libcuda` and fails if it appears. Keep `BackendChoice::Cpu` away from CUDA types.
- **`compare` is a pseudo-variant**: it is expanded to the four real models via `ModelChoice::comparison_models()` before forward/training dispatch. The forward and training match statements on `ModelChoice::Compare` are `unreachable!()` and must stay that way.
- **Env-mutating tests use `unsafe`**: Rust 2024 makes `env::set_var` / `env::remove_var` unsafe. Existing tests in `main.rs` wrap them in `unsafe { ... }` blocks with a SAFETY comment — follow the same pattern when adding more.
- **Burn features**: `burn` is pulled in with `["train", "ndarray", "wgpu"]` in `Cargo.toml`. The `wgpu` feature isn't exercised at runtime but kept so the CPU build stays portable; `cuda` is the only opt-in feature.
- **`data-secret/` is gitignored**: anything you drop there (e.g. `fafolang.txt`, `claude_src.txt`) won't be committed. Use it for corpora you don't want in the repo.
- **Graceful shutdown is training-only**: `runtime_signals` only installs the SIGINT/SIGTERM handler on the training path. First interrupt breaks at the next step boundary so `runtime_training` can save `<checkpoint>.interrupted-step-<N>.mpk` and exit with `INTERRUPTED_EXIT_CODE` (130); a second interrupt within ~2s aborts immediately. **Do not install the handler on the serve/interactive paths** — they rely on the default Ctrl-C abort. The `.interrupted-step-*` save is never pruned by the `--checkpoint-keep` retention window.
- **`--input` / `--corpus` are validated before any I/O**: `loader::input_source::InputSource::parse` is purely syntactic (scheme, charset, dataset/revision shape); local files additionally pass `validate_local_size` against `DEFAULT_MAX_LOCAL_INPUT_BYTES` (1 GiB) using `fs::metadata().len()` only. Consumers must branch on the parsed enum — never re-parse the raw string.
- **Authoritative config reference lives in `docs/configuration.md`**: the flag/env table in this file is a fast lookup; if a default or constraint disagrees with `docs/configuration.md`, treat that file as truth and update here.
