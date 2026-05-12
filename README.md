# rusty-gpt

A character-level GPT built from scratch in Rust on top of the [Burn](https://burn.dev) deep-learning framework. The crate trains four progressively richer models — from a plain embedding-plus-linear baseline up to a multi-block transformer — on a Shakespeare-style corpus, and can serve a trained MiniGPT interactively at the terminal or (module-only, see below) over HTTP.

## Quick start

```bash
# Default: CPU, trivial model, data/input.txt
cargo run

# Train the full MiniGPT (CPU)
cargo run --release -- --model minigpt

# Train on CUDA (requires the CUDA toolkit installed; opt in via the `cuda` Cargo feature)
cargo run --release --features cuda -- --backend cuda --model minigpt

# Compare all four model variants on the same batch
cargo run -- --model compare

# Chat with a saved MiniGPT checkpoint
cargo run -- --model minigpt --interactive-generate --checkpoint checkpoints/mini_gpt
```

Run the test suite (unit tests plus the binary-level smoke test) with:

```bash
cargo test
```

## Configuration

CLI flags and `RUSTY_GPT_*` environment variables can both drive runtime behavior; the CLI wins when both are set.

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--backend cpu\|cuda` | `RUSTY_GPT_BACKEND` | `cpu` | `cuda` is only available when the crate is built with `--features cuda` (requires the CUDA toolkit). |
| `--input <path>` | `RUSTY_GPT_INPUT` | `data/input.txt` | Plain UTF-8 text. |
| `--model <name>` | `RUSTY_GPT_MODEL` | `trivial` | `trivial`, `single-attention`, `multi-attention`, `minigpt` (alias `mini-gpt`), `compare`. |
| `--checkpoint <path>` | `RUSTY_GPT_MINIGPT_CHECKPOINT` | `checkpoints/mini_gpt` | Path without `.mpk` — Burn appends it. |
| `--interactive-generate` | — | off | Requires `--backend cpu` and `--model minigpt`. |
| — | `RUSTY_GPT_TRAIN_STEPS` | `1000` | |
| — | `RUSTY_GPT_EVAL_INTERVAL` | `100` | `0` ⇒ log only the final step. |
| — | `RUSTY_GPT_GENERATE_TOKENS` | `80` | |
| — | `RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM` | `1.0` | Must be > 0. |

Model-shape hyperparameters (`BLOCK_SIZE`, `BATCH_SIZE`, `EMBED_DIM`, `NUM_HEADS`, `NUM_LAYERS`, `DROPOUT`, `LEARNING_RATE`) are compile-time constants at the top of `src/main.rs`.

## What's in the box

- `src/tokenizer/char.rs` — deterministic character-level tokenizer.
- `src/loader/data.rs` — random-window batch sampler returning `(x, y)` where `y` is `x` shifted by one token.
- `src/model/mod.rs` — four models built up step by step:
  1. `TrivialModel` — embedding → linear head.
  2. `SingleAttentionModel` — single causal-masked attention head.
  3. `MultiAttentionModel` — fused-QKV multi-head attention.
  4. `MiniGpt` — token + position embeddings, a stack of pre-norm transformer blocks, final layer norm, LM head; supports greedy generation and attention introspection.
- `src/model/persistence.rs` — `.mpk` checkpoint save/load via Burn's `NamedMpkFileRecorder`.
- `src/server/mod.rs` — Axum router exposing `POST /generate` and `GET /info` with per-layer/per-head attention matrices. **Implemented and unit-tested; not yet wired up from `main.rs`.**
- `tests/default_runtime.rs` — runs the binary and asserts the CPU default path never loads `libcuda`.

## License

Unlicensed personal project.
