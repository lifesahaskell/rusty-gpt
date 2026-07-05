# Sprint 2 — MoeGpt Model, Config, and Training

Goal: a user can run `cargo run -- --model moe-gpt` and train a
Mixture-of-Experts GPT end-to-end on CPU, with the load-balancing auxiliary
loss folded into training, `compare` extended to five models, and checkpoints
(including the metadata sidecar) round-tripping.

## Contents

- [S2-T1 — `MoeGpt` model struct](#s2-t1--moegpt-model-struct)
- [S2-T2 — `ModelChoice::MoeGpt` + compare integration](#s2-t2--modelchoicemoegpt--compare-integration)
- [S2-T3 — Config plumbing for MoE flags](#s2-t3--config-plumbing-for-moe-flags)
- [S2-T4 — Aux loss in the training loop](#s2-t4--aux-loss-in-the-training-loop)
- [S2-T5 — Checkpointing + metadata sidecar](#s2-t5--checkpointing--metadata-sidecar)
- [Sprint exit checklist](#sprint-exit-checklist)

---

## S2-T1 — `MoeGpt` model struct

New model in `src/model/mod.rs` (or a `src/model/moe_gpt.rs` submodule if
`mod.rs` gets unwieldy), mirroring `MiniGpt`:

- Fields: `token_embed`, `position_embed`, `blocks: Vec<Block<B>>` (each block
  built with `FeedForward::Moe`), `ln_final`, `lm_head`, plus shape metadata
  (`vocab_size`, `max_position_embeddings`, `num_experts`, `top_k`).
- Config: extend the pattern of `MiniGptConfig` with a `MoeGptConfig`
  (embedding `MiniGptConfig` fields + `num_experts`, `top_k`,
  `aux_loss_weight`), or add optional MoE fields — prefer a separate
  `MoeGptConfig { base: MiniGptConfig, num_experts, top_k, aux_loss_weight }`
  so `MiniGptConfig` stays untouched.
- Forward paths mirror `MiniGpt`: `forward_tokens`, `forward` (one shared
  causal mask, loops blocks), plus a training-only
  `forward_tokens_with_aux(input) -> (Tensor<B, 3>, Tensor<B, 1>)` that sums
  each block's aux loss (mean over blocks, so the scale is independent of
  `num_layers`).
- Uses the **BPE tokenizer** (`checkpoints/tokenizer.json`), same as MiniGPT —
  `runtime_assets::load_minigpt_tokenizer` becomes the shared loader (rename
  or wrap as needed without breaking its error message contract).

**Files**: `src/model/mod.rs` (and/or `src/model/moe_gpt.rs`), `src/model/moe.rs`

**Acceptance criteria**
- Forward shape test: `[batch, seq]` int tokens → `[batch, seq, vocab_size]`
  logits, seq ≤ `max_position_embeddings`.
- `forward_tokens_with_aux` returns a finite scalar aux loss; with
  `num_experts=1` the aux loss is ≈ 1.0 (degenerate uniform case) and logits
  match a dense MiniGpt of the same shape/weights within tolerance.
- Accessors (`vocab_size()`, `d_model()`, `num_layers()`, `num_heads()`,
  `block_size()`, `num_experts()`, `top_k()`) exist — Sprint 4's server work
  consumes them.

## S2-T2 — `ModelChoice::MoeGpt` + compare integration

In `src/runtime_config.rs`:

- Add `ModelChoice::MoeGpt`, parse `"moe-gpt"` (alias `"moegpt"`) in
  `parse_model_name`, name it `"moe-gpt"` in the display impl.
- `comparison_models()` grows to five entries; the `unreachable!()` arms for
  `Compare` in forward/training dispatch stay as-is.
- `ModelChoice::includes_minigpt()` (`src/runtime_config.rs`) gates the
  BPE-tokenizer requirement today; either extend it or add a sibling
  `requires_bpe_tokenizer()` covering `MoeGpt` so the tokenizer is
  required/loaded for it — audit all `includes_minigpt` call sites and pick
  per-site.
- Dispatch in `runtime_orchestration.rs` / `runtime_training.rs`: `moe-gpt`
  trains like `minigpt` (CPU + CUDA), rejects nothing MiniGPT accepts except
  where later sprints add support (serving/interactive return a clear
  "not yet supported for moe-gpt" error until Sprint 4).

**Files**: `src/runtime_config.rs`, `src/runtime_orchestration.rs`,
`src/runtime_training.rs`, `src/model/mod.rs` (compare training arm)

**Acceptance criteria**
- Unit tests: `--model moe-gpt` and `RUSTY_GPT_MODEL=moe-gpt` both parse to
  `ModelChoice::MoeGpt` (follow the existing `unsafe { env::set_var }` +
  SAFETY-comment pattern for env tests).
- `comparison_models()` returns five models ending with `MoeGpt`;
  `cargo run -- --model compare` trains/evals all five.
- Integration smoke: `RUSTY_GPT_TRAIN_STEPS=1 cargo run -- --model moe-gpt
  --input tests/fixtures/input.txt` exits 0 using the fixture tokenizer.

## S2-T3 — Config plumbing for MoE flags

Follow the full 8-step pattern in `src/runtime_config.rs` (const default →
`Hyperparameters` field → env override in `from_env_and_overrides` →
`HyperparameterOverrides` → `RuntimeEnv` → CLI arm → `validate()` → docs):

| Flag | Env var | Default | Constraint |
| --- | --- | --- | --- |
| `--moe-experts` | `RUSTY_GPT_MOE_EXPERTS` | `4` | must be > 0 |
| `--moe-top-k` | `RUSTY_GPT_MOE_TOP_K` | `2` | ≥ 1 and ≤ `moe_experts` |
| `--moe-aux-loss-weight` | `RUSTY_GPT_MOE_AUX_LOSS_WEIGHT` | `0.01` | must be ≥ 0 |

Validation lives in `Hyperparameters::validate` next to the existing
`embed_dim % num_heads` check. The flags are inert for non-MoE models (no
error if set — same behavior as `--grad-clip-norm` on non-MiniGPT models).

**Files**: `src/runtime_config.rs`, `docs/configuration.md`, `CLAUDE.md`
(fast-lookup table)

**Acceptance criteria**
- Unit tests per flag: CLI wins over env; env alone works; invalid values
  (`--moe-top-k 0`, `--moe-top-k 5 --moe-experts 4`, negative aux weight)
  fail at config-parse time with actionable messages.
- `docs/configuration.md` model-shape/training tables document all three
  (that file is authoritative; CLAUDE.md table updated to match).

## S2-T4 — Aux loss in the training loop

Wire the aux loss into `train_language_model` (`src/model/training.rs`)
without disturbing the four dense variants:

- Widen the forward closure contract from `Tensor<B, 3>` to
  `(Tensor<B, 3>, Option<Tensor<B, 1>>)` — dense models' closures return
  `(logits, None)` (mechanical change at each `train(...)` call site).
- Loss: `total = ce_loss + aux_weight * aux_loss` when aux is `Some`, before
  `.backward()`. `aux_weight` flows in via `TrainingParams` (new optional
  field, builder method alongside `with_grad_clip_norm`).
- `MoeGpt::train` / `train_with_periodic_save` mirror `MiniGpt`'s (including
  `GradientClippingConfig::Norm(grad_clip_norm)`), passing
  `|model, inputs| model.forward_tokens_with_aux(inputs)`.
- `TrainingMetrics` gains `final_aux_loss: Option<f64>`; value-loss probes use
  plain CE only (aux loss is a training regularizer, not an eval metric).

**Files**: `src/model/training.rs`, `src/model/mod.rs`,
`src/runtime_training.rs`

**Acceptance criteria**
- Dense-model regression: all four existing variants train with identical
  losses to `main` for identical inputs (aux path is `None`).
- MoeGpt smoke-training test (few steps, tiny config, fixture corpus):
  loss is finite and decreases over a handful of steps; aux loss is finite.
- Aux-weight test: with `aux_loss_weight = 0.0`, total loss equals plain CE
  loss for the same batch/weights.
- `TrainingOutcome.metrics.final_aux_loss` is `Some` for MoeGpt, `None` for
  dense variants.

## S2-T5 — Checkpointing + metadata sidecar

Extend `src/model/persistence.rs` so MoeGpt checkpoints round-trip and shape
mismatches are caught:

- `CheckpointModelShape` gains `#[serde(default)] num_experts: usize` and
  `#[serde(default)] moe_top_k: usize` (0 ⇒ dense/legacy). Extend
  `checkpoint_compatibility_report` with matching `push_usize_issue` checks.
- Update both construction sites: `src/runtime_training.rs` (save path) and
  `src/runtime_assets.rs` (`expected_shape` for loading).
- MoeGpt save/load via the existing `NamedMpkFileRecorder` wrappers; the
  strict loader (`load_model_with_strict_metadata_validation`) is the
  production path, as for MiniGPT.

**Files**: `src/model/persistence.rs`, `src/runtime_training.rs`,
`src/runtime_assets.rs`

**Acceptance criteria**
- MoeGpt save → strict load round-trip test: loaded model generates identical
  tokens to the saved one for a fixed prompt (greedy).
- Legacy sidecar back-compat: a sidecar JSON **without** the MoE fields
  deserializes with `num_experts = 0` and still loads a dense MiniGPT
  (existing tests plus one explicit fixture test).
- Shape-mismatch test: loading a `num_experts=4` checkpoint into a
  `num_experts=8` model fails with a diff-style report naming the field.
- Training a MoeGpt with `--checkpoint-interval` writes `.step-N.mpk`
  snapshots and prunes per `--checkpoint-keep` (reuse
  `tests/periodic_checkpoints.rs` patterns; full integration coverage lands in
  S3-T4).

## Sprint exit checklist

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features cuda -- -D warnings
cargo fmt --all -- --check

# End-to-end smoke
RUSTY_GPT_TRAIN_STEPS=2 cargo run -- --model moe-gpt --input tests/fixtures/input.txt
cargo run -- --model compare   # five models
```

- `docs/configuration.md` and CLAUDE.md updated for the three new flags and
  the new model name.
