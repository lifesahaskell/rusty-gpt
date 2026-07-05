# Sprint 3 — Training QoL + Expert Observability

Goal: make longer MoE training runs practical and inspectable — learning-rate
scheduling, resume-from-checkpoint, per-expert utilization metrics in the
observability stream, and shutdown/periodic-checkpoint parity for `moe-gpt`.
The LR and resume work is model-agnostic and benefits all five variants.

## Contents

- [S3-T1 — LR scheduling (warmup + cosine)](#s3-t1--lr-scheduling-warmup--cosine)
- [S3-T2 — Resume training from checkpoint](#s3-t2--resume-training-from-checkpoint)
- [S3-T3 — Per-expert utilization metrics](#s3-t3--per-expert-utilization-metrics)
- [S3-T4 — Shutdown + periodic-checkpoint parity for MoeGpt](#s3-t4--shutdown--periodic-checkpoint-parity-for-moegpt)
- [S3-T5 — Expert-balance logging + runbook docs](#s3-t5--expert-balance-logging--runbook-docs)
- [Sprint exit checklist](#sprint-exit-checklist)

---

## S3-T1 — LR scheduling (warmup + cosine)

Add an optional schedule to the shared training loop in
`src/model/training.rs` (`optimizer.step(lr, ...)` already takes a per-step
`lr`, so this is a pure function of the step index):

```text
lr(step) = base_lr * step / warmup_steps                          for step < warmup_steps
         = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * t))   t = progress after warmup
```

Flags (full `src/runtime_config.rs` plumbing pattern + `docs/configuration.md`):

| Flag | Env var | Default | Constraint |
| --- | --- | --- | --- |
| `--warmup-steps` | `RUSTY_GPT_WARMUP_STEPS` | `0` (disabled) | ≥ 0, < `train_steps` |
| `--lr-schedule` | `RUSTY_GPT_LR_SCHEDULE` | `constant` | `constant` \| `cosine` |
| `--min-learning-rate` | `RUSTY_GPT_MIN_LEARNING_RATE` | `0.0` | ≥ 0, ≤ `learning_rate` |

**Files**: `src/model/training.rs`, `src/runtime_config.rs`,
`docs/configuration.md`

**Acceptance criteria**
- Pure-function unit tests: lr(0)≈0 during warmup, lr(warmup_steps)=base_lr,
  lr(train_steps)≈min_lr for cosine; `constant` returns base_lr everywhere.
- Default behavior identical to today (constant schedule, no warmup) —
  existing training tests unchanged.
- `TrainingProgress` observability event gains a `learning_rate` field
  (plain + JSON formats in `src/observability.rs`).

## S3-T2 — Resume training from checkpoint

New flag `--resume-from <checkpoint>` (env `RUSTY_GPT_RESUME_FROM`, path
confined to `checkpoints/` like `--checkpoint`):

- Loads model weights via the strict loader before training, and reads the
  step count from the metadata sidecar (new sidecar field
  `#[serde(default)] completed_steps: usize`, recorded on every save
  including periodic and interrupted saves).
- Training continues from `completed_steps + 1` so the LR schedule (S3-T1)
  and `--checkpoint-interval` numbering stay continuous; total steps still
  honor `--train-steps` as the absolute target.
- Mutually exclusive with nothing, but meaningless combinations
  (`--resume-from` + `--model` ≠ minigpt/moe-gpt) fail at parse time.
- Optimizer state is **not** persisted (Burn recorder scope); document this
  limitation explicitly in the runbook — resume restarts AdamW moments.

**Files**: `src/runtime_config.rs`, `src/runtime_training.rs`,
`src/model/persistence.rs`, `docs/configuration.md`,
`docs/development-runbook.md`

**Acceptance criteria**
- Sidecar round-trip test: saves record `completed_steps`; legacy sidecars
  without it default to 0.
- Integration test: train N steps → save → resume with `--train-steps 2N` →
  the run performs exactly N more steps and the first logged step index is
  N+1; final checkpoint loads and generates.
- Resuming with a mismatched model shape fails with the strict loader's
  diff-style error.
- Works for both `minigpt` and `moe-gpt`.

## S3-T3 — Per-expert utilization metrics

Surface router health during MoeGpt training:

- `MoeForwardAux` (from S1-T3) already carries per-expert token fractions;
  aggregate per training step across blocks into an
  `ExpertUtilization { per_layer: Vec<Vec<f32>> }` (token share per expert per
  layer) plus a scalar router entropy per layer.
- Thread through `train_language_model` into `TrainingMetrics`
  (`final_expert_utilization: Option<...>`) and the
  `RuntimeEvent::TrainingProgress` / `TrainingCompleted` events in
  `src/observability.rs` — JSON format carries the full vectors; plain format
  prints a compact summary (e.g. min/max token share and entropy per layer).
- Only computed on logging steps (`should_log_training_step`) to keep the
  hot loop unchanged for dense models and cheap for MoE.

**Files**: `src/model/moe.rs`, `src/model/training.rs`,
`src/observability.rs`, `src/runtime_training.rs`

**Acceptance criteria**
- Unit test: a hand-routed batch produces the expected token-share vector and
  entropy (uniform routing → entropy ≈ ln(num_experts); collapsed → ≈ 0).
- JSON event schema test: `TrainingProgress` for MoeGpt includes the
  utilization fields; dense variants omit them
  (`#[serde(skip_serializing_if = "Option::is_none")]`, matching the sidecar
  pattern in `persistence.rs`).
- Plain-format snapshot test for the summary line.

## S3-T4 — Shutdown + periodic-checkpoint parity for MoeGpt

The graceful-shutdown and retention machinery
(`src/runtime_signals.rs`, `src/runtime_training.rs`) is model-generic but
only integration-tested against MiniGPT:

- Extend `tests/graceful_shutdown.rs` (Unix-only) with a `--model moe-gpt`
  case: SIGINT mid-training lands `<checkpoint>.interrupted-step-<N>.mpk` +
  sidecar (with `completed_steps`), exit code 130.
- Extend `tests/periodic_checkpoints.rs` with a moe-gpt case: numbered
  snapshots written, oldest pruned per `--checkpoint-keep`, final +
  interrupted saves never pruned.
- Keep runtimes reasonable: tiny hyperparameters via env
  (`RUSTY_GPT_TRAIN_STEPS`, small `--embed-dim`/`--num-layers`/`--moe-experts`).

**Files**: `tests/graceful_shutdown.rs`, `tests/periodic_checkpoints.rs`

**Acceptance criteria**
- Both extended integration suites pass on Linux CI.
- Interrupted moe-gpt checkpoint loads via the strict loader and resumes
  (ties into S3-T2).

## S3-T5 — Expert-balance logging + runbook docs

Operator-facing polish:

- Training demo output (`src/runtime_training.rs`) prints aux loss alongside
  training/value loss on logging steps for MoE runs.
- New runbook section in `docs/development-runbook.md`: "Training moe-gpt" —
  recommended starting config, how to read the expert-utilization summary,
  what expert collapse looks like (one expert's token share → 1.0, entropy →
  0) and the levers to fix it (`--moe-aux-loss-weight`, more warmup, lower LR).
- `scripts/run_training.sh` gains pass-through flags `--moe-experts`,
  `--moe-top-k`, `--moe-aux-loss-weight` (flag-driven, consistent with its
  existing style; no new env vars).

**Files**: `src/runtime_training.rs`, `docs/development-runbook.md`,
`scripts/run_training.sh`

**Acceptance criteria**
- Plain-format training log for a moe-gpt run shows aux loss on logging steps
  (snapshot/assert in an integration or unit test).
- `scripts/run_training.sh --model moe-gpt --moe-experts 4 ...` forwards the
  flags (shell-level check in `scripts/run_e2e_tests.sh` or a bats-style
  assertion if none exists — minimum bar: documented manual verification in
  the PR).
- Runbook section reviewed against an actual training run's output.

## Sprint exit checklist

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features cuda -- -D warnings
cargo fmt --all -- --check

# QoL end-to-end
RUSTY_GPT_TRAIN_STEPS=6 cargo run -- --model moe-gpt --input tests/fixtures/input.txt \
  --checkpoint-interval 2 --warmup-steps 2 --lr-schedule cosine
cargo run -- --model moe-gpt --resume-from checkpoints/mini_gpt.step-4 --train-steps 8 \
  --input tests/fixtures/input.txt
```

- `docs/configuration.md`, CLAUDE.md tables, and the runbook all updated.
