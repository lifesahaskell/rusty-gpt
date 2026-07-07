# S3-T4 — `--resume-checkpoint` flag to continue training from existing checkpoint

- **Value:** product
- **Size:** M (1–2 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** S1-T3 (periodic checkpoints), S1-T1 (clean persistence path)
- **Blocks:** —

## Context

S1 added durability (interrupt + periodic save) but offered only manual resume — the user has to remember which step they were at and start a fresh run. This task closes the loop: a single flag continues training from a saved checkpoint, preserving step count and optimizer state where possible.

## Goal

Add `--resume-checkpoint <PATH>` (and the matching env var) that loads the named checkpoint, resumes training from its step count, and continues to the configured `--train-steps` total.

## Acceptance criteria

- New CLI flag `--resume-checkpoint <PATH>` / env var `RUSTY_GPT_RESUME_CHECKPOINT`.
- Path follows the same convention as `--checkpoint` (no `.mpk` extension; subject to the S2-T3 confinement check).
- On resume:
  - Model weights are loaded from `<path>.mpk`.
  - The matching `<path>.metadata.json` sidecar is read; the model shape **must** match the current `Hyperparameters` or the run fails fast with a clear error naming the mismatch.
  - Starting step is read from the sidecar's `step` field (saved by S1-T3). If the sidecar is missing (legacy checkpoint), starting step defaults to `0` with a warning.
  - Training runs from `starting_step + 1` to `train_steps` (i.e. `--train-steps 10000 --resume-checkpoint <path-at-step-5000>` runs an additional 5000 steps to reach 10000 total — **not** 10000 more steps).
  - If `train_steps <= starting_step`, exit with a clear message ("checkpoint is already at step N, requested total is M; nothing to do").
- Optimizer state is **not** resumed in S3 — Burn's `AdamWConfig` rebuilds the optimizer from scratch with the loaded weights. Document this limitation explicitly: the first few resumed steps will have a different LR-schedule effect than a continuous run. Resuming optimizer state is an S4+ enhancement.
- `--resume-checkpoint` is mutually exclusive with `--load-checkpoint` and `--load-latest-checkpoint` (parse-time error).
- Integration test: train 4 steps, save, resume from the checkpoint and run 4 more, assert the final-step count is 8 and the checkpoint chain is intact.
- README "Common commands" gains an example: `cargo run --release -- --model minigpt --train-steps 20000 --resume-checkpoint checkpoints/mini_gpt.step-10000`.
- CLAUDE.md "Runtime configuration" table gains the new flag.

## Implementation notes

- Reuse `load_model_with_metadata_validation` from `src/model/persistence.rs` — do not reimplement the sidecar parsing.
- The starting-step value flows into the training loop as a `start_step: usize` parameter. The existing `should_log_training_step(step, steps, eval_interval)` helper needs to be re-checked: does it use the absolute step or the relative step? Match the semantics that makes the training log continuous across the resume boundary.
- Periodic checkpoints (S1-T3) named with step numbers must continue the numbering — a checkpoint saved at absolute step 6000 should be named `<checkpoint>.step-6000.mpk`, not `step-1000`.
- Failure modes to test: shape mismatch, missing `.mpk`, missing sidecar (warn + start from 0), corrupt sidecar (fail fast).

## Definition of done

- PR merged.
- The S1 graceful-shutdown + periodic-save + resume flow is now a complete round-trip story; document it as one paragraph in the development runbook.
- T1's optional `from_checkpoint` field in `POST /api/train` can be wired to use this code path — coordinate the merge order.
