# S1-T3 — Periodic mid-run MiniGPT checkpointing every N steps

- **Value:** product / reliability
- **Size:** M (1–2 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** T1 (modules overlap), T2 (share the metadata sidecar conventions)
- **Blocks:** S3-T4 (`--resume-checkpoint`) depends on having well-formed mid-run snapshots to resume from

## Context

The other half of the training-durability gap from `project_training_durability.md`. SIGINT handling (T2) catches operator-initiated stops; periodic saves protect against crashes, OOM kills, power loss, and "I forgot to interrupt it for 4 hours" scenarios. Together they make long training runs safe to walk away from.

## Goal

Add a configurable cadence at which MiniGPT training writes a numbered checkpoint mid-run, without dominating wall-time for small models or excessive disk for long runs.

## Acceptance criteria

- New CLI flag `--checkpoint-interval <N>` (env var `RUSTY_GPT_CHECKPOINT_INTERVAL`) controls the cadence. `0` (or unset) disables periodic saves; positive integers save every N training steps.
- Default value is `0` (disabled) to preserve current behavior for tiny demo runs.
- Periodic save filename includes the step number: `<checkpoint>.step-<N>.mpk` with matching `.step-<N>.metadata.json`.
- Saved metadata sidecar records `step: N`, `interval: <N>`, plus the normal model-shape / tokenizer / hyperparam fields.
- An on-by-default retention policy keeps only the **last K** periodic checkpoints (default `K=3`) and prunes older ones in the same directory. K is controlled by `--checkpoint-keep <K>` / `RUSTY_GPT_CHECKPOINT_KEEP`. The final end-of-run checkpoint and the SIGINT-interrupted checkpoint are never pruned.
- Periodic saves do **not** count toward `--eval-interval` — they are an orthogonal cadence.
- Integration test runs `--train-steps 4 --checkpoint-interval 2 --checkpoint-keep 1`, asserts exactly one numbered checkpoint exists at the end plus the final.
- `--load-latest-checkpoint` (see `DEFAULT_CHECKPOINT_DIR` in `src/main.rs:31`) correctly picks the highest-step periodic checkpoint, not an older final.
- No regression in `cargo test --test default_runtime`.

## Implementation notes

- The save itself must not block the next training step longer than necessary. For S1, a synchronous save is acceptable — async double-buffering is a Sprint 03 enhancement if profiling shows it matters.
- Retention pruning should `fs::read_dir` once per save, sort by step number embedded in the filename, and delete the (oldest, `len > K`) tail. Be defensive: ignore files that don't match the expected pattern.
- Confirm the metadata sidecar is removed alongside its `.mpk` during pruning — orphaned sidecars will confuse `load_model_with_metadata_validation`.
- The retention policy is a **destructive operation in `checkpoints/`** — log every deletion to stderr so the user can audit.

## Definition of done

- PR merged to `main`.
- CLAUDE.md "Runtime configuration" table gains the two new flags.
- `project_training_durability.md` memory updated: durability gap is now closed.
- README "Common commands" adds an example: `cargo run --release --features cuda -- --backend cuda --model minigpt --train-steps 100000 --checkpoint-interval 1000`.
