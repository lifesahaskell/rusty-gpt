# S1-T2 — Graceful shutdown on SIGINT/SIGTERM saves in-progress MiniGPT checkpoint

- **Value:** product / reliability
- **Size:** M (1–2 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** T1
- **Blocks:** —

## Context

Persistent memory entry `project_training_durability.md` flags this as the top reliability gap: long training runs lose all progress on Ctrl-C because there is no signal handler and no end-of-run save until the configured `train_steps` count is reached. Pair this with T3 (periodic in-run saves) to fully close the durability gap.

## Goal

When the user sends SIGINT (Ctrl-C) or SIGTERM during a MiniGPT training run, the process must:

1. Stop the training loop cleanly at the next step boundary.
2. Save the current model weights and metadata sidecar to a well-defined path.
3. Exit with a non-zero status code indicating "interrupted, but checkpoint saved" — distinct from a normal completion or a crash.
4. Print a clear stderr line with the saved checkpoint path so the user knows where the partial run landed.

## Acceptance criteria

- SIGINT during `cargo run --release -- --model minigpt --train-steps 10000` after at least one step saves a checkpoint named `<checkpoint>.interrupted-step-<N>.mpk` (or similar — name it consistently with T3) under `checkpoints/`.
- The accompanying `.metadata.json` sidecar (see `src/model/persistence.rs`) records `interrupted: true`, the step count at interruption, and all the usual model-shape / tokenizer / hyperparam fields.
- A second SIGINT within ~2 seconds of the first triggers immediate exit without waiting for the save — operator escape hatch.
- The signal handler does not fire during CPU inference / `--serve` / `--interactive-generate` modes, or if fires, exits cleanly without attempting a training checkpoint.
- Unit test (or integration test) using `tokio::signal` / `nix::sys::signal` simulates the interrupt and asserts the checkpoint exists on disk afterward.
- Windows build does not break — gate Unix-only signal handling with `#[cfg(unix)]` and provide a no-op (or Ctrl-C only) path for non-Unix.
- The `tests/default_runtime.rs` `libcuda` check still passes.

## Implementation notes

- Burn 0.21 training loops are synchronous; the cleanest implementation is an `AtomicBool` flag set by the signal handler and checked at the top of each step in `MiniGpt::train`.
- Use `tokio::signal::ctrl_c()` if the runtime is already tokio (the Axum server uses it). Otherwise, `signal-hook` is the lightweight choice for sync code — avoid pulling in a full async runtime for the training path.
- Reuse the metadata-emitting path from `src/model/persistence.rs` — do not duplicate the sha256/git-commit logic.
- Exit code suggestion: `130` for SIGINT-after-save (matches the shell convention of `128 + signal`), `0` for normal completion.

## Definition of done

- PR merged to `main`.
- Memory entry [[project_training_durability]] updated to reflect the SIGINT half is closed; T3 still tracks the periodic-save half.
- README "Common commands" or development runbook gains one sentence on the interrupt-and-save behavior.
