# S5-T3 — `DELETE /api/train/{run_id}` stops an active run, checkpoints in place

- **Value:** product
- **Size:** S (~1 day)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** T1, T2
- **Blocks:** T4

## Goal

Give the UI's stop button (T4) something to call. Stopping a run should behave like the existing SIGINT-during-training path: finish the current step, save, exit cleanly — not like a `kill -9`.

## Acceptance criteria

- `DELETE /api/train/{run_id}` on the active run's ID: signals the training task to stop at the next step boundary, writes `<checkpoint>.stopped-step-<N>.mpk` (same sidecar convention as `.interrupted-step-<N>.mpk`, with `stopped: true` instead of `interrupted: true`), updates the manifest to `status: "stopped"`, and returns `202 Accepted`.
- `DELETE` on a `run_id` that isn't the active run (wrong ID, already finished) returns `404`.
- `DELETE` while no run is active returns `404`.
- The stopped checkpoint is never pruned by `--checkpoint-keep`, matching the existing `.interrupted-*` behavior.
- Idempotent: a second `DELETE` on an already-stopping run returns `202` again (not an error) — the client doesn't need to track whether its first stop request landed.
- Integration test: `POST /api/train` a long-enough run, `DELETE` it mid-flight, poll `/api/train/{id}/status` until `status == "stopped"`, assert the `.stopped-step-*.mpk` + sidecar exist.

## Implementation notes

- Reuse the stop-flag mechanism already proven in `runtime_signals` for SIGINT — a shared `AtomicBool` (or equivalent) the training loop checks at step boundaries. `DELETE` sets it directly instead of a signal handler setting it; the training loop's response is identical either way.
- This is the one new design decision this sprint beyond what Sprint 03 scoped: **stop is graceful-by-checkpoint, not graceful-by-cancellation-token-with-partial-state**. Keep it that way — a step-boundary stop is simple, testable, and consistent with how the CLI already behaves under Ctrl-C.

## Definition of done

- PR merged, T4 unblocked.
