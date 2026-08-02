# S5-T3 — `DELETE /api/train/{run_id}` stops an active run, checkpoints in place

- **Value:** product
- **Size:** S (~1 day)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** T1 only (not T2 — this needs T1's run-tracking and the `request_interrupt` fn T1 exposes, not the status endpoint. Can run in parallel with T2 once T1 merges.)
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

- Reuse the exact stop-flag mechanism already proven in `runtime_signals` for SIGINT: `runtime_signals::INTERRUPT_REQUESTED`, checked by the training loop at every step boundary. `DELETE` calls `runtime_signals::request_interrupt()` (the `pub fn` T1 promotes from the current `#[cfg(test)] _test_request_interrupt`) instead of a signal handler setting it; the training loop's response is identical either way.
- **This flag is a process-global static, not scoped per-run.** That's only safe because T1 enforces one run at a time *and* resets the flag at the start of every run (`runtime_signals::reset_interrupt()`, see T1's acceptance criteria). Do not implement T3 without confirming T1's reset call actually landed — if it didn't, this endpoint will work exactly once per server process and then silently break every future run. Add the two-runs integration test described in T1 here too if T1 didn't already cover it: stop run A, start run B, confirm B completes.
- This is the one new design decision this sprint beyond what Sprint 03 scoped: **stop is graceful-by-checkpoint, not graceful-by-cancellation-token-with-partial-state**. Keep it that way — a step-boundary stop is simple, testable, and consistent with how the CLI already behaves under Ctrl-C.

## Definition of done

- PR merged, T4 unblocked.
