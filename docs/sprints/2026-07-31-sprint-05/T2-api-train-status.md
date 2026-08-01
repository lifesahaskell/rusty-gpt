# S5-T2 — `GET /api/train/{run_id}/status` reports step, loss, ETA

- **Value:** product
- **Size:** M (1–2 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** T1 (manifest format + progress-update mechanism)
- **Blocks:** T3, T4

## Goal

A polling endpoint that reports live progress for the run T1 started, and the terminal state once it finishes.

## Acceptance criteria

- `GET /api/train/{run_id}/status` returns:

  ```json
  {
    "run_id": "...",
    "status": "running",
    "step": 340,
    "total_steps": 1000,
    "loss": 2.14,
    "value_loss": 2.31,
    "eta_seconds": 118,
    "started_at": "...",
    "finished_at": null
  }
  ```

  `status` is one of `running | completed | stopped | interrupted | failed`. `finished_at` and the final-metrics fields populate once the run leaves `running`.

- Unknown `run_id` returns `404`. A `run_id` for a completed/older run still returns its terminal manifest (status history isn't ephemeral — it's the same `checkpoints/runs/run-<uuid>.json` T1 wrote).
- `eta_seconds` is derived from steps-per-second observed so far in this run, not a fixed estimate — recompute it each poll from `(total_steps - step) / (step / elapsed_seconds)`.
- Exempt from the rate limiter the same way `/api/health` and `/api/info` are — a polling UI hitting this every second shouldn't burn the `/api/generate` budget. (It's a GET with no training side effect, so this is safe.)
- Integration test: `POST /api/train` a 2-step run, poll `/api/train/{id}/status` until `status == "completed"`, assert the final manifest has `final_value_loss` populated (mirrors the existing `TrainingMetrics` shape from `src/model/training.rs`).

## Implementation notes

- Read straight from the manifest file T1 maintains — don't stand up a second source of truth. If T1's progress-update mechanism was an in-memory atomic rather than a manifest write-through, this endpoint reads that atomic for `running` state and falls back to the manifest file once the run is terminal.
- `docs/configuration.md` gets the response schema; keep it in the same section as T1's request schema so the two are read together.

## Definition of done

- PR merged, T3 and T4 unblocked.
