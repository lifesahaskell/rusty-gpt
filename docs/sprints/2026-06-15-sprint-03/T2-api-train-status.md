# S3-T2 — `GET /api/train/{run_id}/status` reports step, loss, ETA

- **Value:** product
- **Size:** M (1–2 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** T1
- **Blocks:** T3

## Context

T1 returns a `run_id` and persists a manifest; this task makes that run observable. Without a status endpoint, the only feedback channel is server stderr — which the React UI cannot see.

## Goal

Add `GET /api/train/{run_id}/status` returning the current state of a training run, plus a `DELETE /api/train/{run_id}` cancellation endpoint.

## Acceptance criteria

- `GET /api/train/{run_id}/status` returns 200 with:

  ```json
  {
    "run_id": "<uuid>",
    "status": "running",
    "step": 42,
    "total_steps": 1000,
    "current_loss": 3.14,
    "value_loss": 3.45,
    "perplexity": 23.1,
    "started_at": "2026-06-15T10:30:00Z",
    "estimated_completion_at": "2026-06-15T10:35:00Z",
    "checkpoints": [
      {"step": 100, "path": "checkpoints/runs/<run_id>.step-100.mpk"},
      {"step": 200, "path": "checkpoints/runs/<run_id>.step-200.mpk"}
    ]
  }
  ```

- `status` is one of `"running"`, `"completed"`, `"failed"`, `"interrupted"`, `"cancelled"`. For terminal states, include `ended_at` and (for `"failed"`) an `error` field with a user-safe message.
- 404 if `run_id` is unknown (not in the active run and not in `checkpoints/runs/`).
- ETA is computed from `(steps_remaining) × (rolling avg seconds per step over last 20 steps)`. Before 20 steps have elapsed, ETA is `null`.
- `DELETE /api/train/{run_id}` cancels the active run if `run_id` matches, returns 200 with the final status. For a non-active run (already completed), returns 409 Conflict. The cancellation flow reuses the SIGINT-safe shutdown path from S1-T2 — same checkpoint, same metadata, just triggered by the API instead of a signal.
- Status endpoint is **not** rate-limited at the per-IP level applied to `/api/generate` — the UI polls every 1–2 seconds and must not 429. Use a separate, generous bucket or skip the limiter entirely for this route. Document the exemption.
- Path traversal: `run_id` is parsed as a UUID before any filesystem access — invalid `run_id` returns 400, not 500, and never reaches `fs::read`.
- Integration test: POST a 4-step run via T1, poll the status endpoint until `status == "completed"`, assert the checkpoint list is non-empty. A second test: POST a longer run, DELETE it, assert final status is `"cancelled"`.
- README / docs/configuration.md document the status endpoint and cancellation semantics.

## Implementation notes

- The in-memory side of the status (current step, current loss, last-20-step timings) is fed by the training task via an `Arc<RwLock<TrainingProgress>>` or `tokio::sync::watch`. The HTTP handler reads under a short-lived read lock.
- The disk side (manifest, checkpoint listing) is loaded only when `run_id` doesn't match the active run — saves a `fs::read_dir` on the hot path.
- Don't expose `path` as an absolute path — basename only, matching the S2-T4 convention.
- ETA is informational and approximate. Document that it can jump around significantly in the first few steps.

## Definition of done

- PR merged.
- A small `curl`-based example in the development runbook: start a run, poll for status, cancel it.
- T3 (UI integration) is unblocked.
