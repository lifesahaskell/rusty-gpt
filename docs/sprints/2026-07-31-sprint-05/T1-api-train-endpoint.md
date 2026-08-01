# S5-T1 — `POST /api/train` triggers async MiniGPT training, returns run ID

- **Value:** product
- **Size:** L (2–3 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** none (S1's SIGINT-safe training and periodic checkpoints already exist)
- **Blocks:** T2, T3, T4

## Context

This is Sprint 03's T1, re-scoped: it was fully designed then, never implemented. `grep -rn "api/train" src/` still returns nothing. The design below is the same one, carried forward — nothing about it has aged since the repo hasn't grown a competing training-lifecycle mechanism in the meantime.

## Goal

Add an Axum route `POST /api/train` that accepts a training-config payload, spawns the training loop on a background task, and immediately returns a `run_id` the client can poll.

## Acceptance criteria

- New route: `POST /api/train` accepts:

  ```json
  {
    "train_steps": 1000,
    "learning_rate": 1e-4,
    "checkpoint_interval": 100,
    "eval_interval": 100,
    "from_checkpoint": "mini_gpt.step-5000"
  }
  ```

  `from_checkpoint` is optional and reuses the `--resume-checkpoint` path already built for the CLI (S3-T2) — no new resume logic, just a new caller.

- Returns `202 Accepted` with body `{"run_id": "<uuid>"}`. The HTTP response returns within <100ms; training runs on `tokio::task::spawn_blocking` (it's CPU/GPU-bound, not async-friendly).
- Only **one** training run active at a time. A second `POST /api/train` while one is running returns `409 Conflict` with the active `run_id`.
- While a run is active, `POST /api/generate` returns `503 Service Unavailable` with `Retry-After`. Simpler than snapshotting a pre-training model; revisit only if the UX demands otherwise.
- All Sprint 02 request-cap rules apply: cap `train_steps`, `learning_rate`, etc. via the same flag pattern as `--max-prompt-bytes` / `--max-output-tokens` (S2-T2).
- Persisted state: each run writes `checkpoints/runs/run-<uuid>.json` with the request payload, start time, end time, final status, and the periodic-checkpoint paths it produced. T2 reads this file; don't let its shape drift out from under T2.
- Training inherits SIGINT/SIGTERM behavior from `runtime_signals`: a server shutdown mid-run saves current state and marks the manifest `interrupted`, same as the CLI path.
- `docs/configuration.md` documents the endpoint, payload, and the one-run-at-a-time limit.

## Implementation notes

- `ServerState` gets `Arc<Mutex<Option<TrainingRun>>>` (or a typed wrapper) tracking the active run — `Option` because there usually isn't one.
- Model ownership: clone the current weights into the training task's own `MiniGpt`, train on the clone, swap `ServerState.model` atomically on completion. Never hold a lock across the training loop — `/api/generate`'s 503 during training is a deliberate, cheap alternative to that lock contention.
- `run_id` via `uuid::Uuid::new_v4()`, persisted to the manifest immediately (not just on completion) so a crashed server's runs are auditable.
- Progress updates flow to T2 through the manifest file or a shared atomic/channel — pick whichever your `spawn_blocking` task can update without fighting the async runtime; document the choice in the PR since T2 depends on it.
- Localhost-only warning: add a comment on the route noting auth is out of scope for this sprint (see SPRINT.md Risks).

## Definition of done

- PR merged.
- Manifest format documented so T2 can parse it without guessing.
- T2 and T3 unblocked.
