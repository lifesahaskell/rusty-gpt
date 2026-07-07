# S3-T1 — `POST /api/train` triggers async MiniGPT training, returns run ID

- **Value:** product
- **Size:** L (2–3 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** S1-T1 (clean runtime modules), S1-T2 (SIGINT-safe training), S1-T3 (periodic checkpoints)
- **Blocks:** T2, T3

## Context

Today, training happens only via CLI — there is no programmatic way for the React UI (or any other client) to kick off a training run. The `docs/project-refinement-phase.md` explicitly calls for "first-class training lifecycle support beyond CLI logs: start/status/result endpoints, persisted run metadata, and UI progress integration." This task is the start endpoint.

## Goal

Add an Axum route `POST /api/train` that accepts a training-config JSON payload, spawns the training loop on a background task, and immediately returns a `run_id` the client can poll.

## Acceptance criteria

- New route: `POST /api/train` accepts:

  ```json
  {
    "train_steps": 1000,
    "learning_rate": 1e-4,
    "checkpoint_interval": 100,
    "eval_interval": 100,
    "from_checkpoint": "mini_gpt.step-5000"  // optional, nice-to-have if T4 lands first
  }
  ```

- Returns 202 Accepted with body `{"run_id": "<uuid>"}`. The server immediately starts the training run on a background task (`tokio::spawn`) and the HTTP response returns within < 100 ms.
- Only **one** training run can be active at a time (sprint scope). A second `POST /api/train` while one is running returns 409 Conflict with the active `run_id`.
- During an active training run, `POST /api/generate` either: (a) returns 503 Service Unavailable with `Retry-After`, or (b) generates from a snapshot taken at run start. **Pick (a) for S3** — simpler and clearer to debug; revisit in S4 if the UX warrants the snapshot approach.
- All standard rate-limit / body-size / validation rules from Sprint 02 apply. `max_steps`, `max_learning_rate`, etc. should be capped via the same flag pattern as S2-T2.
- Persisted state: each run writes a `run-<uuid>.json` manifest under `checkpoints/runs/` capturing the request payload, start time, end time (when known), final status, and the periodic-checkpoint paths it produced.
- Training inherits the S1 SIGINT/SIGTERM behavior — a server shutdown during an active run saves the current state and marks the manifest `interrupted`.
- Integration test: POST a 2-step training run, poll until the manifest shows `completed`, assert a checkpoint was written. (Pair this test with T2 once T2 lands.)
- README / docs/configuration.md document the endpoint, payload, and concurrency limit.

## Implementation notes

- The shared state: `ServerState` already holds the model and tokenizer. Add an `Arc<Mutex<Option<TrainingRun>>>` (or a typed wrapper) to track the active run. `Option` because most of the time there is no run.
- Background task: spawn via `tokio::task::spawn_blocking` (training is CPU/GPU-bound, not async-friendly). The task gets a clone of the run handle and updates progress fields through a channel or shared atomic counters.
- `run_id`: use `uuid::Uuid::new_v4()`. Persist in the manifest immediately so a crashed server's runs are recoverable / auditable.
- Be careful with the lock around `ServerState.model` — training mutates it. The cleanest design is: clone the initial weights into the training task's owned `MiniGpt`, train on the clone, swap atomically on completion. Don't hold a long-lived mutex across the full training loop.
- This is a security-relevant endpoint — if exposure widens beyond localhost, real auth must be added in S4. Add a clear note in the route's docstring: "ONLY safe for localhost deployment until auth is added."

## Definition of done

- PR merged.
- Manifest format is documented (e.g. in `docs/configuration.md`) so T2's status endpoint can parse it cleanly.
- Sprint backlog updated: T2 and T3 are unblocked.
