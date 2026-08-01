# S5-T1 — `POST /api/train` triggers async MiniGPT training, returns run ID

- **Value:** product
- **Size:** XL (3–4 days — see Context; this absorbed scope a first pass missed)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** none in the docs sense, but see Implementation notes — this task changes `ServerState.model`'s type, which every existing route reads directly
- **Blocks:** T2, T3, T4

## Context

Sprint 03 scoped this endpoint and never built it — `grep -rn "api/train" src/` still returns nothing. A design review before this sprint started found the original Sprint 03 spec understated real scope in two ways, both now folded into acceptance criteria below:

1. There is **no existing mechanism to continue training the live serving model**. `MiniGpt::train` / `train_with_periodic_save` always build a fresh model from `MiniGptConfig`; the only resume path is `MiniGpt::train_prebuilt_with_periodic_save` (`src/model/training.rs`), which loads weights from a **checkpoint file** via the strict metadata loader, not from `ServerState.model` in memory. Don't design around "clone the current weights" — it doesn't exist.
2. `ServerState.model` (`src/server/mod.rs`) is currently a bare field, read directly with no lock at roughly 15 call sites across `/api/generate`, `/api/info`, `/api/health`. Making it swappable on training completion requires giving it real interior mutability, and that touches all of those read sites, not just the new route. This is real scope, not an implementation detail.

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
    "resume_from": "mini_gpt.step-5000"
  }
  ```

  `resume_from` is optional and named to match the existing `--resume-from` CLI flag / `RUSTY_GPT_RESUME_FROM` env var (`src/runtime_config.rs`) — do not call it `from_checkpoint` or invent new naming. It routes to `MiniGpt::train_prebuilt_with_periodic_save` exactly like the CLI path does: load via the strict metadata loader, read `completed_steps` from the sidecar, pass as `TrainingParams::start_step`.

- Returns `202 Accepted` with body `{"run_id": "<uuid>"}`. The HTTP response returns within <100ms; training runs on `tokio::task::spawn_blocking` (it's CPU/GPU-bound, not async-friendly).
- Only **one** training run active at a time. A second `POST /api/train` while one is running returns `409 Conflict` with the active `run_id`.
- While a run is active, `POST /api/generate` returns `503 Service Unavailable` with `Retry-After`. Simpler than snapshotting a pre-training model; revisit only if the UX demands otherwise.
- All Sprint 02 request-cap rules apply: cap `train_steps`, `learning_rate`, etc. via the same flag pattern as `--max-prompt-bytes` / `--max-output-tokens` (S2-T2).
- Persisted state: each run writes `checkpoints/runs/run-<uuid>.json` with the request payload, start time, end time, final status, and the periodic-checkpoint paths it produced. T2 reads this file; don't let its shape drift out from under T2.
- Training inherits SIGINT/SIGTERM behavior from `runtime_signals`: a server shutdown mid-run saves current state and marks the manifest `interrupted`, same as the CLI path.
- **The training loop's global interrupt flag is reset at the start of every run.** `runtime_signals::INTERRUPT_REQUESTED` is a process-global static with no production reset path today (only `#[cfg(test)] _test_reset`). Promote it to a real `pub fn reset_interrupt()` and call it before entering the step loop for every `POST /api/train`-initiated run. Without this, a stopped run (T3) leaves the flag permanently set and **every subsequent training run on that server process dies on its first step**, silently, until the process restarts. Add an integration test: start run A, stop it (once T3 lands — stub with a direct `runtime_signals::request_interrupt()` call if T3 hasn't merged yet), start run B, assert B reaches `completed`.
- `docs/configuration.md` documents the endpoint, payload, and the one-run-at-a-time limit.

## Implementation notes

- **`ServerState.model` needs interior mutability.** Change its type to something swappable under a lock (e.g. `Mutex<ServedModel<B>>` or equivalent) and update every existing read site (`/api/generate`, `/api/info`, `/api/health`, plus their tests) to lock/read through it. This is the single biggest chunk of this task — budget for it explicitly, it is not a one-line change.
- `ServerState` gets `Arc<Mutex<Option<TrainingRun>>>` (or a typed wrapper) tracking the active run — `Option` because there usually isn't one.
- Model provenance: build a fresh `MiniGpt` from config, or load one via `train_prebuilt_with_periodic_save` if `resume_from` is set. Train that instance independently of `ServerState.model`; swap it into `ServerState.model` only on successful completion. Never hold the `ServerState.model` lock across the training loop — `/api/generate`'s 503 during training is the deliberate, cheap alternative to that lock contention.
- `run_id` via `uuid::Uuid::new_v4()`, persisted to the manifest immediately (not just on completion) so a crashed server's runs are auditable.
- **Progress updates: give the training task its own `EventLogger::with_sink(LogFormat::Json, ...)`.** The sink parses each rendered line back into fields and updates the run's shared status (`Arc<Mutex<RunStatus>>` or the manifest) on `training_progress` / `training_completed` events. This is not a new mechanism — it's the exact capture-and-parse pattern already proven in `generate_logs_request_lifecycle` (`src/server/mod.rs`), which captures `EventLogger` sink output into a `Vec<String>` and does `serde_json::from_str` on each line. Reuse it; don't invent a second channel/atomic mechanism.
- Localhost-only warning: add a comment on the route noting auth is out of scope for this sprint (see SPRINT.md Risks).

## Definition of done

- PR merged.
- Manifest format documented so T2 can parse it without guessing.
- `ServerState.model`'s new locking type is stable — T2/T3/T4 build against it, don't change its shape again after this merges.
- T2 and T3 unblocked.
