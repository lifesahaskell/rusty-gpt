# Sprint 05 — Training-via-API, for real this time

- **Sprint window:** 2026-07-31 → 2026-08-13 (2 weeks)
- **Sprint ID:** `2026-07-31-sprint-05`
- **Theme:** Sprint 03 (`2026-06-15-sprint-03`) planned `POST /api/train` + a live-progress UI and closed without either landing — only the CLI-side `--resume-checkpoint` flag (S3-T2) shipped. `mini-gpt-ui/src/components/TrainingDashboard.tsx` exists today as a "Training" tab that accepts drag-and-dropped files into local `useState` and calls no API at all. This sprint finishes the job: a real training endpoint, a status endpoint, a stop endpoint, and a UI panel that is actually wired to them. It also finishes the `src/model/mod.rs` split that S3-T5 started but didn't complete — `mod.rs` is still 1966 lines.

## Sprint goal

A user can trigger a MiniGPT training run from the UI (or `curl`), watch step/loss/ETA update live, stop it early, and have it survive a server SIGTERM the same way CLI training does. Separately, `src/model/mod.rs` shrinks to a thin re-export module with model definitions, attention, and the block/feed-forward stack in their own files.

## Value distribution (this sprint)

- product: 4 (T1, T2, T3, T4)
- maintainability: 1 (T5)
- security: 0 (see risks — this sprint does not add auth)

## Task list

| ID | Title | Value | Size | Suggested agent |
|---|---|---|---|---|
| [T1](T1-api-train-endpoint.md) | `POST /api/train` triggers async MiniGPT training, returns run ID | product | L | fullstack-react-rust-engineer |
| [T2](T2-api-train-status.md) | `GET /api/train/{run_id}/status` reports step, loss, ETA | product | M | fullstack-react-rust-engineer |
| [T3](T3-api-train-stop.md) | `DELETE /api/train/{run_id}` stops an active run, checkpoints in place | product | S | fullstack-react-rust-engineer |
| [T4](T4-ui-live-training.md) | Replace the dead `TrainingDashboard.tsx` stub with a live training panel wired to T1–T3 | product | L | fullstack-react-rust-engineer |
| [T5](T5-split-model-mod.md) | Finish the `src/model/mod.rs` split into submodules (no behavior change) | maintainability | M | fullstack-react-rust-engineer |

## Dependencies

- **T1 → T2 → T3 → T4** is a hard chain. T4 cannot be reviewed against a fake backend — it needs all three routes live.
- **T5** is independent and can run the entire sprint in parallel, but it touches `src/model/mod.rs`, the same file T1's training-task wiring will read from (`MiniGpt`, `TrainingOutcome`). Land T5 first or rebase T1 on top of it — don't let both sit open against the same file for two weeks.
- T1 reuses the SIGINT-safe checkpoint-save path from `runtime_signals` (S1-T2) and the periodic-checkpoint retention from S1-T3. Nothing new to build there, just call the existing machinery from the background task instead of the CLI path.

## Risks

- **T1 is the whole sprint's critical path.** It was already scoped in Sprint 03's T1 doc (concurrency-of-one via `Arc<Mutex<Option<TrainingRun>>>`, `run_id` via `uuid::Uuid::new_v4()`, manifest under `checkpoints/runs/`, `/api/generate` returns 503 while training is active). Nothing about the design has aged — reuse it rather than re-deriving. The only new decision is T3's stop semantics (see T3).
- **Training mutates `ServerState.model`; generation reads it.** Sprint 03's chosen approach — train a cloned model, swap atomically on completion, `/api/generate` returns 503 mid-run — still stands. Don't hold a lock across the training loop.
- **`TrainingDashboard.tsx` isn't a blank slate — it's actively wrong.** It has file-drop state, a status string, and zero API calls. T4 should gut the file-drop UI (there is no corpus-upload endpoint and adding one is out of scope) rather than bolt live-progress polling on top of it.
- **This sprint does not add auth.** `/api/train` is exposed the same way `/api/generate` is today — fine for localhost, not fine if `rusty-gpt` ever gets deployed past that. Carry this forward as a standing parking-lot item until it's actually addressed.

## Exit criteria

- `curl -X POST http://127.0.0.1:8787/api/train -d '{"train_steps": 200, ...}'` returns a `run_id` within 100ms; the run trains in the background.
- `curl http://127.0.0.1:8787/api/train/<run_id>/status` reports `step`, `loss`, `eta_seconds`, and `status` (`running` | `completed` | `stopped` | `interrupted` | `failed`) until the run finishes.
- `curl -X DELETE http://127.0.0.1:8787/api/train/<run_id>` stops the run and writes a checkpoint marked `stopped: true` in its manifest, same shape as the existing `interrupted` sidecar convention.
- The UI's Training tab starts a run, shows a live loss curve and step counter, and has a working stop button — no dead file-drop UI left behind.
- `src/model/mod.rs` is under ~400 lines (re-exports + top-level model structs' `impl` glue only); attention, `Mlp`/`FeedForward`/`Block`, and the ~700-line test module live in their own files.
- `cargo test`, `cargo clippy --all-targets -- -D warnings`, `cargo clippy --all-targets --features cuda -- -D warnings`, `cargo fmt --all -- --check` all pass. `tests/default_runtime.rs` still passes (no `libcuda` load on the CPU default path).

## Out of scope (parking lot)

Carried forward from Sprint 03, still true:

- Multi-tenant training (concurrent runs per server) — one run at a time.
- Auth / API keys on `/api/train` — see Risks. Required before any non-localhost deployment.
- Distributed training across multiple GPUs / processes.
- Model-evaluation dashboard (perplexity over time across runs).
- Corpus upload via API — training still reads from `--input` / `hf://` sources configured server-side; there is no endpoint to push a corpus file to the server.

Not this sprint, tracked separately:

- **Research playground roadmap** (`docs/research/research-playground-roadmap.md`) — six ablation-experiment milestones on the local 4060 Ti, starting with checkpoint/eval cadence. This is hands-on GPU experimentation, not an engineering task, and doesn't block or get blocked by this sprint. Next unclaimed step there is Milestone 1.
