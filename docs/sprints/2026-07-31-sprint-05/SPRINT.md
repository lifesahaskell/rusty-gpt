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
| [T1](T1-api-train-endpoint.md) | `POST /api/train` triggers async MiniGPT training, returns run ID | product | XL | fullstack-react-rust-engineer |
| [T2](T2-api-train-status.md) | `GET /api/train/{run_id}/status` reports step, loss, ETA | product | M | fullstack-react-rust-engineer |
| [T3](T3-api-train-stop.md) | `DELETE /api/train/{run_id}` stops an active run, checkpoints in place | product | S | fullstack-react-rust-engineer |
| [T4](T4-ui-live-training.md) | Replace the dead `TrainingDashboard.tsx` stub with a live training panel wired to T1–T3 | product | L | fullstack-react-rust-engineer |
| [T5](T5-split-model-mod.md) | Finish the `src/model/mod.rs` split into submodules (no behavior change) | maintainability | M | fullstack-react-rust-engineer |

## Dependencies

A pre-sprint design review (see Risks) found the dependency chain was tighter than it needed to be. Corrected shape, in waves:

- **Wave 1 (parallel):** T1 and T5. T5 touches `src/model/mod.rs`'s internal file layout but its own acceptance criteria require public re-export paths (`crate::model::MiniGpt` etc.) to keep resolving unchanged — T1 only consumes those public paths, so the two don't actually conflict as long as T5 honors its own "no public API changes" bar. Land T5 first if there's any doubt; rebase T1 on top rather than let both sit open against the same file for two weeks.
- **Wave 2 (parallel, after T1 merges):** T2 and T3. Both depend only on T1 (run-tracking + the `EventLogger` status mechanism + the `request_interrupt`/`reset_interrupt` pair T1 exposes) — **not on each other**. The original plan chained them (T1 → T2 → T3) for no real reason; running them in parallel shortens the critical path by a full task.
- **Wave 3 (after T2 and T3 both merge):** T4. It needs all three real endpoints — reviewing UI work against a spec instead of running code just means re-deriving the same schema questions twice.
- T1 reuses the SIGINT-safe checkpoint-save path from `runtime_signals` (S1-T2) and the periodic-checkpoint retention from S1-T3. Nothing new to build there, just call the existing machinery from the background task instead of the CLI path.

## Risks

- **T1 is the whole sprint's critical path, and it's bigger than Sprint 03 scoped it.** Sprint 03's T1 doc assumed "clone the current serving model and train the clone" — that path doesn't exist in this codebase; training only ever starts fresh-from-config or resumed-from-checkpoint-*file* (`MiniGpt::train_prebuilt_with_periodic_save`). It also assumed `ServerState.model` could just be swapped — today it's a bare field read at ~15 call sites with no lock, so making it swappable is its own chunk of work, not a side effect. T1's doc has been updated to spell both of these out; treat its size as XL, not L.
- **The stop mechanism is a process-global flag with no production reset path.** `runtime_signals::INTERRUPT_REQUESTED` is safe for the CLI only because the process exits after one run. The server doesn't exit between runs. If T1 doesn't reset the flag at the start of every run, the first `DELETE` (T3) permanently breaks every training run after it until the server restarts — silently. T1 and T3's docs now both call this out explicitly; don't let either land without the reset call and the two-runs regression test.
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
