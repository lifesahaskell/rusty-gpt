# Sprint 03 — Training-via-API and the UI's live training surface

- **Sprint window:** 2026-06-15 → 2026-06-26 (2 weeks)
- **Sprint ID:** `2026-06-15-sprint-03`
- **Theme:** With the runtime stable (S1) and the API hardened (S2), turn `rusty-gpt` into something a non-CLI user can drive end-to-end — kick off a training run, watch it progress, and resume from where they left off. Pay down the last major maintainability rock (`src/model/mod.rs`) while the surrounding code is still fresh.

## Sprint goal

Add an async training endpoint pair (`POST /api/train` + `GET /api/train/{run_id}/status`), surface live progress in the React UI with a stop button, ship `--resume-checkpoint` for continuing training from a saved snapshot, and split `src/model/mod.rs` into focused submodules without behavior change.

## Value distribution (this sprint)

- product: 4 (T1, T2, T3, T4)
- maintainability: 1 (T5)
- security: 0

Sprint 03 is intentionally product-heavy: the security floor was raised in S2, the durability floor was raised in S1, so the team can spend this sprint on user-facing capabilities with confidence the base is solid.

## Task list

| ID | Title | Value | Size | Suggested agent |
|---|---|---|---|---|
| [T1](T1-api-train-endpoint.md) | `POST /api/train` triggers async MiniGPT training, returns run ID | product | L | fullstack-react-rust-engineer |
| [T2](T2-api-train-status.md) | `GET /api/train/{run_id}/status` reports step, loss, ETA | product | M | fullstack-react-rust-engineer |
| [T3](T3-ui-live-training.md) | Expose live training progress in `mini-gpt-ui/` (loss curve, step, stop button) | product | L | fullstack-react-rust-engineer |
| [T4](T4-resume-checkpoint.md) | `--resume-checkpoint` flag to continue training from existing checkpoint | product | M | fullstack-react-rust-engineer |
| [T5](T5-split-model-mod.md) | Split `src/model/mod.rs` into focused submodules (no behavior change) | maintainability | L | fullstack-react-rust-engineer |

## Dependencies

- **T1 → T2 → T3** form a hard chain — the UI work in T3 requires both endpoints from T1 and T2.
- **T4** is independent of the API work and can land anytime in the sprint. Schedule it early so the engineer has time to back-port the `--resume-checkpoint` path into the `POST /api/train` payload as an optional `from_checkpoint` field (a nice-to-have, not required).
- **T5** is independent and can run in parallel with everything else, but it touches a hot file (`src/model/mod.rs`) so coordinate merge order with T1/T4 (which both depend on training internals).
- The S1 graceful-shutdown + periodic-checkpoint work (S1-T2, S1-T3) is the foundation for T1 — the async training run must inherit the same SIGINT-safety and periodic-save behavior.

## Risks

- The L-sized tasks (T1, T3, T5) account for most of the sprint capacity. If any one slips, defer T5 to Sprint 04 rather than rushing it — a half-finished `src/model/mod.rs` split is worse than not starting.
- Async training in-process requires careful state management — the model lives in `ServerState`, training mutates it, generation reads from it. Without a clean ownership story, `/api/generate` could return mid-training garbage. Two acceptable solutions: (1) train a separate model instance and atomically swap on completion, (2) lock generation while training is active and return 503. Choose deliberately in T1.
- The React UI in `mini-gpt-ui/` is described in CLAUDE.md as a black-box consumer with its own toolchain. Coordinate with whoever owns it before promising a delivery date for T3.
- `--resume-checkpoint` (T4) interacts with the metadata sidecar from `src/model/persistence.rs` — a model-shape mismatch on resume must fail fast with a clear message, not crash mid-step.

## Exit criteria

- A user can `POST /api/train` with a JSON payload, receive a `run_id`, poll `GET /api/train/{run_id}/status`, and see step / loss / ETA updates until the run completes (or is stopped).
- The React UI shows a loss curve that updates live during training, a step counter, and a stop button that calls a DELETE / cancellation endpoint.
- `cargo run -- --model minigpt --train-steps 100 --resume-checkpoint checkpoints/mini_gpt.step-5000` continues training from step 5000, not step 0.
- `src/model/mod.rs` is split into at least three submodules (`definitions.rs`, `training.rs`, `generation.rs` or similar) with no behavior change; `cargo test` and `cargo clippy --all-targets -- -D warnings` both pass.
- `tests/default_runtime.rs` still passes.
- All four model variants (`trivial`, `single-attention`, `multi-attention`, `minigpt`) still train end-to-end via CLI.

## Out of scope (parking lot for Sprint 04+)

- Multi-tenant training (multiple concurrent runs per server) — Sprint 03 supports one run at a time.
- Auth / API keys on `/api/train` — the rate limiter from S2-T1 is the only gate. If `/api/train` is exposed beyond localhost, S4 must add real auth.
- Distributed training across multiple GPUs / multiple processes — single-process only.
- Model-evaluation dashboard (perplexity over time across runs) — adjacent product surface; a separate "evaluation sprint" if appetite emerges.
- Splitting `src/main.rs` into runtime-config / training-orchestration / server-startup modules (called out in `docs/project-refinement-phase.md`) — defer to Sprint 04 if S3-T5 lands cleanly.
