# Sprint 01 — Land in-flight work + harden training durability

- **Sprint window:** 2026-05-18 → 2026-05-29 (2 weeks)
- **Sprint ID:** `2026-05-18-sprint-01`
- **Theme:** Stabilize the uncommitted runtime refactor and remove the longest-standing reliability gap in MiniGPT training before any feature work resumes.

## Sprint goal

Get the in-flight runtime refactor (`src/main.rs`, `src/model/generation.rs`, `src/model/mod.rs`, `src/model/persistence.rs`, `src/runtime_assets.rs`, `src/runtime_config.rs`, `src/utils/mod.rs`) onto `main` as a coherent, test-covered change, then close the training-durability gap so long training runs survive Ctrl-C and crashes without losing all progress. Clear pre-existing clippy debt so Sprint 2 can flip CI to `-D warnings`. Land one focused security fix (input URI validation) so the work stream becomes routine instead of episodic.

## Value distribution (this sprint)

- maintainability: 2 (T1, T4)
- product / reliability: 2 (T2, T3)
- security: 1 (T5)

## Task list

| ID | Title | Value | Size | Suggested agent |
|---|---|---|---|---|
| [T1](T1-stabilize-runtime-refactor.md) | Commit and stabilize the in-flight runtime refactor | maintainability | M | fullstack-react-rust-engineer |
| [T2](T2-graceful-shutdown.md) | Graceful shutdown on SIGINT/SIGTERM saves in-progress MiniGPT checkpoint | product | M | fullstack-react-rust-engineer |
| [T3](T3-periodic-checkpoints.md) | Periodic mid-run MiniGPT checkpointing every N steps | product | M | fullstack-react-rust-engineer |
| [T4](T4-clippy-debt.md) | Resolve pre-existing clippy warnings in `src/model/mod.rs` | maintainability | S | fullstack-react-rust-engineer |
| [T5](T5-input-uri-validation.md) | Validate and constrain `--input` / `hf://` URIs before fetching | security | S | principal-security-engineer |

## Dependencies

- **T4 → Sprint 02 / T5 (CI clippy strict)** — must land in S1 so S2 can tighten CI without a flag day.
- **T1 → T2 / T3** — durability work touches the same modules currently mid-refactor; landing T1 first avoids painful rebases.
- **T2 + T3** can land in either order, but a single PR that introduces both signal-handling and step-cadence saves makes the test surface easier to reason about. Prefer sequential merges in the order T2 → T3.

## Risks

- The uncommitted diff (~7 files) may have multiple competing intents glued together. T1 should be the first issue triaged — if the diff doesn't form a single coherent change, the engineer should split it before merging.
- Burn's `NamedMpkFileRecorder` writes the entire model on each save. Periodic checkpointing every N steps can dominate wall-time if N is too small; the task spec calls for a configurable cadence with a tested floor.
- Signal handling on Linux differs from Windows; CI runs on Linux today, but graceful shutdown should not regress the Windows build path even if it skips full coverage.
- `hf://` URIs touch the network at runtime; the security validation must not break offline test fixtures.

## Exit criteria

- `git status` is clean on `main` and every previously-uncommitted file is either merged or explicitly removed with rationale.
- `cargo test` and `cargo test --test default_runtime` pass on CPU.
- A long-running MiniGPT training session can be interrupted with Ctrl-C and resumed (manually) from a saved checkpoint without rerunning all prior steps.
- `cargo clippy --all-targets` in the model crate emits no warnings (CI is not yet strict, but the warning count drops to zero).
- A regression test rejects an `hf://` URI pointing outside the allowlist with a clear error message.

## Out of scope (parking lot for Sprint 02+)

- HTTP API rate limiting, prompt/token caps, checkpoint path-traversal guards — Sprint 02.
- `--resume-checkpoint` automation — Sprint 03 (manual resume is the S1 bar).
- `src/model/mod.rs` module split — Sprint 03 (T4 only burns down lints in place).
- Any UI changes in `mini-gpt-ui/` — not in this sprint.
