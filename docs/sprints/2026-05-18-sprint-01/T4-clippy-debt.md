# S1-T4 — Resolve pre-existing clippy warnings in `src/model/mod.rs`

- **Value:** maintainability
- **Size:** S (half day)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** T1 (model files in the in-flight diff)
- **Blocks:** S2-T5 (`cargo clippy -D warnings` in CI)

## Context

CLAUDE.md "Gotchas" explicitly flags: *"`cargo clippy --all-targets` currently reports a handful of style lints in `src/model/mod.rs` (too-many-args, complex-type, etc.). The CI workflow runs clippy without `-D warnings` for that reason. If you tighten CI, fix those lints first."*

This is the "fix those lints first" task. Sprint 02 will flip CI to strict (S2-T5).

## Goal

Bring `cargo clippy --all-targets` to zero warnings across the workspace, with no behavior change and minimal type-signature churn.

## Acceptance criteria

- `cargo clippy --all-targets -- -D warnings` exits 0 on CPU build.
- `cargo clippy --all-targets --features cuda -- -D warnings` exits 0 (verify the feature gate still compiles clean).
- No model output changes — `cargo test` continues to pass with the same assertions.
- For each warning silenced via `#[allow(clippy::...)]` rather than fixed, leave a one-line `// Why:` comment explaining the trade-off. Prefer fixing over allowing.
- For `too-many-args` warnings on training/forward signatures: introduce a small parameter struct (e.g. `TrainStepInputs`, `ForwardOptions`) rather than just suppressing.
- For `complex-type` warnings: extract a `type` alias next to the function, named after the domain concept (not the shape).

## Implementation notes

- Run `cargo clippy --all-targets --fix` cautiously — review every auto-fix; clippy occasionally rewrites idiom in ways that hurt readability.
- The forward-dispatch `unreachable!()` arms on `ModelChoice::Compare` are intentional (per CLAUDE.md). Do not let clippy rewrite them.
- If a warning category truly cannot be fixed in this scope (e.g. wide generics in `MiniGpt::generate` are load-bearing), add a `#[allow(...)]` at the narrowest possible scope and note it.

## Definition of done

- PR merged to `main`.
- The "Clippy has known pre-existing warnings" gotcha is removed from CLAUDE.md.
- Sprint 02 / T5 (CI tightening) is unblocked.
