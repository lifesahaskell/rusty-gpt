# S2-T5 — Tighten CI to `cargo clippy -D warnings`

- **Value:** maintainability
- **Size:** S (1–2 hours)
- **Suggested agent:** senior-devops-engineer
- **Depends on:** S1-T4 (clippy debt must be at zero first)
- **Blocks:** —

## Context

CLAUDE.md states the CI workflow currently runs clippy **without** `-D warnings` specifically because `src/model/mod.rs` has pre-existing style lints. S1-T4 closes those lints; this task closes the loop by making CI enforce no-new-warnings going forward. Without this step, the S1-T4 work decays — new warnings will accumulate until the next "clippy burn-down sprint."

## Goal

Update `.github/workflows/ci.yml` to fail the build on any clippy warning, in both CPU and CUDA-feature configurations.

## Acceptance criteria

- `.github/workflows/ci.yml` runs:
  - `cargo clippy --all-targets -- -D warnings` on CPU (current job, tightened).
  - `cargo clippy --all-targets --features cuda -- -D warnings` on CPU (verify the gate compiles clean; no GPU needed since clippy doesn't run kernels).
- The CI matrix still runs `cargo fmt --all -- --check` and `cargo test`.
- A pre-flight check: re-run CI on a recent PR (or open a no-op PR) and confirm green with the new bar.
- The CHANGELOG / release notes / commit message mentions the new bar so contributors know.
- CLAUDE.md "Gotchas" entry about clippy is **removed** (replaced by the implicit `cargo clippy -D warnings` bar).
- Local development is still ergonomic: `cargo clippy --all-targets --fix` is suggested in the development runbook for contributors who want to clean up before pushing.

## Implementation notes

- Run clippy as a separate CI step (not chained with `&&` to test) so the failure mode is obvious in the GitHub UI.
- Don't add `--all-features` unless every feature combination compiles — for this repo, `--features cuda` is the only non-default feature and it doesn't need a GPU to type-check.
- If a clippy lint added by a future toolchain bump breaks CI, the answer is to either fix it or `#[allow(...)]` at the narrowest scope with a `// Why:` comment — **not** to remove `-D warnings`.

## Definition of done

- PR merged, CI green at the new bar.
- A short paragraph in the development runbook explains the bar and how to fix common lints locally.
- S3 work proceeds under the stricter bar from day one.
