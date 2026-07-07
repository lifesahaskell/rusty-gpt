# S3-T5 — Split `src/model/mod.rs` into focused submodules (no behavior change)

- **Value:** maintainability
- **Size:** L (2–3 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** S1-T1 (clean runtime modules), S1-T4 (clippy debt gone), S2-T5 (strict clippy in CI catches regressions)
- **Blocks:** —

## Context

`docs/project-refinement-phase.md` explicitly lists: *"Split `src/model/mod.rs` into model definitions, generation, training, metrics, and test-support areas without changing model behavior."* The file is the largest single module in the crate and houses all four model variants, the loss helpers, the training implementations, the generation paths, and the `TrainingLogFormat` / `TrainingLogContext` types. Every cross-cutting change pays a cost in this file.

Sprint 03 is the right time: the runtime is stable, the API surface no longer requires touching this file, and CI is strict enough to catch a regression mid-PR.

## Goal

Decompose `src/model/mod.rs` into focused submodules with clear responsibilities, **without any behavior change** — same public API, same test results, same training output for identical seeds.

## Acceptance criteria

- `src/model/mod.rs` becomes a thin facade that re-exports from the new submodules.
- Suggested decomposition (final names at engineer discretion):
  - `src/model/definitions.rs` — the four `Model*` structs and their `Config`s (including `MiniGptConfig`).
  - `src/model/attention.rs` — `SingleHeadAttention`, `MultiHeadAttention`, the causal-mask helpers.
  - `src/model/training.rs` — every `impl Model { fn train(...) }`, the `TrainingOutcome` / `TrainingMetrics` types, the loss helpers (`language_model_loss`, `value_loss`, `split_training_and_value_tokens`, `should_log_training_step`).
  - `src/model/logging.rs` — `TrainingLogFormat`, `TrainingLogContext`, `TrainingProgress`, `TrainingCompleted` events.
  - Generation code already lives in `src/model/generation.rs` (per the S1 in-flight diff) — leave it.
  - `src/model/persistence.rs` already exists — leave it.
- `cargo test` passes with the **exact same** test count as before the split (107 unit + 1 integration per CLAUDE.md; allow drift if S1/S2 added tests, but no fewer).
- `cargo clippy --all-targets -- -D warnings` passes (S2-T5 bar).
- `tests/default_runtime.rs` passes — the CPU default still doesn't load `libcuda`.
- A deterministic-seed regression test: train MiniGPT for N steps with a fixed seed before and after the split, assert the final loss is bit-identical. (This is a one-off test, not necessarily committed, but it's the gold standard for "no behavior change.")
- The `unreachable!()` arms on `ModelChoice::Compare` are preserved in the dispatch path.
- `src/lib.rs` re-exports stay backward-compatible — no consumer of the crate (bins, integration tests, `mini-gpt-ui` via the server) sees an API change.
- A short architectural note in CLAUDE.md describes the new layout.

## Implementation notes

- Split in small, atomic commits — each commit moves one cohesive chunk and runs tests. Reviewer will thank you.
- Resist the temptation to "improve" code while splitting. This task is purely structural. Any refactor that changes behavior is a separate PR.
- Use `cargo expand` if circular-dependency issues appear between the new submodules (e.g. `training.rs` and `definitions.rs` calling each other).
- The four `Model*` variants and `ModelChoice` enum stay in their current crate-public form — moving them does not require changing their visibility.
- If a true circular dependency emerges (e.g. training needs definitions which need a training-specific type), introduce a `shared.rs` rather than collapsing the split.

## Definition of done

- PR merged.
- `docs/project-refinement-phase.md` gets a checkmark / strikethrough on the model split line item.
- The deterministic-seed regression test (whether kept or not) is documented in the PR description as evidence of no behavior change.
- The S4 backlog can now consider splitting `src/main.rs` next (the project-refinement doc's other big maintainability call) with the same playbook.
