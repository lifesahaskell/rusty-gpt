# S1-T1 — Commit and stabilize the in-flight runtime refactor

- **Value:** maintainability
- **Size:** M (1–2 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** —
- **Blocks:** T2, T3 (both touch overlapping files)

## Context

At sprint kickoff, `main` has uncommitted modifications across:

- `src/main.rs`
- `src/model/generation.rs`
- `src/model/mod.rs`
- `src/model/persistence.rs`
- `src/runtime_assets.rs`
- `src/runtime_config.rs`
- `src/utils/mod.rs`

This spans CLI parsing, model definitions, generation, persistence, runtime asset loading, runtime config, and shared helpers — i.e. the entire runtime spine. Until this is committed, every downstream task pays a rebase tax.

## Goal

Land the in-flight changes on `main` as a single coherent commit (or a tightly scoped series), with passing tests and a clear commit message describing the intent.

## Acceptance criteria

- `git status` is clean on `main` immediately after the merge.
- The change is described by a clear commit message; if the diff turns out to encode more than one intent, split into separate commits (one per intent) before merging.
- `cargo build`, `cargo test`, `cargo test --test default_runtime`, and `cargo build --features cuda` (verify gate) all pass.
- No new clippy warnings are introduced (the pre-existing baseline is the bar; reducing it is bonus).
- `tests/default_runtime.rs` still passes — the CPU default path must not load `libcuda`.
- The four-model teaching progression (`trivial`, `single-attention`, `multi-attention`, `minigpt`) still trains end-to-end with `RUSTY_GPT_TRAIN_STEPS=1`.

## Implementation notes

- Run `git diff` first and write a one-paragraph summary of what the in-flight change is *trying* to accomplish before touching any file. If you can't summarize it, that's the signal to split.
- Confirm `Hyperparameters::from_env_and_overrides → validate()` still rejects invalid combinations (e.g. `embed_dim` not divisible by `num_heads`).
- If `src/model/generation.rs` is new (the file is in the diff but the CLAUDE.md still references inlined generation in `src/model/mod.rs`), confirm `MiniGpt::generate` / `generate_cached` are re-exported from `src/lib.rs` so the bins and tests still compile.

## Definition of done

- PR merged to `main`.
- Sprint backlog updated: T2 and T3 are unblocked.
- Commit message follows project convention (no `Co-Authored-By: Claude` trailer per user memory).
