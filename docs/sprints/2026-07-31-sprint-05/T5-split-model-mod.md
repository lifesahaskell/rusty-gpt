# S5-T5 — Finish the `src/model/mod.rs` split into submodules

- **Value:** maintainability
- **Size:** M (1–2 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** none
- **Blocks:** none (but land before/rebase against T1 — see SPRINT.md Dependencies)

## Context

Sprint 03's T5 pulled `moe.rs`, `persistence.rs`, `training.rs`, and `generation.rs` out of `src/model/mod.rs`, but its own exit criteria ("split into at least three submodules") technically passed while leaving `mod.rs` itself at 1966 lines. What's still in there: `MiniGptConfig`/`MoeGptConfig`, `LayerCache`/`KvCache`, all five model structs (`TrivialModel`, `SingleAttentionModel`, `MultiAttentionModel`, `MiniGpt`, `MoeGpt`), `SingleHeadAttention`, `MultiHeadAttention`, `Mlp`/`FeedForward`/`Block`, `MoeAttentionRouting`, and a ~726-line `mod tests` block (lines 1240–1966).

## Goal

`mod.rs` becomes a thin module root: re-exports plus whatever top-level glue doesn't belong in a more specific file. No behavior change.

## Acceptance criteria

- `src/model/attention.rs` — `SingleHeadAttention`, `MultiHeadAttention`, `LayerCache`, `KvCache`.
- `src/model/block.rs` — `Mlp`, `FeedForward`, `Block`.
- `src/model/definitions.rs` — `MiniGptConfig`, `MoeGptConfig`, `TrivialModel`, `SingleAttentionModel`, `MultiAttentionModel`, `MiniGpt`, `MoeGpt`, `MoeAttentionRouting`.
- Tests move with the code they test, not into one dumping-ground file — e.g. attention tests land in `attention.rs`'s own `#[cfg(test)] mod tests`, block tests in `block.rs`, etc. If a test genuinely spans multiple modules (e.g. a full-model forward-pass test), it goes in `definitions.rs` next to the model it exercises.
- `mod.rs` ends up under ~400 lines: `pub mod` declarations, `pub use` re-exports so `crate::model::MiniGpt` etc. keep working unchanged, and nothing else.
- No public API changes — every existing `use rusty_gpt::model::X` import across `src/`, `tests/`, and doc comments keeps resolving without edits. Verify with `cargo build` and `cargo test` rather than assuming.
- `cargo clippy --all-targets -- -D warnings` and `cargo clippy --all-targets --features cuda -- -D warnings` both pass — this is a pure move, it shouldn't introduce new lints, but module boundaries sometimes surface `unused import` or visibility warnings that need `pub(crate)` adjustments.
- `cargo fmt --all -- --check` passes.

## Implementation notes

- Do this as a mechanical move (cut/paste + fix imports), not a rewrite. Resist the urge to "improve" anything while in there — that's a different PR.
- Watch for circular-looking imports between `attention.rs`, `block.rs`, and `definitions.rs` (e.g. `Block` uses `MultiHeadAttention` and `FeedForward`) — these are fine as long as they go through `pub use` at the `mod.rs` level or direct `super::` / `crate::model::` paths; Rust doesn't care about file boundaries the way it cares about crate boundaries.
- `git mv` where possible before editing, so the diff shows renames cleanly instead of full-file deletes+adds.

## Definition of done

- PR merged. `wc -l src/model/mod.rs` is under 400.
