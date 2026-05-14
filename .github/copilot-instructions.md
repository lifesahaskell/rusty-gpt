# Copilot Instructions

This repository uses `AGENTS.md` and `CLAUDE.md` as the primary AI agent guidance documents.

## Key guidance
- Read `AGENTS.md` first for concise agent-specific instructions.
- Use `CLAUDE.md` for the deeper architecture and implementation context.
- Prefer small, testable Rust changes and preserve the existing model progression and CLI invariants.
- Validate changes with `cargo test`, `cargo fmt --all -- --check`, and `cargo clippy --all-targets`.

## Important repo-specific rules
- `MiniGPT` requires `checkpoints/tokenizer.json`; there is no auto-train fallback.
- `--checkpoint` paths must be passed without the `.mpk` extension.
- `--load-checkpoint` and `--load-latest-checkpoint` are mutually exclusive.
- `--interactive-generate` only works with `--backend cpu` and `--model minigpt`.
- Server routes are nested under `/api`: use `/api/generate` and `/api/info`.
- The CPU default path must not load CUDA; `tests/default_runtime.rs` enforces this.
- When editing the repo, use `git worktree` by default so agent changes remain isolated from the main workspace.

## Workflow
1. Inspect `README.md`, then `CLAUDE.md`, then `AGENTS.md`.
2. Make focused changes and add tests where appropriate.
3. Keep CUDA-specific code gated behind `#[cfg(feature = "cuda")]`.
4. Treat `mini-gpt-ui/` as a separate consumer unless asked to modify the UI.

## References
- [AGENTS.md](../AGENTS.md)
- [CLAUDE.md](../CLAUDE.md)
- [README.md](../README.md)
