# AGENTS.md

## Purpose
This file gives AI coding agents concise, repo-specific guidance for working in `rusty-gpt`.

## What this repo is
- Single-binary Rust crate implementing GPT model training and inference on top of the Burn framework.
- Supports four model variants: `trivial`, `single-attention`, `multi-attention`, and `minigpt`.
- Includes CLI runtime, tokenizer training, corpus collection, HTTP server, and a separate React UI in `mini-gpt-ui/`.

## Primary files to inspect
- `README.md` — user-facing quick start, tooling, and configuration guidance.
- `CLAUDE.md` — architecture and implementation notes for AI agents.
- `src/main.rs` — CLI parsing, runtime dispatch, `--serve`/`--load-*` logic, and backend selection.
- `src/model/mod.rs` — model progression and the four GPT variants.
- `src/server/mod.rs` — API routes: `POST /api/generate` and `GET /api/info`.
- `src/tokenizer/` — char-level tokenizer and BPE tokenizer trainer/load logic.
- `tests/default_runtime.rs` — ensures default CPU execution does not load CUDA.

## Build and test commands
Run from the repo root.

- `cargo build`
- `cargo test`
- `cargo clippy --all-targets`
- `cargo fmt --all -- --check`
- `cargo run` for the default CPU demo
- `./scripts/run_e2e_tests.sh` for the full-stack UI/API test

## Important repo-specific constraints
- `MiniGPT` depends on `checkpoints/tokenizer.json`; there is no auto-train fallback.
- `--checkpoint` paths are passed without the `.mpk` extension.
- `--load-checkpoint` and `--load-latest-checkpoint` are mutually exclusive.
- `--interactive-generate` only works with `--backend cpu` and `--model minigpt`.
- `tests/default_runtime.rs` asserts that the CPU default path must not reference CUDA or `libcuda`.
- CUDA support is opt-in via `--features cuda`; CPU-only CI must not rely on CUDA code.
- `data-secret/` is gitignored; use it only for private corpora.
- Agents should use `git worktree` by default when making changes to this repository, keeping the main workspace stable.

## Guidance for agents
- Preserve the existing architecture in `CLAUDE.md` and link to it rather than duplicating it.
- Prefer small, testable changes; add or update tests for runtime behavior and CLI invariants.
- When changing MiniGPT or tokenizer behavior, ensure checkpoint metadata and tokenizer shape rules are respected.
- For API-related work, remember the server routes are nested under `/api`.
- For React/UI work, treat `mini-gpt-ui/` as a separate consumer; focus on Rust API behavior unless asked to change the frontend.

## Recommended normal workflow
1. Read `README.md` and `CLAUDE.md` first.
2. Use `cargo test` after code changes.
3. Run `cargo run` or `./scripts/run_e2e_tests.sh` for end-to-end validation when touching model generation or server behavior.
4. Avoid making CUDA-dependent changes without verifying the CPU build path still works.

## Links
- [README.md](README.md)
- [CLAUDE.md](CLAUDE.md)
