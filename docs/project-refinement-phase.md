# Project Refinement Phase: Runtime Hardening and Maintainability

## Summary

This phase turns `rusty-gpt` from a working GPT playground into a more reliable, easier-to-extend training and serving stack. The work should preserve the four-model teaching progression while improving MiniGPT runtime ergonomics, API/UI contracts, checkpoint safety, and maintainability.

## Workstreams

### Runtime and Product Capabilities

- Add first-class training lifecycle support beyond CLI logs: start/status/result endpoints, persisted run metadata, and UI progress integration.
- Improve checkpoint/tokenizer ergonomics with explicit compatibility checks, clearer metadata display, and safer latest-checkpoint selection.
- Expand generation controls across API and UI, including temperature, max tokens, `top_k`, model info, and user-visible validation errors.
- Add a lightweight model/corpus evaluation workflow with repeatable benchmarks, validation perplexity, and saved benchmark artifacts.
- Improve release usability with CPU/CUDA artifact notes, release smoke checks, and clearer packaged API/UI startup documentation.

### Maintainability and Refactoring

- Split `src/main.rs` into focused runtime config, training orchestration, checkpoint resolution, server startup, and CLI dispatch modules.
- Split `src/model/mod.rs` into model definitions, generation, training, metrics, and test-support areas without changing model behavior.
- Extract stable contracts for generation options and checkpoint metadata validation so API, CLI, and tests share the same rules.
- Burn down known clippy debt in the model layer before making CI stricter.
- Keep `mini-gpt-ui/` as a separate API consumer with explicit client-side error handling and project-specific docs.

### Exploration

- Investigate explicit opt-in tokenizer training for MiniGPT while preserving the current no-hidden-fallback invariant.
- Evaluate whether CPU/CUDA backend setup can be separated more cleanly without risking CUDA loads on the CPU default path.
- Explore model quality improvements separately from cleanup work: sampling options, dropout behavior, train/validation split controls, and cached generation performance.
- Explore corpus tooling improvements: richer source filters, dataset cache inspection, deduplication, and reproducibility metadata.
- Review the UI information architecture for separate generation, attention visualization, and training surfaces.

## Acceptance Criteria

- Each backend/API/config refactor keeps `cargo test` green.
- New config, checkpoint, tokenizer compatibility, and API validation behavior has focused test coverage.
- `cargo clippy --all-targets` is run during the phase; existing warnings are tracked separately from new regressions.
- API/UI contract work passes `npm run test:all` in `mini-gpt-ui/` and the full-stack `./scripts/run_e2e_tests.sh` smoke when relevant.
- Runtime or CUDA-adjacent changes preserve the CPU default invariant covered by `tests/default_runtime.rs`.

## Initial Slice

The first implementation slice establishes the phase in repo documentation and hardens the existing UI/API contract by loading `/api/info`, exposing generation controls, and surfacing API validation errors in the React UI.
