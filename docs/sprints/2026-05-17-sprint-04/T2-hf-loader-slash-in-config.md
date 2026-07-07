# S4-T2 — Relax HF loader URI parser to allow `/` in config/split/column values

- **Value:** product
- **Size:** S (< 1 day)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** —
- **Blocks:** v3 corpus build (post-sprint)

## Context

`src/loader/huggingface.rs::HuggingFaceDatasetSpec::parse` validates query-parameter values against a character whitelist that excludes `/`. The HuggingFace datasets Hub uses `data/<lang>` as its canonical config naming convention for `bigcode/the-stack-smol` and related repositories (e.g. `config=data/dockerfile`, `config=data/typescript`). The current validator rejects these URIs at parse time, before any HTTP request is made.

This blocks every `the-stack-smol` config, which is the primary source for diverse Dockerfile and TypeScript/Vite corpora needed for v3. Percent-encoding `%2F` does not help because the parser decodes before validating.

The fix is minimal: one regex or allowlist change in the validator. No other caller logic changes.

## Goal

Allow `/` in the `config`, `split`, and `column` query-parameter values of `hf://` URIs so that `hf://bigcode/the-stack-smol?config=data/dockerfile&split=train&rows=100` passes validation and reaches the HTTP fetch layer.

## Acceptance criteria

- `HuggingFaceDatasetSpec::parse("hf://bigcode/the-stack-smol?config=data/dockerfile&split=train&rows=100")` succeeds and populates `config = "data/dockerfile"`, `split = "train"`, `rows = Some(100)`.
- `HuggingFaceDatasetSpec::parse("hf://bigcode/the-stack-smol?config=data/typescript&split=train&rows=50")` succeeds similarly.
- Existing valid URIs without `/` in values (e.g. `hf://Shuu12121/rust-treesitter-dedupe-filtered-datasetsV2?split=train&rows=50000`) continue to parse correctly.
- The validator still rejects obviously malformed values: empty config, values with `..` (path traversal), values with `?` or `#` (would break URL structure).
- At least two new unit tests are added to `src/loader/huggingface.rs`: one asserting the `data/dockerfile` URI parses successfully, one asserting `..` in a value is still rejected.
- `cargo test` passes.
- CLAUDE.md "Common commands" section gains an example showing the `data/dockerfile` URI form so future operators know the syntax works.

## Implementation notes

- The validator lives in `HuggingFaceDatasetSpec::parse` in `src/loader/huggingface.rs`. The regex pattern for param values is the only thing that needs changing — the HTTP fetch and response parsing are unaffected.
- Allow `/` in config/split/column values. Continue rejecting `..`, `?`, `#`, and null bytes. A simple allowlist extension (`[A-Za-z0-9._/-]`) is sufficient; a denylist approach is also acceptable if the agent prefers it.
- Do NOT relax validation on the dataset name itself (`bigcode/the-stack-smol` — the `owner/repo` component) — that already allows `/` by design.
- The S1-T5 input URI validation task hardened the overall URI parsing surface. Ensure this change does not regress any test added in that task.

## Definition of done

- PR merged.
- `cargo run -- --input "hf://bigcode/the-stack-smol?config=data/dockerfile&split=train&rows=10"` runs without a parse error (network errors are acceptable; validator errors are not).
