# Sprint 04 — v3 Training Pipeline: Readable Output and Corpus Coverage

- **Sprint window:** 2026-05-17 → 2026-05-30 (2 weeks)
- **Sprint ID:** `2026-05-17-sprint-04`
- **Theme:** The v2 training run completed but produced unreadable output. Fix the BPE decoder so served text is human-readable, unlock the corpus sources needed to cover Dockerfile and Vite patterns, and eliminate the server startup trap that cost a debug cycle. The v3 training run itself is out of scope — this sprint clears every technical blocker so the run can start cleanly.

## Sprint goal

By the end of this sprint a user can `POST /api/generate` and read syntactically coherent code (spaces between tokens), `bigcode/the-stack-smol?config=data/dockerfile` loads without error, and `--serve` against a non-default-shape checkpoint no longer requires manually restating every hyperparameter flag.

## Value distribution (this sprint)

- product: 3 (T1, T2, T3)
- maintainability: 0
- security: 0

Sprint 04 is narrowly focused on unblocking the v3 training run. No new product features; no API surface changes beyond T3.

## Task list

| ID | Title | Value | Size | Suggested agent |
|---|---|---|---|---|
| [T1](T1-bpe-space-prefix.md) | Fix BPE decoder: GPT-style space-prefix convention so decoded output is readable | product | L | fullstack-react-rust-engineer |
| [T2](T2-hf-loader-slash-in-config.md) | Relax HF loader URI parser to allow `/` in `config`/`split`/`column` values | product | S | fullstack-react-rust-engineer |
| [T3](T3-serve-shape-from-sidecar.md) | Derive model-shape flags at `--serve` time from the checkpoint's metadata sidecar | product | M | fullstack-react-rust-engineer |

## Dependencies

- **T1** has no code dependency on T2 or T3 and can start on day 1. It is the highest-risk ticket (decoder rework touches `src/tokenizer/bpe.rs` and may require re-training the tokenizer artifact) so it should land and be reviewed before sprint end.
- **T2** is a one-file regex/validation change in `src/loader/huggingface.rs`. It is independently runnable in parallel with T1. It is a prerequisite for the v3 corpus build (the stack-smol Dockerfile and TypeScript configs), but the corpus build is out of scope for this sprint.
- **T3** depends on the metadata sidecar structure defined in `src/model/persistence.rs`, which already exists. No dependency on T1 or T2.
- There are no sequential dependencies within the sprint. All three tickets can run in parallel once T1 is scoped.

## Risks

- **T1 is L-sized and may require a tokenizer retrain.** If encoding a space-prefix into the BPE vocabulary changes the token IDs, the existing `checkpoints/tokenizer.json` artifact is invalidated and any saved model checkpoint that used the old tokenizer cannot be loaded against the new one (the strict loader's hash check will reject it). This is acceptable — v2 checkpoints are already known-bad output-wise — but must be documented in the PR and flagged so no one wastes time trying to reuse `mini_gpt_rdv.mpk` post-merge.
- **T3 changes server startup behavior.** If the sidecar is missing (legacy checkpoint), the server must have a clear fallback path. The existing `load_model_with_metadata_validation` lenient loader is the right starting point; do not use the strict loader here or `--serve` will break against any checkpoint saved before S1-T3.

## Out of scope

- **The v3 training run** — this sprint is purely unblocking. The run starts after T1 lands and the tokenizer is retrained.
- **`--resume-checkpoint` (S3-T4)** — the durability memory notes this is the remaining gap on the Sprint 3 backlog. Given that a v3 run is expected to take ~5 hours on an RTX 4060 Ti and T1/T2/T3 already fill the sprint, resume support is deferred to Sprint 05. The cost of a mid-run crash is one 5-hour re-run; that is tolerable. If the run timeline extends past 8 hours, pull T4 forward.
- **BPE encoder performance** — the single-threaded O(N×M) encoder takes 70+ minutes on a 94 MB corpus. For v3 the corpus will stay at ~45-50 MB, so this remains an acceptable slow start. A multi-threaded encoder or progress event is a Sprint 05 quality-of-life item.
- **Corpus degeneration / top-k tuning** — the v2 model's repetition loops (`&auth_user_id` × 8) are corpus-diversity problems, not sampler-parameter problems. Fixing them requires a richer corpus (more distinct Dockerfiles via the-stack-smol once T2 lands, more Vite configs). That corpus curation is pre-training prep, not in-sprint work.
- **`codeparrot/github-code-clean`** — this dataset returns HTTP 501 from HF datasets-server ("Job manager killed"). Do not retry; it is too large for on-demand indexing. Document the exclusion in the v3 corpus prep notes.
- **`mini-gpt-ui/` React changes** — the UI already consumes `/api/generate` text. Once T1 fixes decoding, the UI improvement is automatic. No separate frontend ticket needed.
- **CI / devops changes** — no `.github/workflows/ci.yml` changes anticipated. No `senior-devops-engineer` tasks this sprint.

## Exit criteria (definition of done for the sprint)

1. `POST /api/generate` with prompt `"pub async fn"` returns text where tokens are separated by spaces — `"pub async fn foo"` not `"pubasyncfnfoo"`. Verified by a unit test in `src/tokenizer/bpe.rs` that encodes then decodes a whitespace-containing string and asserts round-trip fidelity.
2. `cargo run -- --input "hf://bigcode/the-stack-smol?config=data/dockerfile&split=train&rows=100"` resolves the URI without a parse error and begins fetching (HTTP 200 or a clear network error, not a validator rejection).
3. `cargo run -- --serve --load-latest-checkpoint` against `checkpoints/mini_gpt_rdv.mpk` (which has a `.metadata.json` sidecar) starts the server with the correct `embed_dim`, `num_heads`, `num_layers`, and `block_size` read from the sidecar — without the user passing those flags manually.
4. `cargo test` passes on all three ticket branches before merge.
5. `cargo clippy --all-targets` introduces no new warnings beyond the pre-existing set documented in CLAUDE.md.
6. `tests/default_runtime.rs` still passes (CPU default does not load `libcuda`).

## Commit conventions

Do not add `Co-Authored-By: Claude` trailers to commit messages.
