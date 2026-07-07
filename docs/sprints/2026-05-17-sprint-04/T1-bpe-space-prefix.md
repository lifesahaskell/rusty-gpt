# S4-T1 — Fix BPE decoder: GPT-style space-prefix convention

- **Value:** product
- **Size:** L (2–4 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** —
- **Blocks:** v3 training run (post-sprint)

## Context

The v2 training run completed with a final val perplexity of 20.48, but every word boundary disappeared in decoded output. `pub async fn` renders as `pubasyncfn`; `async_trait` renders as `async_trait` (lucky — one token), but multi-word expressions are unreadable. This is the highest-impact blocker: fixing the model's corpus, architecture, or training regime is wasted effort if the decoder discards spaces on the way out.

Root cause is in `src/tokenizer/bpe.rs`. The current BPE trainer merges adjacent bytes without any `Ġ` (U+0120) space-prefix convention. The encoder treats ` fn` and `fn` as different byte sequences that may or may not be merged together, but the decoder has no way to know whether a boundary between two token strings was originally a space. Standard GPT-style BPE (tiktoken, HuggingFace tokenizers) pre-processes the corpus by replacing each leading space with `Ġ` before training, so the presence of `Ġ` in a token is the canonical signal that a space precedes it.

## Goal

Modify the BPE encoder/decoder in `src/tokenizer/bpe.rs` so that:
1. During **tokenizer training** (`BpeTrainer`), the pre-tokenization step replaces each leading space in every word with `Ġ` (U+0120), mirroring the GPT-2 convention.
2. During **encoding**, the same pre-tokenization is applied to input text before BPE merge application.
3. During **decoding**, `Ġ` prefixes are replaced with a literal space, restoring word boundaries.
4. The `train-tokenizer` binary re-trained with these changes produces a `checkpoints/tokenizer.json` that round-trips whitespace correctly.

An alternative approach — switching to the `tokenizers` crate — is acceptable only if the BPE path becomes disproportionately complex. Prefer staying in-crate to avoid a new dependency; the agent should make this call after assessing the diff size.

## Acceptance criteria

- `BpeTokenizer::decode(BpeTokenizer::encode("pub async fn foo"))` returns `"pub async fn foo"` (spaces preserved). Add this as a unit test named `bpe_decode_preserves_word_boundaries`.
- `BpeTokenizer::decode(BpeTokenizer::encode(""))` returns `""`.
- `BpeTokenizer::encode` is still byte-safe — arbitrary UTF-8 input does not panic. Test with a non-ASCII string (e.g. a comment with an accented character).
- The tokenizer JSON format (`checkpoints/tokenizer.json`) remains loadable by the existing `BpeTokenizer::load` path. If the format changes to store `Ġ`-prefixed vocab entries, bump the format version field or add a migration note in the PR description.
- The existing checkpoint `checkpoints/mini_gpt_rdv.mpk` is **invalidated** by this change (different token IDs). The PR description must state this explicitly so no one attempts to load the v2 checkpoint against the new tokenizer.
- The fixture at `tests/fixtures/tokenizer.json` must be regenerated or updated to match the new encoder format so existing unit tests that load it continue to pass.
- `cargo test` passes, including `tests/default_runtime.rs`.
- `cargo clippy --all-targets` introduces no new warnings beyond the pre-existing set in CLAUDE.md.

## Implementation notes

- The conventional pre-tokenization splits on whitespace (or uses a regex like `'s|'t|'re|...` for contractions) and prefixes each non-first piece with `Ġ`. For code corpora the simpler split-on-whitespace approach is fine — contractions are rare.
- `src/tokenizer/bpe.rs` contains `BpeTrainer` (training) and `BpeTokenizer` (encode/decode). Both need changes. `char.rs` is unaffected.
- `RuntimeTokenizer::Char` is also unaffected — only the `Bpe` variant needs the fix.
- The interactive loop (`runtime_orchestration.rs`) and the HTTP `/generate` endpoint (`src/server/mod.rs`) both call `try_encode` — verify that the new pre-tokenization is applied consistently in both paths.
- After this PR merges, the v3 corpus preparation step must re-run `cargo run --bin train-tokenizer` to regenerate `checkpoints/tokenizer.json` before the training run starts.

## Definition of done

- PR merged.
- `bpe_decode_preserves_word_boundaries` test passes and is visible in `cargo test` output.
- The development runbook (`docs/development-runbook.md`) gains one sentence: "After any change to the BPE encoder/decoder, regenerate `checkpoints/tokenizer.json` before starting a training run."
