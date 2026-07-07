# S2-T2 — Enforce max prompt length and max `max_tokens` caps

- **Value:** security
- **Size:** S (half day)
- **Suggested agent:** principal-security-engineer
- **Depends on:** —
- **Blocks:** —

## Context

`POST /api/generate` accepts a `prompt` string and a `max_tokens` integer with only the temperature > 0 and `top_k != 0` checks in place. Without explicit caps, a single request can:

- Allocate a multi-megabyte token tensor from a huge prompt (memory pressure).
- Request `max_tokens = 1_000_000_000` and tie up the request handler indefinitely (CPU exhaustion, blocks other requests behind the rate limiter's burst window).

Both are routine denial-of-service vectors that are also routine input-validation hygiene.

## Goal

Enforce documented, configurable hard caps on prompt size and generation length at the API edge, returning HTTP 400 with a clear error before any model work happens.

## Acceptance criteria

- New CLI flags / env vars: `--max-prompt-bytes <N>` / `RUSTY_GPT_MAX_PROMPT_BYTES` (default `8192`), `--max-output-tokens <N>` / `RUSTY_GPT_MAX_OUTPUT_TOKENS` (default `512`).
- Prompt is rejected (400) if its UTF-8 byte length exceeds `max_prompt_bytes`. Error body: `{"error": "prompt_too_large", "max_bytes": N, "actual_bytes": M}`.
- `max_tokens` is rejected (400) if it exceeds `max_output_tokens` or is `<= 0`. Error body: `{"error": "max_tokens_out_of_range", "max_allowed": N, "requested": M}`.
- An Axum-level body size limit (`tower_http::limit::RequestBodyLimitLayer`) caps the **request body** at `max_prompt_bytes + 4 KiB` (small headroom for JSON overhead and other fields) to prevent oversized payloads from being parsed at all.
- The 400 errors include enough detail for the React UI to show a useful message without leaking implementation details (no stack traces, no internal paths).
- Integration tests cover: prompt one byte over the limit (400), `max_tokens` one over the limit (400), zero `max_tokens` (400), prompt exactly at limit (200), `max_tokens` exactly at limit (200).
- Existing happy-path tests still pass with default caps.

## Implementation notes

- Validation should run **before** the rate-limit middleware so that an invalid request does not consume a rate-limit token (see S2-T1 implementation note on middleware order).
- The body-size limit is a defense-in-depth layer — even if the prompt-length check has a bug, the request can't be larger than `max_prompt_bytes + overhead`.
- Tokenizer encoding of the prompt should happen **after** the byte-length check, not before. Encoding a 100 MB prompt just to count tokens defeats the purpose.

## Definition of done

- PR merged.
- README "API" section (or `docs/configuration.md`) documents the limits and the corresponding error codes.
- A note in CLAUDE.md "Gotchas" reminds future authors that any new `/api/*` endpoint must opt in to the body-size middleware explicitly.
