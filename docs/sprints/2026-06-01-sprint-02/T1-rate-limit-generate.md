# S2-T1 — Configurable rate limiting on `POST /api/generate`

- **Value:** security
- **Size:** M (1–2 days)
- **Suggested agent:** principal-security-engineer
- **Depends on:** —
- **Blocks:** —

## Context

`POST /api/generate` runs synchronous greedy/sampling decoding against a MiniGPT model. Even a small model on CPU can take seconds per request; an unauthenticated localhost endpoint with no rate limit is trivially turned into a CPU-pegging attack vector by anyone who can reach the port (LAN, dev-container port-forwards, accidental `0.0.0.0` bind). With the compose stack and dev-container work expanding the exposure surface, a basic in-process rate limit is the cheapest meaningful defense.

## Goal

Add an in-process rate limiter to `POST /api/generate` with configurable thresholds, returning HTTP 429 with a `Retry-After` header when exceeded.

## Acceptance criteria

- New CLI flags / env vars: `--rate-limit-rps <N>` / `RUSTY_GPT_RATE_LIMIT_RPS` (default `5`), `--rate-limit-burst <N>` / `RUSTY_GPT_RATE_LIMIT_BURST` (default `10`). Setting `--rate-limit-rps 0` disables the limiter entirely (back-compat / opt-out).
- Limiter scope is **per peer IP** for `POST /api/generate`. `GET /api/info` and `GET /api/health` are not rate-limited (or limited far more permissively).
- A token-bucket implementation backed by `tower-governor` or an equivalent `tower` middleware crate is preferred over hand-rolled logic — avoid reimplementing the wheel for security-critical code.
- HTTP 429 responses include `Retry-After` (seconds) and a JSON body `{"error": "rate_limited", "retry_after_seconds": N}`.
- An integration test fires `burst + 5` requests in a tight loop against a test server and asserts exactly `burst` succeed (200/400) and the rest return 429.
- The rate limiter does **not** count requests rejected for prompt/token validation (S2-T2) against the bucket — only successfully-validated requests consume capacity. (Implementation: middleware order matters; validation runs before the limiter, or the limiter counts only post-validation hits.)
- README and CLAUDE.md "Runtime configuration" table get the two new flags.

## Implementation notes

- `tower-governor` is the lowest-friction choice; it integrates cleanly with `axum::Router` and supports per-IP keying out of the box.
- For tests, build the router with `Router::with_state` then drive requests via `tower::ServiceExt::oneshot` rather than spinning up a real TCP listener.
- Peer IP extraction: if the server is ever behind a reverse proxy, `X-Forwarded-For` is untrusted by default. **Do not** parse `X-Forwarded-For` in this task — that's a separate, intentional decision that requires a trusted-proxy config.
- Document explicitly: the limiter is per-process; if the compose stack scales to N replicas, the effective limit is `N × rps`. That's acceptable for S2.

## Definition of done

- PR merged.
- A `curl`-driven 429 reproduction is recorded in the development runbook or `docs/configuration.md`.
- Sprint 03's `/api/train` task (S3-T1) inherits the same middleware automatically — confirm the router structure supports that.
