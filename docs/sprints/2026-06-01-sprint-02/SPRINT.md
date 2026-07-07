# Sprint 02 — Secure and observe the Axum server

- **Sprint window:** 2026-06-01 → 2026-06-12 (2 weeks)
- **Sprint ID:** `2026-06-01-sprint-02`
- **Theme:** Take the `/api` surface from "works on localhost" to "safe to point a friend at." Add the operational hooks needed to actually run it (health, structured limits) and lock CI down so the maintainability debt from Sprint 01 stays paid.

## Sprint goal

Harden `POST /api/generate` against the obvious abuse vectors (unbounded prompts, runaway `max_tokens`, request flooding), prevent the obvious filesystem foot-gun (path traversal via `--checkpoint`), expose a real `/api/health` endpoint so the dev-container and compose stack can run liveness probes, and flip CI to `cargo clippy -D warnings` now that the S1 debt is gone.

## Value distribution (this sprint)

- security: 3 (T1, T2, T3)
- maintainability: 2 (T4, T5)
- product: 0

This sprint is intentionally security-heavy. The recent compose / dev-container / NVIDIA-toolkit commits indicate the server is being prepared for wider exposure; closing the obvious holes before Sprint 03 adds remote-triggered training endpoints is deliberate.

## Task list

| ID | Title | Value | Size | Suggested agent |
|---|---|---|---|---|
| [T1](T1-rate-limit-generate.md) | Configurable rate limiting on `POST /api/generate` | security | M | principal-security-engineer |
| [T2](T2-prompt-token-caps.md) | Enforce max prompt length and max `max_tokens` caps | security | S | principal-security-engineer |
| [T3](T3-checkpoint-path-confinement.md) | Confine `--checkpoint` paths to `checkpoints/` (prevent traversal) | security | S | principal-security-engineer |
| [T4](T4-api-health.md) | `GET /api/health` reporting checkpoint, model shape, uptime | maintainability | S | fullstack-react-rust-engineer |
| [T5](T5-ci-clippy-strict.md) | Tighten CI to `cargo clippy -D warnings` | maintainability | S | senior-devops-engineer |

## Dependencies

- **T5 depends on S1-T4** (the clippy burn-down). Sprint 01 must close before T5 can ship.
- **T1, T2, T3** are independent of each other and can run in parallel, but bundling T1 and T2 into one review pass is sensible since both touch the `/api/generate` handler.
- **T4** is independent and a good warm-up task; ship it first to validate the test harness for the API surface.

## Risks

- Rate-limit choice (in-memory vs. distributed) affects how the dev-container scales. For S2, in-memory per-process is the right answer — distributed Redis-backed is a Sprint 03+ decision if compose ever runs multiple replicas.
- Path-confinement (T3) must not break the existing `--load-latest-checkpoint` flow that scans `checkpoints/` already — coordinate test data.
- Strict clippy in CI (T5) will block PRs that introduce new lints. Communicate the flip to the team (or in CHANGELOG) so nobody is surprised.
- `/api/health` payload (T4) is the right place to expose model shape and tokenizer sha — be careful **not** to leak the absolute checkpoint path (information disclosure). Report basename + sha256 only.

## Exit criteria

- All five PRs merged.
- `curl -sS localhost:8787/api/health` returns 200 with model shape, tokenizer sha256, and uptime fields documented in CLAUDE.md.
- A scripted load test (`hey -n 1000 -c 50 ...`) against `/api/generate` triggers the rate limiter within the documented threshold and returns 429.
- Sending a 1 MB prompt body or `max_tokens=10000` returns 400 with a clear error.
- An attempt to pass `--checkpoint ../../etc/passwd` exits 1 at parse time with a clear "outside checkpoints/" message.
- `cargo clippy --all-targets -- -D warnings` is the new CI bar; one prior-green PR has been re-run to confirm it stays green.
- `tests/default_runtime.rs` still passes.

## Out of scope (parking lot for Sprint 03+)

- Auth / API keys on `/api/generate` — defer until product needs (S3 likely doesn't need it for localhost training UI, but flag for S4).
- Distributed / Redis-backed rate limiting — current single-process design does not need it.
- Structured logging / tracing overhaul — a separate "observability sprint" if appetite emerges.
- TLS termination — handled at the compose/ingress layer, not in `axum::serve`.
