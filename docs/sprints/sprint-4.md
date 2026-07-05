# Sprint 4 — Serving + Inference Performance

Goal: `moe-gpt` reaches full runtime parity with MiniGPT on the inference
side — served over the HTTP API, usable in `--interactive-generate`, fast via
the cached-generation path — and exposes router statistics through the API so
the UI can visualize expert routing. CUDA training is validated behind the
feature gate.

## Contents

- [S4-T1 — Serve MoeGpt over the HTTP API](#s4-t1--serve-moegpt-over-the-http-api)
- [S4-T2 — Interactive generation for moe-gpt](#s4-t2--interactive-generation-for-moe-gpt)
- [S4-T3 — Cached generation (KV-cache) for MoeGpt](#s4-t3--cached-generation-kv-cache-for-moegpt)
- [S4-T4 — Router statistics in the generate API](#s4-t4--router-statistics-in-the-generate-api)
- [S4-T5 — CUDA validation for MoE](#s4-t5--cuda-validation-for-moe)
- [Sprint exit checklist](#sprint-exit-checklist)

---

## S4-T1 — Serve MoeGpt over the HTTP API

`--serve` currently hosts MiniGPT only. Generalize `ServerState` /
`run_http_server` (`src/server/mod.rs`, `src/runtime_orchestration.rs`) to
hold either model — a `ServedModel<B>` enum (`MiniGpt` | `MoeGpt`) keeps the
handler code monomorphic and avoids trait objects:

- `--serve --model moe-gpt` loads a MoeGpt (fresh template,
  `--load-checkpoint`, or `--load-latest-checkpoint`), using the strict
  metadata loader. The `--load-latest-checkpoint` scan must respect the
  sidecar's model shape: a moe-gpt serve refuses a dense checkpoint (and vice
  versa) with a clear error, rather than a record-decode panic.
- `GET /api/info` gains `num_experts` / `moe_top_k` fields (0 or omitted for
  dense); `GET /api/health`'s `model` block gains `kind: "moe-gpt"` plus the
  same fields. The basename-only information-disclosure boundary and the
  `health_never_exposes_absolute_path` test must keep passing.
- Rate limiting, body-size limits, and prompt/max_tokens validation are
  shared — no divergence between the two served models.

**Files**: `src/server/mod.rs`, `src/runtime_orchestration.rs`,
`src/runtime_config.rs` (allow `--serve` with `--model moe-gpt`)

**Acceptance criteria**
- Server unit tests (the existing tower/oneshot style) pass for a MoeGpt
  state: `/api/generate` returns tokens, `/api/info` and `/api/health` report
  the MoE shape fields, dense responses omit or zero them (schema test both
  ways).
- Wrong-model checkpoint load fails with an actionable error naming both
  kinds.
- Existing MiniGPT server tests unchanged.

## S4-T2 — Interactive generation for moe-gpt

Extend `--interactive-generate` (`src/runtime_orchestration.rs`) to accept
`--model moe-gpt`, keeping the existing constraints (CPU backend only; other
models still rejected with the current error message shape).

**Files**: `src/runtime_orchestration.rs`, `src/runtime_config.rs`

**Acceptance criteria**
- Config test: `--interactive-generate --model moe-gpt --backend cpu` is
  accepted; `--backend cuda` still rejected; `--model trivial` still rejected.
- Manual smoke (documented in PR): interactive session generates from a
  trained moe-gpt checkpoint.

## S4-T3 — Cached generation (KV-cache) for MoeGpt

Bring `generate_cached` parity:

- `Block::forward_with_cache` already threads KV state; the MoE feed-forward
  is stateless per token, so the cache path needs only the `FeedForward`
  dispatch from S1-T5 (aux discarded). Implement
  `MoeGpt::generate` / `generate_cached` / `_with_options` variants mirroring
  `MiniGpt` (greedy + temperature/top-k sampling via
  `model/generation.rs`), with context cropped to `block_size`.
- This is also the sprint to replace the S1 dense-compute expert dispatch
  with sparse dispatch (only run an expert on its assigned tokens) **if**
  profiling shows single-token decode is dominated by expert compute; keep it
  a separate commit so the correctness-vs-speed change is bisectable.

**Files**: `src/model/mod.rs` (or `moe_gpt.rs`), `src/model/moe.rs`,
`src/model/generation.rs` (only if shared helpers need widening)

**Acceptance criteria**
- Equivalence test: `generate_cached` produces the identical token sequence
  to naive `generate` for the same prompt/weights (greedy), matching the
  existing MiniGPT parity test pattern.
- Sampling variants respect `GenerationOptions` (temperature > 0, top_k)
  exactly as MiniGPT's do.
- If sparse dispatch lands: dense-vs-sparse dispatch equivalence test (same
  outputs within tolerance).

## S4-T4 — Router statistics in the generate API

The API contract the UI (S5-T3) consumes, alongside the existing
`AttentionData`:

- `MoeGpt::forward_with_attention` grows a router-introspection sibling
  (`forward_with_introspection`) returning attention weights **and** per-layer
  router data for the final forward pass.
- `GenerateResponse` (`src/server/mod.rs`) gains an optional field:

```jsonc
"routing": [                      // omitted entirely for dense models
  {
    "layer": 0,
    "experts": [[1, 3], ...],     // per generated-context token: top-k expert ids
    "weights": [[0.7, 0.3], ...]  // matching renormalized router weights
  }
]
```

- Size guard: like attention data, routing data scales with tokens × layers;
  reuse whatever truncation/limit policy the attention payload applies (and
  if none exists, cap both consistently under this task).

**Files**: `src/model/mod.rs` / `moe_gpt.rs`, `src/server/mod.rs`

**Acceptance criteria**
- Server test: `/api/generate` against a MoeGpt state returns `routing` with
  one entry per layer, expert ids `< num_experts`, weights summing to ~1 per
  token; dense MiniGPT responses have no `routing` key (serde skip).
- Introspection forward is inference-only (no autodiff requirement) and does
  not disturb `generate_cached` outputs.
- OpenAPI-ish doc: response shape documented in `docs/configuration.md` or a
  server section of the runbook (wherever `/api/generate`'s schema currently
  lives; add it if undocumented).

## S4-T5 — CUDA validation for MoE

- Audit new MoE code for CPU-only assumptions (host-side loops over tensor
  data, `into_data()` in hot paths); everything must compile and run under
  `--features cuda` with `Cuda` backend.
- `cargo check --features cuda` and the strict CUDA clippy pass are already
  CI gates; add a feature-gated smoke test or a documented manual run:
  `cargo run --release --features cuda -- --backend cuda --model moe-gpt`
  for a short step count on a CUDA host.
- `tests/default_runtime.rs` must keep passing — no CUDA types reachable from
  the CPU default path via the new MoE modules.

**Files**: `src/model/moe.rs`, `scripts/test_cuda_passthrough.sh` /
`docs/development-runbook.md` (manual-validation recipe)

**Acceptance criteria**
- `cargo clippy --all-targets --features cuda -- -D warnings` clean.
- `tests/default_runtime.rs` green (no `libcuda` in default run output).
- Documented CUDA training run of moe-gpt completes with decreasing loss
  (runbook records the command + expected output shape; CI cannot run CUDA).

## Sprint exit checklist

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features cuda -- -D warnings
cargo fmt --all -- --check

# Serve a trained moe-gpt and poke the API
cargo run -- --serve --model moe-gpt --load-latest-checkpoint
curl -s localhost:8787/api/health | jq .model
curl -s localhost:8787/api/info | jq .
curl -s localhost:8787/api/generate -d '{"prompt":"ROMEO:","max_tokens":16,"temperature":0.8}' \
  -H 'content-type: application/json' | jq '.routing[0]'
```

- CLAUDE.md gotchas updated: "`--serve` only hosts MiniGpt" becomes
  "`--serve` hosts MiniGpt and MoeGpt; the three char-level variants are
  training-only".
