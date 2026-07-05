# Mixture-of-Experts Roadmap

Five two-week sprints that add a **Mixture-of-Experts GPT** (`moe-gpt`) to the
teaching progression, then build out training quality-of-life, serving,
observability, and evaluation around it. Each sprint has its own plan document
with `S<n>-T<m>` task IDs and test-driven acceptance criteria.

## Contents

- [Goals](#goals)
- [Design decisions](#design-decisions)
- [Sprint summary](#sprint-summary)
- [Dependency order](#dependency-order)
- [Definition of done](#definition-of-done)

---

## Goals

1. Extend the teaching progression (`trivial` → `single-attention` →
   `multi-attention` → `minigpt`) with a fifth variant, **`moe-gpt`**, that
   demonstrates sparse expert routing on top of the existing pre-norm
   transformer stack.
2. Reach **full runtime parity with MiniGPT**: training, `compare`, checkpoint
   save/load with the metadata sidecar, `--serve`, `--interactive-generate`,
   cached generation, and CUDA.
3. Use the MoE work to pull forward general improvements the repo wants
   anyway: LR scheduling, resume-from-checkpoint, richer observability,
   benchmarking, and UI visualization.

## Design decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Integration | New `ModelChoice::MoeGpt` variant (`--model moe-gpt`) | Keeps `MiniGpt` untouched as the dense baseline; continues the teaching progression |
| Routing | Linear gate → softmax → top-k selection, renormalized weights | Switch/Mixtral-style; the standard reference design |
| Load balancing | Auxiliary loss = num_experts × Σ(fraction of tokens per expert × mean router probability per expert), weighted into the training loss | Prevents expert collapse without capacity factors or token dropping |
| Capacity / dropping | None — every token is processed by its top-k experts | CPU-first teaching repo; correctness and readability over throughput tricks |
| Code sharing | `Block`'s feed-forward slot becomes a generic parameter (`Block<B, F = Mlp<B>>` bounded by a `FeedForward<B>` trait implemented by `Mlp` and `MoeFeedForward`) — deviation from the originally planned enum, see the S1-T1 note in [sprint-1.md](sprint-1.md) | MoeGpt reuses `Block`, attention, masks, and all six forward variants instead of duplicating ~600 lines; the generic keeps the dense checkpoint record tree unchanged, which an enum module cannot |
| Tokenizer | BPE (`checkpoints/tokenizer.json`), same as MiniGPT | `moe-gpt` is a full GPT, not a char-level toy |
| Framework | Hand-built from `burn::nn::Linear` + softmax/top-k | Burn 0.21 has no built-in MoE module |

## Sprint summary

| Sprint | Theme | Headline deliverable |
| --- | --- | --- |
| [Sprint 1](sprint-1.md) | MoE building blocks | `Router`, `MoeFeedForward`, load-balancing aux loss, `FeedForward` abstraction in `Block` — all unit-tested, no user-visible change |
| [Sprint 2](sprint-2.md) | MoeGpt model, config, training | `--model moe-gpt` trains end-to-end with aux loss, joins `compare`, checkpoints with extended metadata sidecar |
| [Sprint 3](sprint-3.md) | Training QoL + expert observability | LR warmup/cosine, resume-from-checkpoint, per-expert utilization metrics in observability events |
| [Sprint 4](sprint-4.md) | Serving + inference perf | `--serve` and `--interactive-generate` host MoeGpt, cached generation, router stats in the API, CUDA validation |
| [Sprint 5](sprint-5.md) | Evaluation, benchmarking, UI | Dense-vs-MoE matched comparison, generation benchmarks, expert-routing heatmap in `mini-gpt-ui/`, docs + release sweep |

## Dependency order

Sprints are sequential; within a sprint, tasks are ordered so that each builds
on the previous one.

- Sprint 1 is purely internal refactor + new modules; nothing downstream of
  `src/model/` changes.
- Sprint 2 depends on all of Sprint 1 (the model consumes `FeedForward::Moe`
  and the aux-loss channel).
- Sprint 3's expert metrics (S3-T3) depend on Sprint 2's training wiring;
  LR scheduling and resume (S3-T1/T2) are independent and can land first.
- Sprint 4 depends on Sprint 2's checkpoint format for serving loads, and on
  Sprint 1's forward variants for cached generation.
- Sprint 5 depends on Sprint 4's API contract (S4-T4) for the UI work, and on
  everything prior for the evaluation harness.

## Definition of done

Every task lists explicit acceptance criteria; the common bar for a sprint to
close is:

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features cuda -- -D warnings
cargo fmt --all -- --check
```

Additional standing invariants that no sprint may break:

- `tests/default_runtime.rs` — the CPU default run must never load `libcuda`.
- `ModelChoice::Compare` stays a pseudo-variant expanded via
  `comparison_models()`; forward/training match arms on `Compare` remain
  `unreachable!()`.
- `GET /api/health` exposes file basenames only, never absolute paths.
- Legacy checkpoints (dense MiniGPT `.mpk` + sidecar files written before the
  MoE fields existed) keep loading; new sidecar fields use `#[serde(default)]`.
- New flags follow the full plumbing pattern in `src/runtime_config.rs` and are
  documented in `docs/configuration.md` (the authoritative reference) plus the
  CLAUDE.md fast-lookup table.
