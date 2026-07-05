# Sprint 5 — Evaluation, Benchmarking, UI

Goal: answer "was MoE worth it?" with data, expose expert routing visually in
the React UI, and finish the docs/release sweep so `moe-gpt` is a first-class
citizen of the repo.

## Contents

- [S5-T1 — Dense-vs-MoE matched comparison](#s5-t1--dense-vs-moe-matched-comparison)
- [S5-T2 — Generation benchmarks for MoeGpt](#s5-t2--generation-benchmarks-for-moegpt)
- [S5-T3 — Expert-routing visualization in the UI](#s5-t3--expert-routing-visualization-in-the-ui)
- [S5-T4 — Docs sweep](#s5-t4--docs-sweep)
- [S5-T5 — Release candidate + e2e parity](#s5-t5--release-candidate--e2e-parity)
- [Sprint exit checklist](#sprint-exit-checklist)

---

## S5-T1 — Dense-vs-MoE matched comparison

An evaluation recipe (script + doc, not a one-off) comparing MiniGPT and
MoeGpt at matched budgets on the held-out split:

- Two comparisons: **matched active parameters** (MoE with `top_k=2` and
  smaller `d_ff` per expert vs dense with standard `4*d_model`) and
  **matched total parameters** (same total weights; MoE activates fewer per
  token).
- Add a small param/FLOP accounting helper (`MiniGpt::parameter_count()` /
  `MoeGpt::{parameter_count, active_parameter_count}()`) so the configs are
  derived, not hand-waved.
- Driver: extend `scripts/run_e2e_tests.sh`-style tooling or add
  `scripts/run_moe_eval.sh` that trains both configs for a fixed step budget
  on `data/input.txt`, then reports final value loss + perplexity (already in
  `TrainingMetrics` / `TrainingCompleted` events; parse the `--log-format
  json` stream).
- Results land as a table in `docs/release-and-evaluation.md` with the exact
  reproduction commands.

**Files**: `src/model/mod.rs` (param-count accessors),
`scripts/run_moe_eval.sh`, `docs/release-and-evaluation.md`

**Acceptance criteria**
- Unit tests for `parameter_count` / `active_parameter_count` against
  hand-computed values for a tiny config.
- Script runs end-to-end on CPU with a reduced step budget
  (`RUSTY_GPT_TRAIN_STEPS` override) and emits a well-formed results table.
- `docs/release-and-evaluation.md` gains the methodology + a filled-in
  results table from at least one real run.

## S5-T2 — Generation benchmarks for MoeGpt

Extend `--benchmark-generation` (`src/utils/mod.rs`,
`benchmark_generation_cases`) with MoeGpt cases:

- Naive vs cached generation for MoeGpt (mirroring the MiniGPT cases), plus a
  dense-MiniGPT-vs-MoeGpt tokens/sec comparison at matched active parameters.
- If S4-T3's sparse dispatch landed, benchmark dense-compute vs sparse
  dispatch too (guarded by config, both paths kept testable).
- `--benchmark-generation` currently requires `--model minigpt` or `compare`;
  extend the gate to accept `moe-gpt`.
- Document the observed cost profile (router overhead per token, expert
  compute scaling with `top_k`) in `docs/release-and-evaluation.md`.

**Files**: `src/utils/mod.rs`, `src/runtime_config.rs` /
`src/runtime_orchestration.rs` (gate), `docs/release-and-evaluation.md`

**Acceptance criteria**
- `benchmark_generation_cases` returns one result per supported case
  including the new MoE cases (extend the existing
  `benchmark_generation_cases_returns_one_result_per_supported_case` test).
- `cargo run -- --model moe-gpt --benchmark-generation` runs and prints
  timings on CPU.
- Cached-vs-naive MoE outputs asserted identical (correctness inside the
  benchmark harness, same as the dense cases).

## S5-T3 — Expert-routing visualization in the UI

`mini-gpt-ui/` consumes S4-T4's `routing` field from `/api/generate`:

- New panel alongside the attention visualization: a per-layer heatmap of
  token → expert assignment (color = expert id, opacity/annotation = router
  weight; layer selector shared with the attention view).
- Panel renders only when `routing` is present in the response (dense MiniGPT
  servers keep working with no UI change).
- Follow the UI repo's own conventions (its README/toolchain) — this task is
  scoped to the frontend; the API contract is frozen from S4-T4.

**Files**: `mini-gpt-ui/src/...` (per that project's structure)

**Acceptance criteria**
- UI unit/component tests (that project's test runner) for: routing panel
  renders from a fixture response; hidden when `routing` absent; expert
  colors stable across layers.
- `scripts/test_devcontainer_ui.sh` (or the compose `serve` profile) shows
  the heatmap end-to-end against a served moe-gpt checkpoint — recorded as a
  screenshot/manual check in the PR.
- `mini-gpt-ui` lint/build passes.

## S5-T4 — Docs sweep

Bring every document in line with the finished feature set:

- **CLAUDE.md**: model progression gains step 5 (`MoeGpt`), architecture tree
  gains `src/model/moe.rs`, flag table gains the MoE/LR/resume flags, gotchas
  updated (serve parity, MoE checkpoint compatibility fields, aux-loss
  behavior).
- **README.md**: quick-start snippet for training + serving moe-gpt.
- **docs/configuration.md**: authoritative tables complete for all new flags
  (`--moe-*`, `--warmup-steps`, `--lr-schedule`, `--min-learning-rate`,
  `--resume-from`).
- **docs/development-runbook.md**: final pass on the "Training moe-gpt"
  section (S3-T5) + serving/interactive recipes.

**Files**: `CLAUDE.md`, `README.md`, `docs/configuration.md`,
`docs/development-runbook.md`

**Acceptance criteria**
- Every flag in `parse_runtime_config_with_checkpoint` has a row in
  `docs/configuration.md` (spot-check by grepping the parse arms).
- All command snippets in the touched docs execute as written (copy-paste
  check on CPU).
- CLAUDE.md fast-lookup table and `docs/configuration.md` agree (the latter
  is authoritative).

## S5-T5 — Release candidate + e2e parity

- `scripts/build_release_candidate.sh` exercises moe-gpt: build, short train,
  checkpoint, serve, generate — whatever it currently does for MiniGPT.
- Compose profiles: `train` accepts `MODEL=moe-gpt` (or equivalent flag
  pass-through); `serve` can host a moe-gpt checkpoint; devcontainer e2e
  probes (`scripts/test_devcontainer_{generate,server,ui}.sh`) get moe-gpt
  variants or parameters.
- Final CI run on the release branch: full matrix green.

**Files**: `scripts/build_release_candidate.sh`, `compose.yaml` /
`compose.override.yaml`, `scripts/test_devcontainer_*.sh`,
`scripts/run_e2e_tests.sh`

**Acceptance criteria**
- `scripts/run_e2e_tests.sh` includes at least one moe-gpt path and passes.
- `docker compose --profile train` run with the moe-gpt parameterization
  produces a loadable checkpoint; `serve` profile serves it and
  `/api/health` reports `kind: "moe-gpt"`.
- Release-candidate script completes end-to-end on a clean checkout.

## Sprint exit checklist

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features cuda -- -D warnings
cargo fmt --all -- --check
bash scripts/run_e2e_tests.sh

# The roadmap's definition of overall done:
cargo run -- --model compare                                   # five models
cargo run -- --model moe-gpt --benchmark-generation            # timings print
cargo run -- --serve --model moe-gpt --load-latest-checkpoint  # UI shows routing heatmap
```

- `docs/release-and-evaluation.md` contains the dense-vs-MoE results table.
- All five sprint docs' acceptance criteria checked off; roadmap README
  updated with any deviations taken along the way.
