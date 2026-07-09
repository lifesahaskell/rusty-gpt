# Research playground roadmap

Source map: [Map research-playground roadmap for ML systems learning](https://github.com/lifesahaskell/rusty-gpt/issues/21)

Goal: learn ML systems on one local 4060 Ti-class NVIDIA GPU. This roadmap is planning only; each experiment is a small, rerunnable 3-run ablation.

## Fixed protocol

Use the protocol resolved in issue 22:

- Run shape: baseline + 2 variants.
- Budget: at most 6 GPU-hours per experiment.
- Record: git commit, command/config, dataset/tokenizer, seed when available, wall-clock, final train loss, final validation loss/perplexity, tokens/sec, peak VRAM, and one-line mechanism takeaway.
- Artifact: one notebook/table per experiment, with mechanism-specific plots.
- Learning bar: explain what changed, why it should matter, what happened, code paths touched, when it helps/hurts, and how to diagnose failure.

## Baseline data and tokenizer policy

Default baseline:

- Dataset: committed Shakespeare corpus, `data/input.txt`.
- Split: existing train/value split from `runtime_training`; do not reshuffle policy per experiment unless the experiment is about sampling.
- Tokenizer: one checked-in or locally pinned BPE tokenizer at `checkpoints/tokenizer.json`, trained from the same baseline corpus.
- Model: MiniGPT on CUDA, with one lane-baseline config captured in a config file or exact command before variants run.

Rules:

1. Reuse the baseline tokenizer for training-loop and architecture-internals experiments.
2. Retrain tokenizers only for tokenizer/data-system experiments.
3. If a tokenizer changes, record its vocab size and SHA-256, and do not compare raw loss as a pure architecture result.
4. If the dataset changes, make that the named variable under test; do not also change model shape or tokenizer unless the card explicitly says so.
5. Keep all 3 runs in an experiment on the same corpus slice size unless slice size is the variable.

## Lane choices

### Architecture-internals lane

Order:

1. **Attention-head shape** — no new model code; teaches `embed_dim`, `num_heads`, `head_dim`, parallel attention views, and CUDA shape/throughput tradeoffs.
2. **RMSNorm** — first small architecture implementation; teaches residual-stream scale control and normalization cost/stability.
3. **MLP gating** — replaces the feed-forward sublayer; teaches capacity, activation/gating mechanics, parameter fairness, and throughput tradeoffs.
4. **RoPE** — deeper attention/position experiment; teaches where position enters Q/K geometry and why context-length behavior changes.

Skip for this roadmap: broad architecture search, MoE, large benchmark suites, and anything needing multi-GPU.

### Training-loop and data-systems lane

Order:

1. **Checkpoint/eval cadence** — cheapest systems shakedown; teaches observability overhead, recovery tradeoffs, artifact management, and whether the notebook/table workflow works.
2. **Learning-rate schedule** — small training-loop change; teaches warmup, decay, optimizer step mechanics, and loss-curve diagnosis.
3. **Tokenizer vocabulary size** — data-systems core; teaches subword granularity, sequence pressure, vocab/head size, tokenizer hash compatibility, and throughput/loss interpretation.
4. **Batch sampling strategy** — loader-focused; teaches stochastic windows, epoch semantics, cache locality, leakage risks, and loss smoothness.

Defer gradient accumulation until Burn API friction is known; add it only if direct batch-size experiments hit VRAM limits and the mechanism remains worth learning.

## Three-month sequence

Assume six two-week milestones. Each milestone should leave one notebook and one short experiment card update.

### Month 1 — prove the protocol

#### Milestone 1: lane baseline + checkpoint/eval cadence

Experiment card:

- Question: how much do eval/checkpoint choices perturb throughput and observability?
- Runs: sparse cadence baseline, medium cadence, frequent cadence.
- Primary code paths: `src/runtime_training.rs`, checkpoint retention, observability events.
- Plots: step time/tokens-sec, validation-loss visibility, checkpoint count/size.
- Stop when: a rerunnable command/config exists and the summary table is filled.

#### Milestone 2: attention-head shape

Experiment card:

- Question: what changes when the same `embed_dim` is split into fewer/wider vs more/narrower heads?
- Runs: baseline heads, fewer heads, more heads with `embed_dim % num_heads == 0`.
- Primary code paths: `src/runtime_config.rs`, `src/model/mod.rs` attention modules.
- Plots: loss, tokens/sec, attention heatmaps from `/api/generate` if useful.
- Stop when: the mechanism writeup explains `head_dim` and any throughput/loss movement.

### Month 2 — training and data mechanics

#### Milestone 3: learning-rate schedule

Experiment card:

- Question: does warmup/decay improve stability or final validation loss under the fixed token budget?
- Runs: constant LR baseline, warmup+cosine, warmup+linear decay.
- Primary code paths: `src/model/training.rs`, runtime config docs if a flag is added.
- Plots: LR over step, train/value loss, tokens/sec.
- Stop when: schedule behavior is visible in logs and failed schedules are diagnosable.

#### Milestone 4: tokenizer vocabulary size

Experiment card:

- Question: how does BPE vocab size trade sequence length, output-head size, speed, and loss?
- Runs: baseline vocab, smaller vocab, larger vocab on the same corpus slice.
- Primary code paths: `src/bin/train-tokenizer.rs`, `src/tokenizer/bpe.rs`, checkpoint metadata validation.
- Plots: tokens per byte/word, loss/perplexity, tokens/sec, model parameter/head-size note.
- Stop when: the writeup separates tokenizer mechanics from model-quality claims.

### Month 3 — architecture implementation and loader mechanics

#### Milestone 5: RMSNorm or MLP gating

Default to RMSNorm first. Pick MLP gating instead only if normalization feels too small after Milestone 3.

RMSNorm card:

- Question: can RMS-only normalization match LayerNorm behavior with different cost/stability tradeoffs?
- Runs: LayerNorm baseline, RMSNorm blocks, RMSNorm blocks plus final norm variant if safe.
- Primary code paths: `src/model/mod.rs` normalization inside `Block` and final norm.
- Plots: loss, tokens/sec, gradient norm if already exposed.
- Stop when: the writeup explains residual scale control and failure modes.

#### Milestone 6: batch sampling strategy, then RoPE only if time remains

Batch sampling card:

- Question: how do random windows compare to sequential/epoch-style windows for stability, locality, and leakage risk?
- Runs: random-window baseline, deterministic sequential windows, shuffled chunk/epoch-style sampling.
- Primary code paths: `src/loader/data.rs`, train/value split handling.
- Plots: loss smoothness, tokens/sec, duplicate-window or coverage summary if cheap.
- Stop when: the notebook identifies one sampling policy to keep for future experiments.

RoPE stretch card:

- Question: where does positional information enter attention, and what changes near the context limit?
- Runs: learned absolute baseline, sinusoidal or RoPE variant, one context-length stress variant.
- Primary code paths: MiniGPT position embeddings and attention Q/K path.
- Stop when: Q/K position mechanics can be explained without hand-waving.

## Expected artifacts

- `docs/research/research-playground-roadmap.md` stays the handoff roadmap.
- Experiment cards live under `docs/research/experiments/`:
  - `checkpoint-eval-cadence.md`
  - `attention-head-shape.md`
  - `learning-rate-schedule.md`
  - `tokenizer-vocab-size.md`
  - `rmsnorm.md`
  - `batch-sampling-strategy.md`
- Automation notes live in `docs/research/automations.md`.
- One notebook/table per completed experiment.
- Optional config files only after the first command becomes too long to safely rerun.

## Global stopping rules

Stop or shrink an experiment when any of these happens:

- It cannot finish baseline + 2 variants within 6 GPU-hours.
- Burn/framework work dominates the mechanism being studied.
- The result cannot be interpreted because two major variables changed at once.
- The notebook cannot answer what changed, why it mattered, what happened, and how to diagnose failure.

Do not add distributed training, production mixed precision, quantization, or benchmark-suite work to this 3-month roadmap.
