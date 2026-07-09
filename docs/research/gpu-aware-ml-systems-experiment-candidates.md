# GPU-aware ML-systems experiment candidates

Source ticket: [Research compact GPU-aware ML-systems experiment candidates](https://github.com/lifesahaskell/rusty-gpt/issues/23)

## Question

Which compact, hours-scale ML-systems experiments are most worth considering for a 4060 Ti-class `rusty-gpt` research playground across training loop mechanics, data systems, and architecture internals?

This is a candidate list only. It does not choose the final roadmap.

## Local baseline facts

- `rusty-gpt` already has a CUDA-gated training path (`--features cuda -- --backend cuda`) and MiniGPT defaults are configurable by CLI/env, including `block_size`, `batch_size`, `embed_dim`, `num_heads`, `num_layers`, `dropout`, `learning_rate`, `train_steps`, `eval_interval`, gradient clipping, prefetch depth, and checkpoint cadence. Source: `README.md`, `CLAUDE.md`, `src/runtime_config.rs`.
- The shared training loop uses AdamW with a constant learning rate, optional norm gradient clipping, cross-entropy language-model loss, random-window batches, value loss/perplexity logging, and throughput logging. Source: `src/model/training.rs`.
- The data loader samples independent random windows from an in-memory token vector and shifts targets by one token. Source: `src/loader/data.rs`.
- MiniGPT uses token + learned position embeddings, pre-norm `LayerNorm`, multi-head self-attention, GELU MLP, and a final `LayerNorm`. Source: `src/model/mod.rs`.
- The BPE tokenizer is repo-local and byte-based: it starts from byte IDs, greedily merges frequent adjacent pairs, and saves JSON. Source: `src/tokenizer/bpe.rs`.

## Primary-source anchors

- Transformer attention and positional encodings: Vaswani et al., “Attention Is All You Need,” defines scaled dot-product attention, multi-head attention, residual/norm sublayers, and sinusoidal positional encodings. Source: <https://arxiv.org/abs/1706.03762>.
- AdamW: Loshchilov and Hutter, “Decoupled Weight Decay Regularization,” separates weight decay from the adaptive gradient update, which is the optimizer family already used here. Source: <https://arxiv.org/abs/1711.05101>.
- BPE for open-vocabulary text: Sennrich et al., “Neural Machine Translation of Rare Words with Subword Units,” uses byte-pair encoding to represent rare words with subword units. Source: <https://arxiv.org/abs/1508.07909>.
- RoPE: Su et al., “RoFormer,” applies rotary position embeddings to self-attention so positional information is represented by rotating query/key vectors. Source: <https://arxiv.org/abs/2104.09864>.
- RMSNorm: Zhang and Sennrich, “Root Mean Square Layer Normalization,” removes mean-centering from LayerNorm and normalizes by RMS, reducing normalization work while preserving re-scaling invariance. Source: <https://arxiv.org/abs/1910.07467>.
- GLU variants: Shazeer, “GLU Variants Improve Transformer,” evaluates gated feed-forward variants including GEGLU/SwiGLU as Transformer FFN replacements. Source: <https://arxiv.org/abs/2002.05202>.

## Short candidate list

### Training loop mechanics

1. **Learning-rate schedule ablation**
   - Shape: constant LR baseline vs warmup+cosine vs warmup+linear decay.
   - Why it fits: current loop already passes one scalar LR into `optimizer.step`; adding a schedule is a small code-path change with visible loss-curve effects.
   - Teaches: optimizer step mechanics, warmup instability, decay tradeoffs, and how loss curves reflect schedule choices.
   - Plots: LR over steps, train/value loss, tokens/sec.
   - Risk: schedule effects can be hidden if runs are too short; keep total token budget fixed.

2. **Effective batch size ablation**
   - Shape: baseline batch size vs smaller microbatch+gradient accumulation vs larger direct batch if VRAM allows.
   - Why it fits: the repo already exposes `batch_size`; accumulation would teach the boundary between memory, optimizer step count, and token throughput.
   - Teaches: GPU memory pressure, update frequency, gradient noise, and throughput/VRAM tradeoffs.
   - Plots: loss by optimizer step and by tokens seen, tokens/sec, peak VRAM.
   - Risk: Burn gradient accumulation API details may drive implementation cost; keep as a later training-systems candidate if API friction is high.

3. **Checkpoint/eval cadence ablation**
   - Shape: sparse eval/checkpoint vs medium cadence vs frequent cadence.
   - Why it fits: `eval_interval`, `checkpoint_interval`, and checkpoint metadata already exist.
   - Teaches: measurement overhead, recoverability, and how observability choices perturb training throughput.
   - Plots: step time spikes, tokens/sec, artifact count/size, value-loss resolution.
   - Risk: mostly systems learning, not model-quality learning; useful early because it hardens the experiment protocol.

### Data systems

4. **Tokenizer vocabulary-size ablation**
   - Shape: lane baseline vocab vs smaller vs larger BPE vocab on the same corpus slice.
   - Why it fits: repo owns `train-tokenizer`; MiniGPT strict metadata already records tokenizer hash, making tokenizer swaps visible.
   - Teaches: subword granularity, sequence length pressure, vocab projection size, and checkpoint/tokenizer compatibility.
   - Plots: tokens per byte/word, final loss/perplexity, tokens/sec, model parameter count if exposed.
   - Risk: changing vocab changes model head shape, so compare mechanism and throughput more than raw loss.

5. **Dataset slice/mixture ablation**
   - Shape: Shakespeare/local baseline vs Wikitext slice vs code-text slice, with tokenizer policy held fixed or explicitly varied.
   - Why it fits: `hf://` dataset loading and local corpus collection already exist.
   - Teaches: domain mismatch, validation leakage risks, data volume vs repetition, and why data policy matters before architecture claims.
   - Plots: train/value gap, sample generations, token distribution summaries.
   - Risk: easy to confound tokenizer and dataset; only run after the data/tokenizer policy ticket is decided.

6. **Batch sampling strategy ablation**
   - Shape: current random windows vs deterministic sequential windows vs shuffled epoch-style chunks.
   - Why it fits: `DataLoader::next_raw_batch` is tiny and centralized.
   - Teaches: stochastic sampling, epoch semantics, cache locality, and validation comparability.
   - Plots: loss smoothness, tokens/sec, duplicate-window rate if instrumented.
   - Risk: sequential strategy needs careful train/value split to avoid accidental leakage.

### Architecture internals

7. **Position encoding ablation**
   - Shape: learned absolute position embeddings baseline vs sinusoidal vs RoPE.
   - Why it fits: MiniGPT position handling is concentrated around embeddings and attention Q/K paths.
   - Teaches: where position enters the Transformer, extrapolation limits, and query/key geometry.
   - Plots: value loss by context length, attention patterns, generation at contexts near `block_size`.
   - Risk: RoPE touches attention internals; do after one simpler architecture swap.

8. **Normalization ablation**
   - Shape: LayerNorm baseline vs RMSNorm variant vs maybe no final norm if safe.
   - Why it fits: MiniGPT pre-norm blocks and final norm are explicit in `src/model/mod.rs`.
   - Teaches: residual stream scale control, normalization cost, and training stability.
   - Plots: loss curves, step time, gradient norm if exposed.
   - Risk: Burn may not ship RMSNorm directly; a tiny local implementation may be needed.

9. **MLP activation/gating ablation**
   - Shape: GELU MLP baseline vs ReLU/GELU size tweak vs SwiGLU/GEGLU.
   - Why it fits: MiniGPT MLP is isolated inside `Block`/MLP code.
   - Teaches: feed-forward capacity, gating, parameter-count fairness, and activation cost.
   - Plots: loss, tokens/sec, parameter count, maybe activation memory.
   - Risk: fair comparison requires matching parameter budget or documenting that the variant changes capacity.

10. **Attention-head shape ablation**
   - Shape: same `d_model` with fewer/wider heads vs baseline vs more/narrower heads.
   - Why it fits: `num_heads` is already configurable and validated against `embed_dim`.
   - Teaches: head dimension, parallel attention views, and divisibility/shape constraints.
   - Plots: attention heatmaps, loss, tokens/sec.
   - Risk: may teach less new code than RoPE/RMSNorm, but it is very cheap and good as a first architecture-lane warmup.

## Best first candidates for the roadmap to consider

1. **Checkpoint/eval cadence ablation** — cheapest way to validate the experiment protocol and notebook/table workflow.
2. **Learning-rate schedule ablation** — high learning value, small code path, visible curves.
3. **Tokenizer vocabulary-size ablation** — strong data-systems lesson using code already present.
4. **Attention-head shape ablation** — nearly no implementation risk; good model-internals warmup.
5. **RMSNorm or MLP gating ablation** — first real architecture implementation once the protocol is working.
6. **RoPE ablation** — best deeper model-internals experiment, but save until attention internals are comfortable.

## Avoid for this 3-month roadmap unless scope expands

- Full distributed training, multi-GPU, or cluster orchestration: outside one local 4060 Ti-class setup.
- Production mixed precision/quantization stack: useful later, but likely framework/API-heavy compared with the learning target.
- Large benchmark suites: violates the hours-scale, mechanism-first protocol.
