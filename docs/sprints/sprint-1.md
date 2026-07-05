# Sprint 1 — MoE Building Blocks

Goal: land the Mixture-of-Experts primitives — router, expert feed-forward,
load-balancing auxiliary loss — as fully unit-tested modules, and generalize
`Block` so its feed-forward slot is pluggable. **No user-visible behavior
changes this sprint**: MiniGPT and the three char-level variants must be
bit-for-bit unaffected.

## Contents

- [S1-T1 — `FeedForward` abstraction in `Block`](#s1-t1--feedforward-abstraction-in-block)
- [S1-T2 — `Router` module](#s1-t2--router-module)
- [S1-T3 — `MoeFeedForward`](#s1-t3--moefeedforward)
- [S1-T4 — Load-balancing auxiliary loss](#s1-t4--load-balancing-auxiliary-loss)
- [S1-T5 — Aux-loss return channel through `Block`](#s1-t5--aux-loss-return-channel-through-block)
- [Sprint exit checklist](#sprint-exit-checklist)

---

## S1-T1 — `FeedForward` abstraction in `Block`

Replace `Block`'s concrete `mlp: Mlp<B>` field (`src/model/mod.rs`) with an
enum module:

```rust
#[derive(Module, Debug)]
pub enum FeedForward<B: Backend> {
    Dense(Mlp<B>),
    Moe(MoeFeedForward<B>), // added in S1-T3; stub or feature-order the enum accordingly
}
```

`Block::new` keeps its current signature and constructs `Dense` (preserving
`d_ff = 4 * d_model`); add a `Block::new_with_feed_forward` (or a small
builder) for MoE construction later. Confirm `#[derive(Module)]` works on the
enum with Burn 0.21 (Burn supports enum modules); if it does not, fall back to
a two-field struct with exactly one populated option and document why.

> **Deviation taken during implementation.** `#[derive(Module)]` does work on
> enums in Burn 0.21, but the generated record item is an externally tagged
> serde enum (`{"Dense": {...}}`), and record fields carry no
> `#[serde(default)]` — so both the enum and the two-field-struct fallback
> change `Block`'s record tree and break loading every existing dense
> checkpoint, which the acceptance criteria forbid. Implemented instead as a
> generic feed-forward slot: `Block<B, F = Mlp<B>>` plus a
> `pub trait FeedForward<B: Backend>: Module<B>` implemented by `Mlp` (dense)
> and `moe::MoeFeedForward`. The default parameter keeps `Block<B>` meaning
> the dense block everywhere, so the record tree — and therefore old `.mpk`
> checkpoints — are unchanged (locked in by the
> `minigpt_loads_checkpoints_saved_with_pre_feedforward_record_layout` test).
> Burn's derive supports generic module parameters
> (`ModuleWithGenericModule<B, M>` in burn-core's own test suite). Later
> sprints that reference `FeedForward::Dense` / `FeedForward::Moe` map to
> `Mlp` / `MoeFeedForward` as `F`, e.g. MoeGpt uses
> `Vec<Block<B, MoeFeedForward<B>>>`.

**Files**: `src/model/mod.rs`

**Acceptance criteria**
- All existing tests pass unchanged (`cargo test`), including MiniGPT forward
  shape tests and training smoke tests — proof the refactor is behavior-neutral.
- New unit test: a `Block` built via `Block::new` reports/uses a `Dense`
  feed-forward and produces identical output to a directly-constructed
  `Mlp` on the same input tensor.
- MiniGPT checkpoint save→load round-trip test still passes (record structure
  of a `Dense` block must remain loadable; if the enum wrapper changes the
  record tree, this task must include a compatibility shim or explicitly
  document the break — breaking old *dense* checkpoints is **not** acceptable).

## S1-T2 — `Router` module

New file `src/model/moe.rs` (registered in `src/model/mod.rs`). The router is
a linear gate over `d_model`:

```text
gate: Linear<B> [d_model -> num_experts]
forward(x: Tensor<B, 3>) ->
    RouterOutput {
        probs:   Tensor<B, 3>,        // [batch, seq, num_experts] softmax over experts
        top_k_indices: Tensor<B, 3, Int>, // [batch, seq, k]
        top_k_weights: Tensor<B, 3>,  // [batch, seq, k], renormalized to sum to 1
    }
```

Use `Tensor::topk`-style selection if available in Burn 0.21, otherwise a
manual iterative-max + mask approach (the same style as the causal-mask
handling in `MultiHeadAttention`).

**Files**: `src/model/moe.rs`, `src/model/mod.rs` (module registration + re-export)

**Acceptance criteria**
- Shape tests: for `[batch=2, seq=4, d_model=8]`, `num_experts=4`, `k=2`,
  outputs have the shapes above.
- `top_k_weights` rows sum to 1 (within float tolerance) and are non-negative.
- `top_k_indices` values are unique per token and `< num_experts`.
- Constructor panics (with message) on `num_experts == 0`, `k == 0`, or
  `k > num_experts` — same assert style as `MultiHeadAttention::new`'s
  divisibility panic.
- Deterministic test with hand-set gate weights: a token whose gate logits
  favor expert 2 routes to expert 2 first.

## S1-T3 — `MoeFeedForward`

In `src/model/moe.rs`:

```text
MoeFeedForward {
    router: Router<B>,
    experts: Vec<Mlp<B>>,   // num_experts × Mlp(d_model, d_ff)
    num_experts, top_k,
}
forward(x: Tensor<B, 3>) -> (Tensor<B, 3>, MoeForwardAux)
```

Dispatch: compute every selected expert's output for its tokens and combine
weighted by `top_k_weights`. A dense-compute reference implementation (run all
experts on all tokens, then mask/weight) is acceptable for correctness first;
a gather/scatter sparse dispatch is a later optimization (S4/S5 benchmarks
will quantify it). `MoeForwardAux` carries what the aux loss needs: router
`probs` and the top-k selection (or the already-reduced per-expert token
fractions).

**Files**: `src/model/moe.rs`

**Acceptance criteria**
- Output shape `[batch, seq, d_model]` for arbitrary valid configs.
- Equivalence test: `num_experts=1, top_k=1` produces output equal (within
  tolerance) to a plain `Mlp` initialized with the same weights.
- With `top_k = num_experts`, output equals the router-probability-weighted
  mixture of all experts (dense mixture check).
- Autodiff test: `.backward()` on a scalar reduction of the output produces
  gradients for **every** expert that received at least one token, and for
  the router gate.

## S1-T4 — Load-balancing auxiliary loss

Switch-Transformer-style loss in `src/model/moe.rs`:

```text
load_balancing_loss(probs, top_k_indices, num_experts) -> Tensor<B, 1>
  f_i = fraction of tokens whose top-1 choice is expert i
  P_i = mean router probability assigned to expert i
  loss = num_experts * Σ_i f_i * P_i
```

Perfectly uniform routing yields `loss ≈ 1.0`; collapse onto one expert yields
`loss ≈ num_experts`. Keep the function pure (tensor in, scalar tensor out) so
it is trivially unit-testable and reusable per-layer.

**Files**: `src/model/moe.rs`

**Acceptance criteria**
- Uniform-routing fixture (identical logits for all experts) → loss within
  tolerance of `1.0`.
- Collapsed-routing fixture (one expert always wins with prob→1) → loss within
  tolerance of `num_experts as f32`.
- Loss is differentiable: `.backward()` produces router-gate gradients.
- Property check across random fixtures: `1.0 <= loss <= num_experts` (within
  tolerance).

## S1-T5 — Aux-loss return channel through `Block`

The MoE aux loss is produced inside the feed-forward but consumed by the
training loop, so `Block` needs a way to surface it. Add aux-carrying forward
variants without disturbing the six existing signatures:

- `FeedForward::forward(x) -> Tensor<B, 3>` stays for dense-only callers;
  add `FeedForward::forward_with_aux(x) -> (Tensor<B, 3>, Option<Tensor<B, 1>>)`
  (`Dense` returns `None`, `Moe` returns `Some(load_balancing_loss)`).
- Add `Block::forward_with_mask_and_aux(...)` (the training path uses
  `forward_with_mask`) and route the other five variants
  (`forward`, `forward_with_weights`, `forward_with_weights_and_mask`,
  `forward_with_attention`, `forward_with_cache` in `src/model/mod.rs`)
  through the aux-aware implementation internally, discarding the aux value,
  so there is exactly **one** feed-forward tail implementation.

Inference paths (cached generation, attention introspection) deliberately drop
the aux loss — it only matters under autodiff.

**Files**: `src/model/mod.rs`, `src/model/moe.rs`

**Acceptance criteria**
- `Block::forward_with_mask_and_aux` with a `Dense` feed-forward returns
  `None` aux and output identical to `forward_with_mask`.
- With a `Moe` feed-forward, returns `Some(aux)` where `aux` matches calling
  `load_balancing_loss` directly on the router outputs for the same input.
- No behavior change in the six pre-existing `Block` forward variants
  (existing tests green).

## Sprint exit checklist

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --features cuda -- -D warnings
cargo fmt --all -- --check
```

- `cargo run -- --model compare` output unchanged versus `main` (four models,
  same losses for a fixed seed if seeding exists; otherwise same shape of
  output).
- No new flags, no doc changes required this sprint (all internal).
