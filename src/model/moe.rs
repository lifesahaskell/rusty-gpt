//! Mixture-of-Experts building blocks: a linear top-k [`Router`], a
//! dense-compute [`MoeFeedForward`], and the Switch-Transformer-style
//! [`load_balancing_loss`].
//!
//! These are the primitives a later `MoeGpt` model plugs into a transformer
//! [`Block`](super::Block) via the [`FeedForward`](super::FeedForward) enum.
//! Nothing here changes user-visible behavior on its own — the dense MiniGPT
//! path never constructs any of these types.
//!
//! Routing follows the standard Switch/Mixtral reference design: a linear gate
//! over `d_model` produces per-expert logits, softmax turns them into
//! probabilities, and the top-`k` experts are selected with their weights
//! renormalized to sum to 1. The feed-forward is a *dense-compute* reference
//! implementation — every expert runs on every token and unselected experts
//! are masked out by a zero combine weight. A sparse gather/scatter dispatch is
//! a later optimization (tracked for the S4/S5 benchmarks).

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use super::Mlp;

/// Result of routing a batch of token representations through the gate.
#[derive(Debug, Clone)]
pub struct RouterOutput<B: Backend> {
    /// Softmax probabilities over experts, `[batch, seq, num_experts]`.
    pub probs: Tensor<B, 3>,
    /// Indices of the selected top-k experts per token, `[batch, seq, k]`.
    /// Sorted by descending gate probability, so column 0 is the top-1 choice.
    pub top_k_indices: Tensor<B, 3, Int>,
    /// Weights of the selected experts, `[batch, seq, k]`, renormalized so each
    /// token's row sums to 1.
    pub top_k_weights: Tensor<B, 3>,
}

/// Linear gate that routes each token to its top-k experts.
#[derive(Module, Debug)]
pub struct Router<B: Backend> {
    gate: Linear<B>,
    num_experts: usize,
    top_k: usize,
}

impl<B: Backend> Router<B> {
    /// Build a router over `d_model` selecting `top_k` of `num_experts`.
    ///
    /// Panics — in the same assert style as
    /// [`MultiHeadAttention::new`](super::MultiHeadAttention::new) — on
    /// `num_experts == 0`, `top_k == 0`, or `top_k > num_experts`.
    pub fn new(d_model: usize, num_experts: usize, top_k: usize, device: &B::Device) -> Self {
        assert!(num_experts > 0, "num_experts must be greater than zero");
        assert!(top_k > 0, "top_k must be greater than zero");
        assert!(top_k <= num_experts, "top_k must not exceed num_experts");

        Self {
            gate: LinearConfig::new(d_model, num_experts).init(device),
            num_experts,
            top_k,
        }
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Route `x` (`[batch, seq, d_model]`) to experts.
    pub fn forward(&self, x: Tensor<B, 3>) -> RouterOutput<B> {
        let logits = self.gate.forward(x); // [batch, seq, num_experts]
        let probs = softmax(logits, 2);

        // Top-k via a descending sort + take-first-k. `Tensor::topk` asserts
        // `len > k`, which rejects the valid `top_k == num_experts` (dense
        // mixture) case, so we select the leading `k` of the sorted order
        // instead — this also keeps the per-token indices unique.
        let (sorted_probs, sorted_indices) = probs.clone().sort_descending_with_indices(2);
        let k_range = Tensor::arange(0..self.top_k as i64, &probs.device());
        let top_k_weights_raw = sorted_probs.select(2, k_range.clone()); // [batch, seq, k]
        let top_k_indices = sorted_indices.select(2, k_range); // [batch, seq, k]

        // Renormalize the selected weights so each token's row sums to 1.
        let denom = top_k_weights_raw.clone().sum_dim(2); // [batch, seq, 1]
        let top_k_weights = top_k_weights_raw / denom;

        RouterOutput {
            probs,
            top_k_indices,
            top_k_weights,
        }
    }
}

/// Auxiliary data surfaced from a [`MoeFeedForward`] forward pass, carrying
/// exactly what the load-balancing loss needs.
#[derive(Debug, Clone)]
pub struct MoeForwardAux<B: Backend> {
    /// Router softmax probabilities, `[batch, seq, num_experts]`.
    pub probs: Tensor<B, 3>,
    /// Selected top-k expert indices, `[batch, seq, k]`.
    pub top_k_indices: Tensor<B, 3, Int>,
    /// Number of experts (needed to scale the loss).
    pub num_experts: usize,
}

impl<B: Backend> MoeForwardAux<B> {
    /// Compute the Switch-Transformer load-balancing loss for this pass.
    pub fn load_balancing_loss(&self) -> Tensor<B, 1> {
        load_balancing_loss(
            self.probs.clone(),
            self.top_k_indices.clone(),
            self.num_experts,
        )
    }
}

/// Mixture-of-experts feed-forward: a [`Router`] over a pool of expert
/// [`Mlp`]s. Dense-compute reference implementation — every expert runs on
/// every token and the router's renormalized top-k weights combine them
/// (unselected experts contribute a zero weight).
#[derive(Module, Debug)]
pub struct MoeFeedForward<B: Backend> {
    router: Router<B>,
    experts: Vec<Mlp<B>>,
    num_experts: usize,
    top_k: usize,
}

impl<B: Backend> MoeFeedForward<B> {
    /// Build a MoE feed-forward: `num_experts` experts each `Mlp(d_model,
    /// d_ff)`, routed top-`top_k`. Shares [`Router::new`]'s panics.
    pub fn new(
        d_model: usize,
        d_ff: usize,
        num_experts: usize,
        top_k: usize,
        device: &B::Device,
    ) -> Self {
        let router = Router::new(d_model, num_experts, top_k, device);
        let experts = (0..num_experts)
            .map(|_| Mlp::new(d_model, d_ff, device))
            .collect();

        Self {
            router,
            experts,
            num_experts,
            top_k,
        }
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Forward `x` (`[batch, seq, d_model]`), returning the combined expert
    /// output and the [`MoeForwardAux`] needed for the load-balancing loss.
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, MoeForwardAux<B>) {
        let [batch, seq, d_model] = x.shape().dims();
        let router_out = self.router.forward(x.clone());

        // Scatter the renormalized top-k weights into a dense
        // `[batch, seq, num_experts]` combine matrix. `one_hot_fill` on the
        // selected indices gives `[batch, seq, k, num_experts]`; weight each
        // slot and sum over the k dimension.
        let selection = router_out.top_k_indices.clone().float().one_hot_fill::<4>(
            self.num_experts,
            1.0,
            0.0,
            -1,
        ); // [batch, seq, k, num_experts]
        let weights = router_out.top_k_weights.clone().unsqueeze_dim(3); // [batch, seq, k, 1]
        let combine = (selection * weights)
            .sum_dim(2)
            .reshape([batch, seq, self.num_experts]); // [batch, seq, num_experts]

        // Dense compute: run every expert, weight by its combine column.
        let mut output = Tensor::zeros([batch, seq, d_model], &x.device());
        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let expert_out = expert.forward(x.clone()); // [batch, seq, d_model]
            let gate = combine
                .clone()
                .slice([0..batch, 0..seq, expert_idx..expert_idx + 1]); // [batch, seq, 1]
            output = output + expert_out * gate; // broadcast [batch, seq, 1] over d_model
        }

        let aux = MoeForwardAux {
            probs: router_out.probs,
            top_k_indices: router_out.top_k_indices,
            num_experts: self.num_experts,
        };

        (output, aux)
    }
}

/// Switch-Transformer load-balancing auxiliary loss.
///
/// ```text
/// f_i  = fraction of tokens whose top-1 choice is expert i
/// P_i  = mean router probability assigned to expert i
/// loss = num_experts * Σ_i f_i * P_i
/// ```
///
/// Perfectly uniform routing yields `≈ 1.0`; total collapse onto one expert
/// yields `≈ num_experts`. Pure (tensors in, scalar tensor out) and
/// differentiable through `probs`, so it is trivially unit-testable and
/// reusable per-layer.
///
/// * `probs` — router softmax probabilities, `[batch, seq, num_experts]`.
/// * `top_k_indices` — selected expert indices, `[batch, seq, k]`, column 0 the
///   top-1 choice.
pub fn load_balancing_loss<B: Backend>(
    probs: Tensor<B, 3>,
    top_k_indices: Tensor<B, 3, Int>,
    num_experts: usize,
) -> Tensor<B, 1> {
    let [batch, seq, _experts] = probs.shape().dims();
    let num_tokens = (batch * seq) as f32;

    // f_i: fraction of tokens whose top-1 expert is i.
    let top1 = top_k_indices.slice([0..batch, 0..seq, 0..1]); // [batch, seq, 1]
    let top1_one_hot = top1
        .float()
        .one_hot_fill::<4>(num_experts, 1.0, 0.0, -1)
        .reshape([batch, seq, num_experts]); // [batch, seq, num_experts]
    let f = top1_one_hot.sum_dim(0).sum_dim(1).reshape([num_experts]) / num_tokens;

    // P_i: mean router probability for expert i.
    let p = probs.sum_dim(0).sum_dim(1).reshape([num_experts]) / num_tokens;

    (f * p).sum() * num_experts as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::module::Param;
    use burn::tensor::{Tensor, TensorData};

    type TestBackend = NdArray<f32, i64>;
    type AutodiffBackend = Autodiff<NdArray<f32, i64>>;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ----- S1-T2: Router -----

    #[test]
    fn router_produces_expected_output_shapes() {
        let device = NdArrayDevice::Cpu;
        let router = Router::<TestBackend>::new(8, 4, 2, &device);
        let x = Tensor::<TestBackend, 3>::zeros([2, 4, 8], &device);

        let out = router.forward(x);

        assert_eq!([2, 4, 4], out.probs.shape().dims());
        assert_eq!([2, 4, 2], out.top_k_indices.shape().dims());
        assert_eq!([2, 4, 2], out.top_k_weights.shape().dims());
    }

    #[test]
    fn router_top_k_weights_are_non_negative_and_sum_to_one() {
        let device = NdArrayDevice::Cpu;
        let router = Router::<TestBackend>::new(8, 4, 2, &device);
        let x = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0..2 * 3 * 8)
                    .map(|i| (i as f32) * 0.01)
                    .collect::<Vec<_>>(),
                [2, 3, 8],
            ),
            &device,
        );

        let out = router.forward(x);
        let weights = out
            .top_k_weights
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            weights.iter().all(|&w| w >= 0.0),
            "weights must be non-negative"
        );

        let row_sums = out
            .top_k_weights
            .sum_dim(2)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for sum in row_sums {
            assert!(approx_eq(sum, 1.0, 1e-5), "row sum {sum} should be 1.0");
        }
    }

    #[test]
    fn router_top_k_indices_are_unique_per_token_and_in_range() {
        let device = NdArrayDevice::Cpu;
        let router = Router::<TestBackend>::new(8, 4, 3, &device);
        let x = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0..2 * 2 * 8).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
                [2, 2, 8],
            ),
            &device,
        );

        let out = router.forward(x);
        let indices = out.top_k_indices.into_data().to_vec::<i64>().unwrap();
        // 2*2 tokens, k=3 each.
        for token in indices.chunks(3) {
            assert!(
                token.iter().all(|&i| (0..4).contains(&i)),
                "index out of range"
            );
            let mut sorted = token.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                token.len(),
                "indices must be unique per token"
            );
        }
    }

    #[test]
    #[should_panic(expected = "num_experts must be greater than zero")]
    fn router_rejects_zero_experts() {
        let device = NdArrayDevice::Cpu;
        let _ = Router::<TestBackend>::new(8, 0, 1, &device);
    }

    #[test]
    #[should_panic(expected = "top_k must be greater than zero")]
    fn router_rejects_zero_top_k() {
        let device = NdArrayDevice::Cpu;
        let _ = Router::<TestBackend>::new(8, 4, 0, &device);
    }

    #[test]
    #[should_panic(expected = "top_k must not exceed num_experts")]
    fn router_rejects_top_k_larger_than_num_experts() {
        let device = NdArrayDevice::Cpu;
        let _ = Router::<TestBackend>::new(8, 2, 3, &device);
    }

    #[test]
    fn router_sends_token_to_the_expert_its_gate_favors() {
        let device = NdArrayDevice::Cpu;
        let mut router = Router::<TestBackend>::new(4, 4, 2, &device);

        // Hand-set the gate: zero weights, bias favoring expert 2, so every
        // token's top-1 choice is expert 2 regardless of input.
        router.gate.weight = Param::from_tensor(Tensor::<TestBackend, 2>::zeros([4, 4], &device));
        router.gate.bias = Some(Param::from_tensor(Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 10.0, 0.0], [4]),
            &device,
        )));

        let x = Tensor::<TestBackend, 3>::ones([1, 3, 4], &device);
        let out = router.forward(x);
        let indices = out.top_k_indices.into_data().to_vec::<i64>().unwrap();
        // First selected expert per token (k=2, stride 2) must be expert 2.
        for token in indices.chunks(2) {
            assert_eq!(2, token[0], "token should route to expert 2 first");
        }
    }

    // ----- S1-T3: MoeFeedForward -----

    #[test]
    fn moe_feed_forward_returns_model_dim_for_each_token_position() {
        let device = NdArrayDevice::Cpu;
        let moe = MoeFeedForward::<TestBackend>::new(8, 16, 4, 2, &device);
        let x = Tensor::<TestBackend, 3>::zeros([2, 5, 8], &device);

        let (output, _aux) = moe.forward(x);

        assert_eq!([2, 5, 8], output.shape().dims());
    }

    #[test]
    fn moe_with_single_expert_equals_plain_mlp() {
        let device = NdArrayDevice::Cpu;
        let moe = MoeFeedForward::<TestBackend>::new(8, 16, 1, 1, &device);
        let x = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0..2 * 3 * 8)
                    .map(|i| (i as f32) * 0.03 - 0.5)
                    .collect::<Vec<_>>(),
                [2, 3, 8],
            ),
            &device,
        );

        // With one expert the router weight renormalizes to 1, so the MoE
        // output must equal the single expert's plain Mlp forward.
        let expected = moe.experts[0]
            .forward(x.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let (output, _aux) = moe.forward(x);
        let actual = output.into_data().to_vec::<f32>().unwrap();

        assert_eq!(expected.len(), actual.len());
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!(approx_eq(*e, *a, 1e-5), "expected {e}, got {a}");
        }
    }

    #[test]
    fn moe_with_top_k_equal_to_num_experts_is_probability_weighted_mixture() {
        let device = NdArrayDevice::Cpu;
        let moe = MoeFeedForward::<TestBackend>::new(8, 16, 3, 3, &device);
        let x = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0..2 * 3 * 8)
                    .map(|i| (i as f32).cos() * 0.4)
                    .collect::<Vec<_>>(),
                [2, 3, 8],
            ),
            &device,
        );

        // Reference: with top_k == num_experts, the renormalized weights equal
        // the router probabilities, so the output is Σ_e probs_e * expert_e(x).
        let router_out = moe.router.forward(x.clone());
        let [batch, seq, _] = x.shape().dims();
        let mut expected = Tensor::<TestBackend, 3>::zeros([batch, seq, 8], &device);
        for (idx, expert) in moe.experts.iter().enumerate() {
            let prob = router_out
                .probs
                .clone()
                .slice([0..batch, 0..seq, idx..idx + 1]); // [batch, seq, 1]
            expected = expected + expert.forward(x.clone()) * prob;
        }
        let expected = expected.into_data().to_vec::<f32>().unwrap();

        let (output, _aux) = moe.forward(x);
        let actual = output.into_data().to_vec::<f32>().unwrap();
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!(approx_eq(*e, *a, 1e-5), "expected {e}, got {a}");
        }
    }

    #[test]
    fn moe_backward_produces_gradients_for_experts_and_router() {
        let device = NdArrayDevice::Cpu;
        // top_k == num_experts guarantees every expert receives tokens.
        let moe = MoeFeedForward::<AutodiffBackend>::new(8, 16, 2, 2, &device);
        let x = Tensor::<AutodiffBackend, 3>::from_data(
            TensorData::new(
                (0..4 * 8).map(|i| (i as f32) * 0.05).collect::<Vec<_>>(),
                [1, 4, 8],
            ),
            &device,
        );

        let (output, _aux) = moe.forward(x);
        let grads = output.sum().backward();

        for (idx, expert) in moe.experts.iter().enumerate() {
            let grad = expert.fc1.weight.val().grad(&grads);
            let grad = grad.unwrap_or_else(|| panic!("expert {idx} fc1 should have a gradient"));
            let magnitude: f32 = grad.abs().sum().into_scalar();
            assert!(magnitude > 0.0, "expert {idx} gradient should be non-zero");
        }

        let gate_grad = moe
            .router
            .gate
            .weight
            .val()
            .grad(&grads)
            .expect("router gate should have a gradient");
        assert!(
            gate_grad.abs().sum().into_scalar() >= 0.0,
            "router gate gradient should be defined"
        );
    }

    // ----- S1-T4: load-balancing loss -----

    fn router_outputs_for(
        logits: Vec<f32>,
        batch: usize,
        seq: usize,
        num_experts: usize,
        top_k: usize,
        device: &NdArrayDevice,
    ) -> RouterOutput<TestBackend> {
        // Build a router whose gate reproduces the given per-token logits by
        // setting zero weights and per-call biases is awkward; instead we feed
        // the logits straight through the same top-k math the router uses.
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(logits, [batch, seq, num_experts]),
            device,
        );
        let probs = softmax(logits, 2);
        let (sorted_probs, sorted_indices) = probs.clone().sort_descending_with_indices(2);
        let k_range = Tensor::arange(0..top_k as i64, device);
        let top_k_weights_raw = sorted_probs.select(2, k_range.clone());
        let top_k_indices = sorted_indices.select(2, k_range);
        let denom = top_k_weights_raw.clone().sum_dim(2);
        RouterOutput {
            probs,
            top_k_indices,
            top_k_weights: top_k_weights_raw / denom,
        }
    }

    #[test]
    fn load_balancing_loss_is_one_for_uniform_routing() {
        let device = NdArrayDevice::Cpu;
        let num_experts = 4;
        // Identical logits for all experts => uniform probabilities.
        let logits = vec![0.5f32; 2 * 3 * num_experts];
        let out = router_outputs_for(logits, 2, 3, num_experts, 2, &device);

        let loss: f32 =
            load_balancing_loss(out.probs, out.top_k_indices, num_experts).into_scalar();

        assert!(
            approx_eq(loss, 1.0, 1e-4),
            "uniform loss {loss} should be ~1.0"
        );
    }

    #[test]
    fn load_balancing_loss_approaches_num_experts_on_collapse() {
        let device = NdArrayDevice::Cpu;
        let num_experts = 4;
        // Every token overwhelmingly prefers expert 1.
        let mut logits = Vec::new();
        for _ in 0..2 * 3 {
            logits.extend_from_slice(&[0.0, 30.0, 0.0, 0.0]);
        }
        let out = router_outputs_for(logits, 2, 3, num_experts, 2, &device);

        let loss: f32 =
            load_balancing_loss(out.probs, out.top_k_indices, num_experts).into_scalar();

        assert!(
            approx_eq(loss, num_experts as f32, 1e-3),
            "collapsed loss {loss} should be ~{num_experts}"
        );
    }

    #[test]
    fn load_balancing_loss_is_differentiable() {
        let device = NdArrayDevice::Cpu;
        let router = Router::<AutodiffBackend>::new(8, 4, 2, &device);
        let x = Tensor::<AutodiffBackend, 3>::from_data(
            TensorData::new(
                (0..2 * 3 * 8)
                    .map(|i| (i as f32) * 0.02)
                    .collect::<Vec<_>>(),
                [2, 3, 8],
            ),
            &device,
        );

        let out = router.forward(x);
        let loss = load_balancing_loss(out.probs, out.top_k_indices, 4);
        let grads = loss.backward();

        let gate_grad = router
            .gate
            .weight
            .val()
            .grad(&grads)
            .expect("gate weight should have a gradient");
        assert!(
            gate_grad.abs().sum().into_scalar() > 0.0,
            "router gate gradient should be non-zero"
        );
    }

    #[test]
    fn load_balancing_loss_is_bounded_between_one_and_num_experts() {
        let device = NdArrayDevice::Cpu;
        let num_experts = 5;
        let top_k = 2;
        for seed in 0..8u32 {
            let logits = (0..2 * 4 * num_experts)
                .map(|i| {
                    ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) as f32).sin() * 3.0
                })
                .collect::<Vec<_>>();
            let out = router_outputs_for(logits, 2, 4, num_experts, top_k, &device);
            let loss: f32 =
                load_balancing_loss(out.probs, out.top_k_indices, num_experts).into_scalar();
            assert!(
                loss >= 1.0 - 1e-3 && loss <= num_experts as f32 + 1e-3,
                "loss {loss} out of [1, {num_experts}]"
            );
        }
    }
}
