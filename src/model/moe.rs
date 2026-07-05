//! Mixture-of-Experts building blocks: the softmax router and the expert
//! feed-forward layer that plugs into `Block` via the `FeedForward` trait.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{IndexingUpdateOp, Int, Tensor};

use super::{FeedForward, Mlp};

/// Linear gate over `d_model` that scores each token against every expert and
/// selects the `top_k` best.
#[derive(Module, Debug)]
pub struct Router<B: Backend> {
    gate: Linear<B>,
    num_experts: usize,
    top_k: usize,
}

/// Routing decision for a batch of token activations.
pub struct RouterOutput<B: Backend> {
    /// Softmax over experts: `[batch, seq, num_experts]`.
    pub probs: Tensor<B, 3>,
    /// Selected expert ids per token: `[batch, seq, top_k]`.
    pub top_k_indices: Tensor<B, 3, Int>,
    /// Selection weights renormalized to sum to 1 per token: `[batch, seq, top_k]`.
    pub top_k_weights: Tensor<B, 3>,
}

impl<B: Backend> Router<B> {
    pub fn new(d_model: usize, num_experts: usize, top_k: usize, device: &B::Device) -> Self {
        assert!(num_experts > 0, "num_experts must be greater than zero");
        assert!(top_k > 0, "top_k must be greater than zero");
        assert!(
            top_k <= num_experts,
            "top_k must not exceed num_experts ({top_k} > {num_experts})"
        );

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

    pub fn forward(&self, x: Tensor<B, 3>) -> RouterOutput<B> {
        let probs = softmax(self.gate.forward(x), 2);
        let (top_k_probs, top_k_indices) = probs.clone().topk_with_indices(self.top_k, 2);
        let selected_mass = top_k_probs.clone().sum_dim(2);
        let top_k_weights = top_k_probs / selected_mass;

        RouterOutput {
            probs,
            top_k_indices,
            top_k_weights,
        }
    }
}

/// Mixture-of-Experts feed-forward: routes each token to its `top_k` experts
/// and combines their outputs weighted by the renormalized router weights.
#[derive(Module, Debug)]
pub struct MoeFeedForward<B: Backend> {
    router: Router<B>,
    experts: Vec<Mlp<B>>,
}

/// Router statistics surfaced alongside the MoE output; carries what the
/// load-balancing auxiliary loss needs.
#[derive(Debug, Clone)]
pub struct MoeForwardAux<B: Backend> {
    /// Router softmax over experts: `[batch, seq, num_experts]`.
    pub probs: Tensor<B, 3>,
    /// Selected expert ids per token: `[batch, seq, top_k]`.
    pub top_k_indices: Tensor<B, 3, Int>,
}

impl<B: Backend> MoeFeedForward<B> {
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

        Self { router, experts }
    }

    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    pub fn top_k(&self) -> usize {
        self.router.top_k()
    }

    /// Dense-compute reference dispatch: every expert runs on every token and
    /// the outputs are combined with per-token weights that are zero for
    /// unselected experts. Sparse gather/scatter dispatch is a later
    /// optimization once benchmarks justify it.
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, MoeForwardAux<B>) {
        let routing = self.router.forward(x.clone());
        let [batch_size, seq_len, _] = x.shape().dims();

        let combine_weights = Tensor::zeros([batch_size, seq_len, self.num_experts()], &x.device())
            .scatter(
                2,
                routing.top_k_indices.clone(),
                routing.top_k_weights,
                IndexingUpdateOp::Add,
            );

        let mut output = x.zeros_like();
        for (expert_index, expert) in self.experts.iter().enumerate() {
            let weight = combine_weights.clone().slice([
                0..batch_size,
                0..seq_len,
                expert_index..expert_index + 1,
            ]);
            output = output + expert.forward(x.clone()) * weight;
        }

        (
            output,
            MoeForwardAux {
                probs: routing.probs,
                top_k_indices: routing.top_k_indices,
            },
        )
    }
}

impl<B: Backend> FeedForward<B> for MoeFeedForward<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (output, _aux) = MoeFeedForward::forward(self, x);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::module::Param;

    type TestBackend = NdArray<f32, i64>;

    fn assert_close(expected: &[f32], actual: &[f32], tolerance: f32) {
        assert_eq!(expected.len(), actual.len(), "length mismatch");
        for (index, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (e - a).abs() <= tolerance,
                "values differ at index {index}: expected {e}, actual {a}"
            );
        }
    }

    #[test]
    fn router_returns_documented_output_shapes() {
        let device = NdArrayDevice::Cpu;
        let router = Router::<TestBackend>::new(8, 4, 2, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 4, 8], &device);

        let output = router.forward(input);

        assert_eq!([2, 4, 4], output.probs.shape().dims());
        assert_eq!([2, 4, 2], output.top_k_indices.shape().dims());
        assert_eq!([2, 4, 2], output.top_k_weights.shape().dims());
    }

    #[test]
    fn router_probs_sum_to_one_per_token() {
        let device = NdArrayDevice::Cpu;
        let router = Router::<TestBackend>::new(8, 4, 2, &device);
        let input = Tensor::<TestBackend, 3>::random(
            [2, 4, 8],
            burn::tensor::Distribution::Default,
            &device,
        );

        let sums = router
            .forward(input)
            .probs
            .sum_dim(2)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_close(&vec![1.0; sums.len()], &sums, 1e-5);
    }

    #[test]
    fn router_top_k_weights_are_non_negative_and_sum_to_one_per_token() {
        let device = NdArrayDevice::Cpu;
        let router = Router::<TestBackend>::new(8, 4, 2, &device);
        let input = Tensor::<TestBackend, 3>::random(
            [2, 4, 8],
            burn::tensor::Distribution::Default,
            &device,
        );

        let weights = router.forward(input).top_k_weights;
        let values = weights.clone().into_data().to_vec::<f32>().unwrap();
        let sums = weights.sum_dim(2).into_data().to_vec::<f32>().unwrap();

        assert!(values.iter().all(|&w| w >= 0.0), "weights: {values:?}");
        assert_close(&vec![1.0; sums.len()], &sums, 1e-5);
    }

    #[test]
    fn router_top_k_indices_are_unique_per_token_and_in_range() {
        let device = NdArrayDevice::Cpu;
        let num_experts = 4;
        let top_k = 3;
        let router = Router::<TestBackend>::new(8, num_experts, top_k, &device);
        let input = Tensor::<TestBackend, 3>::random(
            [2, 4, 8],
            burn::tensor::Distribution::Default,
            &device,
        );

        let indices = router
            .forward(input)
            .top_k_indices
            .into_data()
            .to_vec::<i64>()
            .unwrap();

        for token_indices in indices.chunks(top_k) {
            let mut seen = std::collections::HashSet::new();
            for &index in token_indices {
                assert!(index >= 0 && (index as usize) < num_experts);
                assert!(seen.insert(index), "duplicate expert in {token_indices:?}");
            }
        }
    }

    #[test]
    fn router_routes_token_to_favored_expert_first() {
        let device = NdArrayDevice::Cpu;
        let mut router = Router::<TestBackend>::new(4, 4, 2, &device);
        // Gate = 5 * identity: a one-hot token activation produces gate logits
        // that favor the matching expert.
        let identity = Tensor::<TestBackend, 2>::from_data(
            [
                [5.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],
            ],
            &device,
        );
        router.gate.weight = Param::from_tensor(identity);
        router.gate.bias = Some(Param::from_tensor(Tensor::zeros([4], &device)));
        let input = Tensor::<TestBackend, 3>::from_data([[[0.0, 0.0, 1.0, 0.0]]], &device);

        let output = router.forward(input);
        let indices = output.top_k_indices.into_data().to_vec::<i64>().unwrap();
        let weights = output.top_k_weights.into_data().to_vec::<f32>().unwrap();

        assert_eq!(2, indices[0], "expert 2 should be the top choice");
        assert!(
            weights[0] > weights[1],
            "top choice should carry the larger weight: {weights:?}"
        );
    }

    #[test]
    #[should_panic(expected = "num_experts must be greater than zero")]
    fn router_rejects_zero_experts() {
        let device = NdArrayDevice::Cpu;
        let _router = Router::<TestBackend>::new(8, 0, 1, &device);
    }

    #[test]
    #[should_panic(expected = "top_k must be greater than zero")]
    fn router_rejects_zero_top_k() {
        let device = NdArrayDevice::Cpu;
        let _router = Router::<TestBackend>::new(8, 4, 0, &device);
    }

    #[test]
    #[should_panic(expected = "top_k must not exceed num_experts")]
    fn router_rejects_top_k_larger_than_num_experts() {
        let device = NdArrayDevice::Cpu;
        let _router = Router::<TestBackend>::new(8, 4, 5, &device);
    }

    #[test]
    fn moe_feed_forward_returns_model_dim_for_each_token_position() {
        let device = NdArrayDevice::Cpu;
        let moe = MoeFeedForward::<TestBackend>::new(8, 16, 4, 2, &device);
        let input = Tensor::<TestBackend, 3>::random(
            [2, 3, 8],
            burn::tensor::Distribution::Default,
            &device,
        );

        let (output, aux) = moe.forward(input);

        assert_eq!([2, 3, 8], output.shape().dims());
        assert_eq!([2, 3, 4], aux.probs.shape().dims());
        assert_eq!([2, 3, 2], aux.top_k_indices.shape().dims());
    }

    #[test]
    fn moe_with_single_expert_matches_plain_mlp() {
        let device = NdArrayDevice::Cpu;
        let moe = MoeFeedForward::<TestBackend>::new(8, 16, 1, 1, &device);
        let input = Tensor::<TestBackend, 3>::random(
            [2, 3, 8],
            burn::tensor::Distribution::Default,
            &device,
        );

        // Forward first: Burn params initialize lazily, and cloning an
        // unmaterialized module would re-run the random initializer.
        let (moe_output, _aux) = moe.forward(input.clone());
        let mlp = moe.experts[0].clone();
        let expected = mlp.forward(input).into_data().to_vec::<f32>().unwrap();
        let actual = moe_output.into_data().to_vec::<f32>().unwrap();

        assert_close(&expected, &actual, 1e-6);
    }

    #[test]
    fn moe_with_full_top_k_matches_probability_weighted_mixture() {
        let device = NdArrayDevice::Cpu;
        let num_experts = 3;
        let moe = MoeFeedForward::<TestBackend>::new(8, 16, num_experts, num_experts, &device);
        let input = Tensor::<TestBackend, 3>::random(
            [2, 3, 8],
            burn::tensor::Distribution::Default,
            &device,
        );

        let (moe_output, aux) = moe.forward(input.clone());
        let mut expected = input.zeros_like();
        for (expert_index, expert) in moe.experts.iter().enumerate() {
            let prob = aux
                .probs
                .clone()
                .slice([0..2, 0..3, expert_index..expert_index + 1]);
            expected = expected + expert.forward(input.clone()) * prob;
        }

        assert_close(
            &expected.into_data().to_vec::<f32>().unwrap(),
            &moe_output.into_data().to_vec::<f32>().unwrap(),
            1e-5,
        );
    }

    #[test]
    fn moe_backward_reaches_every_expert_and_the_router_gate() {
        type AutodiffBackend = Autodiff<NdArray<f32, i64>>;
        let device = NdArrayDevice::Cpu;
        let moe = MoeFeedForward::<AutodiffBackend>::new(4, 8, 3, 2, &device);
        let input = Tensor::<AutodiffBackend, 3>::random(
            [2, 3, 4],
            burn::tensor::Distribution::Default,
            &device,
        );

        let (output, _aux) = moe.forward(input);
        let gradients = output.sum().backward();

        for (expert_index, expert) in moe.experts.iter().enumerate() {
            assert!(
                expert.fc1.weight.grad(&gradients).is_some(),
                "expert {expert_index} fc1 should receive gradients"
            );
            assert!(
                expert.fc2.weight.grad(&gradients).is_some(),
                "expert {expert_index} fc2 should receive gradients"
            );
        }
        assert!(
            moe.router.gate.weight.grad(&gradients).is_some(),
            "router gate should receive gradients"
        );
    }

    #[test]
    fn moe_feed_forward_trait_forward_matches_inherent_forward_output() {
        let device = NdArrayDevice::Cpu;
        let moe = MoeFeedForward::<TestBackend>::new(8, 16, 4, 2, &device);
        let input = Tensor::<TestBackend, 3>::random(
            [2, 3, 8],
            burn::tensor::Distribution::Default,
            &device,
        );

        let (inherent, _aux) = moe.forward(input.clone());
        let via_trait = FeedForward::forward(&moe, input);

        assert_eq!(
            inherent.into_data().to_vec::<f32>().unwrap(),
            via_trait.into_data().to_vec::<f32>().unwrap()
        );
    }
}
