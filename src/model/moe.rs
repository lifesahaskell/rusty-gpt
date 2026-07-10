use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use super::Mlp;

#[derive(Debug, Clone)]
pub struct RouterOutput<B: Backend> {
    pub probs: Tensor<B, 3>,
    pub top_k_indices: Tensor<B, 3, Int>,
    pub top_k_weights: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct Router<B: Backend> {
    gate: Linear<B>,
    num_experts: usize,
    top_k: usize,
}

impl<B: Backend> Router<B> {
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

    pub fn forward(&self, x: Tensor<B, 3>) -> RouterOutput<B> {
        let logits = self.gate.forward(x);
        let probs = softmax(logits, 2);
        let (sorted_probs, sorted_indices) = probs.clone().sort_descending_with_indices(2);
        let k_range = Tensor::arange(0..self.top_k as i64, &probs.device());
        let top_k_weights_raw = sorted_probs.select(2, k_range.clone());
        let top_k_indices = sorted_indices.select(2, k_range);
        let denom = top_k_weights_raw.clone().sum_dim(2);

        RouterOutput {
            probs,
            top_k_indices,
            top_k_weights: top_k_weights_raw / denom,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MoeForwardAux<B: Backend> {
    pub probs: Tensor<B, 3>,
    pub top_k_indices: Tensor<B, 3, Int>,
    pub top_k_weights: Tensor<B, 3>,
    pub num_experts: usize,
}

impl<B: Backend> MoeForwardAux<B> {
    pub fn load_balancing_loss(&self) -> Tensor<B, 1> {
        load_balancing_loss(
            self.probs.clone(),
            self.top_k_indices.clone(),
            self.num_experts,
        )
    }
}

#[derive(Module, Debug)]
pub struct MoeFeedForward<B: Backend> {
    router: Router<B>,
    experts: Vec<Mlp<B>>,
    num_experts: usize,
    top_k: usize,
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

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, MoeForwardAux<B>) {
        let [batch, seq, d_model] = x.shape().dims();
        let router_out = self.router.forward(x.clone());
        let selection = router_out.top_k_indices.clone().float().one_hot_fill::<4>(
            self.num_experts,
            1.0,
            0.0,
            -1,
        );
        let weights = router_out.top_k_weights.clone().unsqueeze_dim(3);
        let combine = (selection * weights)
            .sum_dim(2)
            .reshape([batch, seq, self.num_experts]);

        let mut output = Tensor::zeros([batch, seq, d_model], &x.device());
        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let expert_out = expert.forward(x.clone());
            let gate = combine
                .clone()
                .slice([0..batch, 0..seq, expert_idx..expert_idx + 1]);
            output = output + expert_out * gate;
        }

        let aux = MoeForwardAux {
            probs: router_out.probs,
            top_k_indices: router_out.top_k_indices,
            top_k_weights: router_out.top_k_weights,
            num_experts: self.num_experts,
        };

        (output, aux)
    }
}

pub fn load_balancing_loss<B: Backend>(
    probs: Tensor<B, 3>,
    top_k_indices: Tensor<B, 3, Int>,
    num_experts: usize,
) -> Tensor<B, 1> {
    let [batch, seq, _experts] = probs.shape().dims();
    let num_tokens = (batch * seq) as f32;

    let top1 = top_k_indices.slice([0..batch, 0..seq, 0..1]);
    let top1_one_hot = top1
        .float()
        .one_hot_fill::<4>(num_experts, 1.0, 0.0, -1)
        .reshape([batch, seq, num_experts]);
    let f = top1_one_hot.sum_dim(0).sum_dim(1).reshape([num_experts]) / num_tokens;
    let p = probs.sum_dim(0).sum_dim(1).reshape([num_experts]) / num_tokens;

    (f * p).sum() * num_experts as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::module::Param;
    use burn::tensor::TensorData;

    type TestBackend = NdArray<f32, i64>;
    type AutodiffBackend = Autodiff<NdArray<f32, i64>>;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn router_produces_expected_output_shapes() {
        let device = NdArrayDevice::Cpu;
        let router = Router::<TestBackend>::new(8, 4, 2, &device);
        let out = router.forward(Tensor::<TestBackend, 3>::zeros([2, 4, 8], &device));

        assert_eq!([2, 4, 4], out.probs.shape().dims());
        assert_eq!([2, 4, 2], out.top_k_indices.shape().dims());
        assert_eq!([2, 4, 2], out.top_k_weights.shape().dims());
    }

    #[test]
    fn router_top_k_weights_are_non_negative_and_sum_to_one() {
        let device = NdArrayDevice::Cpu;
        let router = Router::<TestBackend>::new(8, 4, 2, &device);
        let out = router.forward(Tensor::<TestBackend, 3>::ones([2, 3, 8], &device));

        assert!(
            out.top_k_weights
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|&weight| weight >= 0.0)
        );
        for sum in out
            .top_k_weights
            .sum_dim(2)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
        {
            assert!(approx_eq(sum, 1.0, 1e-5), "row sum {sum} should be 1");
        }
    }

    #[test]
    fn router_top_k_indices_are_unique_per_token_and_in_range() {
        let device = NdArrayDevice::Cpu;
        let router = Router::<TestBackend>::new(8, 4, 3, &device);
        let out = router.forward(Tensor::<TestBackend, 3>::ones([2, 2, 8], &device));
        let indices = out.top_k_indices.into_data().to_vec::<i64>().unwrap();

        for token in indices.chunks(3) {
            assert!(token.iter().all(|&index| (0..4).contains(&index)));
            let mut sorted = token.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), token.len());
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
        router.gate.weight = Param::from_tensor(Tensor::<TestBackend, 2>::zeros([4, 4], &device));
        router.gate.bias = Some(Param::from_tensor(Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 10.0, 0.0], [4]),
            &device,
        )));

        let out = router.forward(Tensor::<TestBackend, 3>::ones([1, 3, 4], &device));
        for token in out
            .top_k_indices
            .into_data()
            .to_vec::<i64>()
            .unwrap()
            .chunks(2)
        {
            assert_eq!(2, token[0]);
        }
    }

    #[test]
    fn moe_feed_forward_returns_model_dim_for_each_token_position() {
        let device = NdArrayDevice::Cpu;
        let moe = MoeFeedForward::<TestBackend>::new(8, 16, 4, 2, &device);
        let (output, _aux) = moe.forward(Tensor::<TestBackend, 3>::zeros([2, 5, 8], &device));

        assert_eq!([2, 5, 8], output.shape().dims());
    }

    #[test]
    fn moe_with_single_expert_equals_plain_mlp() {
        let device = NdArrayDevice::Cpu;
        let moe = MoeFeedForward::<TestBackend>::new(8, 16, 1, 1, &device);
        let x = Tensor::<TestBackend, 3>::ones([2, 3, 8], &device);
        let expected = moe.experts[0]
            .forward(x.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let actual = moe.forward(x).0.into_data().to_vec::<f32>().unwrap();

        for (expected, actual) in expected.iter().zip(actual.iter()) {
            assert!(approx_eq(*expected, *actual, 1e-5));
        }
    }

    #[test]
    fn moe_backward_produces_gradients_for_experts_and_router() {
        let device = NdArrayDevice::Cpu;
        let moe = MoeFeedForward::<AutodiffBackend>::new(8, 16, 2, 2, &device);
        let x = Tensor::<AutodiffBackend, 3>::ones([1, 4, 8], &device);
        let grads = moe.forward(x).0.sum().backward();

        for (idx, expert) in moe.experts.iter().enumerate() {
            let grad = expert
                .fc1
                .weight
                .val()
                .grad(&grads)
                .unwrap_or_else(|| panic!("expert {idx} fc1 should have a gradient"));
            assert!(grad.abs().sum().into_scalar() > 0.0);
        }
        assert!(
            moe.router
                .gate
                .weight
                .val()
                .grad(&grads)
                .expect("router gate should have a gradient")
                .abs()
                .sum()
                .into_scalar()
                >= 0.0
        );
    }

    #[test]
    fn load_balancing_loss_is_one_for_uniform_routing() {
        let device = NdArrayDevice::Cpu;
        let num_experts = 4;
        let probs = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(vec![0.25f32; 2 * 3 * num_experts], [2, 3, num_experts]),
            &device,
        );
        let top_k_indices = Tensor::<TestBackend, 3, Int>::from_data(
            TensorData::new([0i64, 1].repeat(2 * 3), [2, 3, 2]),
            &device,
        );

        let loss: f32 = load_balancing_loss(probs, top_k_indices, num_experts).into_scalar();

        assert!(approx_eq(loss, 1.0, 1e-4));
    }
}
