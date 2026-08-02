use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

use super::attention::{LayerCache, MultiHeadAttention};
use super::moe::{MoeFeedForward, MoeForwardAux};

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub(super) fc1: Linear<B>,
    pub(super) fc2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(d_model: usize, d_ff: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(d_model, d_ff).init(device),
            fc2: LinearConfig::new(d_ff, d_model).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = gelu(x);
        self.fc2.forward(x)
    }
}

#[derive(Module, Debug)]
pub enum FeedForward<B: Backend> {
    Dense(Mlp<B>),
    Moe(MoeFeedForward<B>),
}

impl<B: Backend> FeedForward<B> {
    pub fn dense(d_model: usize, device: &B::Device) -> Self {
        Self::Dense(Mlp::new(d_model, 4 * d_model, device))
    }

    pub fn moe(d_model: usize, num_experts: usize, top_k: usize, device: &B::Device) -> Self {
        Self::Moe(MoeFeedForward::new(
            d_model,
            4 * d_model,
            num_experts,
            top_k,
            device,
        ))
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_aux(x).0
    }

    pub fn forward_with_aux(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Option<MoeForwardAux<B>>) {
        match self {
            Self::Dense(mlp) => (mlp.forward(x), None),
            Self::Moe(moe) => {
                let (output, aux) = moe.forward(x);
                (output, Some(aux))
            }
        }
    }

    pub fn num_experts(&self) -> usize {
        match self {
            Self::Dense(_) => 0,
            Self::Moe(moe) => moe.num_experts(),
        }
    }

    pub fn top_k(&self) -> usize {
        match self {
            Self::Dense(_) => 0,
            Self::Moe(moe) => moe.top_k(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    ln1: LayerNorm<B>,
    pub(super) attn: MultiHeadAttention<B>,
    ln2: LayerNorm<B>,
    mlp: FeedForward<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(d_model: usize, num_heads: usize, device: &B::Device) -> Self {
        Self {
            ln1: LayerNormConfig::new(d_model).init(device),
            attn: MultiHeadAttention::new(d_model, num_heads, device),
            ln2: LayerNormConfig::new(d_model).init(device),
            mlp: FeedForward::dense(d_model, device),
        }
    }

    pub fn new_with_feed_forward(
        d_model: usize,
        num_heads: usize,
        feed_forward: FeedForward<B>,
        device: &B::Device,
    ) -> Self {
        Self {
            ln1: LayerNormConfig::new(d_model).init(device),
            attn: MultiHeadAttention::new(d_model, num_heads, device),
            ln2: LayerNormConfig::new(d_model).init(device),
            mlp: feed_forward,
        }
    }

    fn feed_forward_residual(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Option<MoeForwardAux<B>>) {
        let (ff_output, aux) = self.mlp.forward_with_aux(self.ln2.forward(x.clone()));
        (x + ff_output, aux)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.ln1.forward(x));
        self.feed_forward_residual(x).0
    }

    pub fn forward_with_mask(&self, x: Tensor<B, 3>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        self.forward_with_mask_and_aux(x, mask).0
    }

    pub fn forward_with_mask_and_aux(
        &self,
        x: Tensor<B, 3>,
        mask: Tensor<B, 4, Bool>,
    ) -> (Tensor<B, 3>, Option<MoeForwardAux<B>>) {
        let x = x.clone() + self.attn.forward_with_mask(self.ln1.forward(x), mask);
        self.feed_forward_residual(x)
    }

    pub fn forward_with_weights(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let (attn_output, attn_weights) =
            self.attn.forward_with_weights(self.ln1.forward(x.clone()));
        let x = self.feed_forward_residual(x + attn_output).0;
        (x, attn_weights)
    }

    pub fn forward_with_weights_and_mask(
        &self,
        x: Tensor<B, 3>,
        mask: Tensor<B, 4, Bool>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let (attn_output, attn_weights) = self
            .attn
            .forward_with_weights_and_mask(self.ln1.forward(x.clone()), mask);
        let x = self.feed_forward_residual(x + attn_output).0;
        (x, attn_weights)
    }

    pub fn forward_with_weights_mask_and_aux(
        &self,
        x: Tensor<B, 3>,
        mask: Tensor<B, 4, Bool>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Option<MoeForwardAux<B>>) {
        let (attn_output, attn_weights) = self
            .attn
            .forward_with_weights_and_mask(self.ln1.forward(x.clone()), mask);
        let (x, aux) = self.feed_forward_residual(x + attn_output);
        (x, attn_weights, aux)
    }

    pub fn forward_with_attention(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 4>) {
        self.forward_with_weights(x)
    }

    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        cache: Option<LayerCache<B>>,
    ) -> (Tensor<B, 3>, LayerCache<B>) {
        let (attn_output, cache) = self
            .attn
            .forward_with_cache(self.ln1.forward(x.clone()), cache);
        let x = self.feed_forward_residual(x + attn_output).0;
        (x, cache)
    }

    pub fn num_heads(&self) -> usize {
        self.attn.num_heads
    }

    pub fn num_experts(&self) -> usize {
        self.mlp.num_experts()
    }

    pub fn moe_top_k(&self) -> usize {
        self.mlp.top_k()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::Tensor;

    #[test]
    fn mlp_initializes_feed_forward_projection_shapes() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;

        let mlp = Mlp::<TestBackend>::new(8, 32, &device);

        assert_eq!([8, 32], mlp.fc1.weight.shape().dims());
        assert_eq!([32, 8], mlp.fc2.weight.shape().dims());
        assert_eq!([32], mlp.fc1.bias.as_ref().unwrap().shape().dims());
        assert_eq!([8], mlp.fc2.bias.as_ref().unwrap().shape().dims());
    }

    #[test]
    fn mlp_returns_model_dim_for_each_token_position() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let mlp = Mlp::<TestBackend>::new(8, 32, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);

        let output = mlp.forward(input);

        assert_eq!([2, 3, 8], output.shape().dims());
    }

    #[test]
    fn block_initializes_attention_norm_and_mlp_shapes() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;

        let block = Block::<TestBackend>::new(8, 2, &device);

        assert_eq!([8], block.ln1.gamma.shape().dims());
        assert_eq!([8], block.ln1.beta.as_ref().unwrap().shape().dims());
        assert_eq!([8], block.ln2.gamma.shape().dims());
        assert_eq!([8], block.ln2.beta.as_ref().unwrap().shape().dims());
        assert_eq!(2, block.attn.num_heads);
        assert_eq!(4, block.attn.head_dim);
        let FeedForward::Dense(mlp) = &block.mlp else {
            panic!("Block::new should build a dense feed-forward");
        };
        assert_eq!([8, 32], mlp.fc1.weight.shape().dims());
        assert_eq!([32, 8], mlp.fc2.weight.shape().dims());
    }

    #[test]
    fn block_returns_model_dim_for_each_token_position() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let block = Block::<TestBackend>::new(8, 2, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);

        let output = block.forward(input);

        assert_eq!([2, 3, 8], output.shape().dims());
    }

    #[test]
    fn block_can_return_attention_weights() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let block = Block::<TestBackend>::new(8, 2, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);

        let (output, attention) = block.forward_with_attention(input);

        assert_eq!([2, 3, 8], output.shape().dims());
        assert_eq!([2, 2, 3, 3], attention.shape().dims());
    }
}
