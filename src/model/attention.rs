use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor, TensorData};

pub struct LayerCache<B: Backend> {
    pub keys: Tensor<B, 4>,   // [B, NUM_HEADS, T_cached, HEAD_DIM]
    pub values: Tensor<B, 4>, // [B, NUM_HEADS, T_cached, HEAD_DIM]
}

pub struct KvCache<B: Backend> {
    pub layers: Vec<Option<LayerCache<B>>>,
}

impl<B: Backend> KvCache<B> {
    pub fn new(layers: Vec<Option<LayerCache<B>>>) -> Self {
        Self { layers }
    }

    pub fn empty(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
        }
    }
}

#[derive(Module, Debug)]
pub struct SingleHeadAttention<B: Backend> {
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub head_dim: usize,
}

impl<B: Backend> SingleHeadAttention<B> {
    pub fn new(d_model: usize, head_dim: usize, device: &B::Device) -> Self {
        Self {
            query: LinearConfig::new(d_model, head_dim).init(device),
            key: LinearConfig::new(d_model, head_dim).init(device),
            value: LinearConfig::new(d_model, head_dim).init(device),
            head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        let scores = q.matmul(k.transpose()) / (self.head_dim as f32).sqrt();
        let [batch_size, seq_len, _] = scores.shape().dims();
        let mask = Self::causal_mask(batch_size, seq_len, &scores.device());
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);

        let attn_weights = softmax(scores, 2);
        attn_weights.matmul(v)
    }

    fn causal_mask(batch_size: usize, seq_len: usize, device: &B::Device) -> Tensor<B, 3, Bool> {
        Tensor::<B, 2, Bool>::tril_mask([seq_len, seq_len], 0, device)
            .reshape([1, seq_len, seq_len])
            .repeat_dim(0, batch_size)
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    qkv: Linear<B>,    // EMBED_DIM -> 3 * EMBED_DIM
    output: Linear<B>, // EMBED_DIM -> EMBED_DIM
    pub(super) num_heads: usize,
    pub(super) head_dim: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(d_model: usize, num_heads: usize, device: &B::Device) -> Self {
        assert!(num_heads > 0, "num_heads must be greater than zero");
        assert_eq!(
            d_model % num_heads,
            0,
            "d_model must be divisible by num_heads"
        );

        Self {
            qkv: LinearConfig::new(d_model, 3 * d_model).init(device),
            output: LinearConfig::new(d_model, d_model).init(device),
            num_heads,
            head_dim: d_model / num_heads,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (output, _attn_weights) = self.forward_inner(x, false);
        output
    }

    pub fn forward_with_mask(&self, x: Tensor<B, 3>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let (output, _attn_weights) = self.forward_inner_with_mask(x, mask, false);
        output
    }

    pub fn forward_with_weights(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let (output, attn_weights) = self.forward_inner(x, true);
        (
            output,
            attn_weights.expect("attention weights should be returned when requested"),
        )
    }

    pub fn forward_with_weights_and_mask(
        &self,
        x: Tensor<B, 3>,
        mask: Tensor<B, 4, Bool>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let (output, attn_weights) = self.forward_inner_with_mask(x, mask, true);
        (
            output,
            attn_weights.expect("attention weights should be returned when requested"),
        )
    }

    pub fn forward_with_cache(
        &self,
        x: Tensor<B, 3>,
        cache: Option<LayerCache<B>>,
    ) -> (Tensor<B, 3>, LayerCache<B>) {
        let embed_dim = self.num_heads * self.head_dim;
        let (q, k, v) = self.project_qkv(x);
        let [batch_size, _, new_seq_len, _] = q.shape().dims();

        let (full_k, full_v) = match cache {
            Some(cache) => (
                Tensor::cat(vec![cache.keys, k], 2),
                Tensor::cat(vec![cache.values, v], 2),
            ),
            None => (k, v),
        };

        let [_batch_size, _num_heads, total_seq_len, _head_dim] = full_k.shape().dims();
        let scores = q.matmul(full_k.clone().swap_dims(2, 3)) / (self.head_dim as f32).sqrt();
        let scores = if new_seq_len > 1 || new_seq_len == total_seq_len {
            let mask = Self::causal_cache_mask(
                batch_size,
                self.num_heads,
                new_seq_len,
                total_seq_len,
                &scores.device(),
            );
            scores.mask_fill(mask, f32::NEG_INFINITY)
        } else {
            scores
        };
        let attn_weights = softmax(scores, 3);
        let context = attn_weights
            .matmul(full_v.clone())
            .swap_dims(1, 2)
            .reshape([batch_size, new_seq_len, embed_dim]);

        (
            self.output.forward(context),
            LayerCache {
                keys: full_k,
                values: full_v,
            },
        )
    }

    fn forward_inner(
        &self,
        x: Tensor<B, 3>,
        return_weights: bool,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 4>>) {
        let [batch_size, seq_len, _] = x.shape().dims();
        let mask = Self::causal_mask(batch_size, self.num_heads, seq_len, &x.device());
        self.forward_inner_with_mask(x, mask, return_weights)
    }

    fn forward_inner_with_mask(
        &self,
        x: Tensor<B, 3>,
        mask: Tensor<B, 4, Bool>,
        return_weights: bool,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 4>>) {
        let embed_dim = self.num_heads * self.head_dim;
        let (q, k, v) = self.project_qkv(x);
        let [batch_size, _, seq_len, _] = q.shape().dims();

        let scores = q.matmul(k.swap_dims(2, 3)) / (self.head_dim as f32).sqrt();
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        let attn_weights = softmax(scores, 3);
        let (context, attn_weights) = if return_weights {
            (
                attn_weights
                    .clone()
                    .matmul(v)
                    .swap_dims(1, 2)
                    .reshape([batch_size, seq_len, embed_dim]),
                Some(attn_weights),
            )
        } else {
            (
                attn_weights
                    .matmul(v)
                    .swap_dims(1, 2)
                    .reshape([batch_size, seq_len, embed_dim]),
                None,
            )
        };

        (self.output.forward(context), attn_weights)
    }

    fn project_qkv(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let qkv = self.qkv.forward(x);
        let [batch_size, seq_len, _] = qkv.shape().dims();
        let qkv = qkv.reshape([batch_size, seq_len, 3, self.num_heads, self.head_dim]);
        let q = qkv
            .clone()
            .slice([
                0..batch_size,
                0..seq_len,
                0..1,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = qkv
            .clone()
            .slice([
                0..batch_size,
                0..seq_len,
                1..2,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = qkv
            .slice([
                0..batch_size,
                0..seq_len,
                2..3,
                0..self.num_heads,
                0..self.head_dim,
            ])
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        (q, k, v)
    }

    pub(super) fn causal_mask(
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 4, Bool> {
        Tensor::<B, 2, Bool>::tril_mask([seq_len, seq_len], 0, device)
            .reshape([1, 1, seq_len, seq_len])
            .repeat_dim(0, batch_size)
            .repeat_dim(1, num_heads)
    }

    fn causal_cache_mask(
        batch_size: usize,
        num_heads: usize,
        new_seq_len: usize,
        total_seq_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 4, Bool> {
        let cached_seq_len = total_seq_len - new_seq_len;
        let mut mask = Vec::with_capacity(batch_size * num_heads * new_seq_len * total_seq_len);

        for _batch in 0..batch_size {
            for _head in 0..num_heads {
                for query_pos in 0..new_seq_len {
                    let max_visible_key = cached_seq_len + query_pos;
                    for key_pos in 0..total_seq_len {
                        mask.push(key_pos > max_visible_key);
                    }
                }
            }
        }

        Tensor::<B, 4, Bool>::from_data(
            TensorData::new(mask, [batch_size, num_heads, new_seq_len, total_seq_len]),
            device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::Tensor;

    #[test]
    fn single_head_attention_returns_head_dim_for_each_token_position() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let attention = SingleHeadAttention::<TestBackend>::new(8, 4, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);

        let output = attention.forward(input);

        assert_eq!(4, attention.head_dim);
        assert_eq!([2, 3, 4], output.shape().dims());
    }

    #[test]
    fn single_head_attention_causal_mask_blocks_future_positions() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;

        let mask = SingleHeadAttention::<TestBackend>::causal_mask(1, 3, &device);

        assert_eq!([1, 3, 3], mask.shape().dims());
        assert_eq!(
            vec![false, true, true, false, false, true, false, false, false],
            mask.into_data().to_vec::<bool>().unwrap()
        );
    }

    #[test]
    fn multi_head_attention_returns_model_dim_for_each_token_position() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let attention = MultiHeadAttention::<TestBackend>::new(8, 2, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);

        let output = attention.forward(input);

        assert_eq!(2, attention.num_heads);
        assert_eq!(4, attention.head_dim);
        assert_eq!([2, 3, 8], output.shape().dims());
    }
    #[test]
    fn multi_head_attention_causal_mask_blocks_future_positions_for_each_head() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;

        let mask = MultiHeadAttention::<TestBackend>::causal_mask(1, 2, 3, &device);

        assert_eq!([1, 2, 3, 3], mask.shape().dims());
        assert_eq!(
            vec![
                false, true, true, false, false, true, false, false, false, false, true, true,
                false, false, true, false, false, false,
            ],
            mask.into_data().to_vec::<bool>().unwrap()
        );
    }

    #[test]
    fn multi_head_attention_forward_with_cache_matches_full_forward_without_cache() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let attention = MultiHeadAttention::<TestBackend>::new(8, 2, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);

        let expected = attention
            .forward(input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let (actual, cache) = attention.forward_with_cache(input, None);

        assert_eq!([2, 3, 8], actual.shape().dims());
        assert_eq!([2, 2, 3, 4], cache.keys.shape().dims());
        assert_eq!([2, 2, 3, 4], cache.values.shape().dims());
        assert_eq!(expected, actual.into_data().to_vec::<f32>().unwrap());
    }

    #[test]
    fn multi_head_attention_forward_with_cache_appends_new_keys_and_values() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let attention = MultiHeadAttention::<TestBackend>::new(8, 2, &device);
        let prompt = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);
        let next_token = Tensor::<TestBackend, 3>::ones([2, 1, 8], &device);

        let (_prompt_output, cache) = attention.forward_with_cache(prompt, None);
        let (next_output, next_cache) = attention.forward_with_cache(next_token, Some(cache));

        assert_eq!([2, 1, 8], next_output.shape().dims());
        assert_eq!([2, 2, 4, 4], next_cache.keys.shape().dims());
        assert_eq!([2, 2, 4, 4], next_cache.values.shape().dims());
    }

    #[test]
    fn multi_head_attention_causal_cache_mask_allows_cached_tokens_and_blocks_future_new_tokens() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;

        let mask = MultiHeadAttention::<TestBackend>::causal_cache_mask(1, 1, 2, 5, &device);

        assert_eq!([1, 1, 2, 5], mask.shape().dims());
        assert_eq!(
            vec![
                false, false, false, false, true, false, false, false, false, false
            ],
            mask.into_data().to_vec::<bool>().unwrap()
        );
    }

    #[test]
    #[should_panic(expected = "d_model must be divisible by num_heads")]
    fn multi_head_attention_requires_even_head_split() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;

        let _attention = MultiHeadAttention::<TestBackend>::new(7, 2, &device);
    }
}
