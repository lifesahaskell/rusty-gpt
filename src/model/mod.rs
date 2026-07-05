use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor, TensorData};

pub mod generation;
pub mod moe;
pub mod persistence;
pub mod training;

pub use generation::GenerationOptions;
pub use moe::{MoeFeedForward, MoeForwardAux, Router, RouterOutput};
pub use training::{
    TrainingLogContext, TrainingLogFormat, TrainingMetrics, TrainingOutcome, TrainingParams,
};
#[cfg(test)]
use training::{TrainingThroughput, should_log_training_step, training_progress_log_line};

#[derive(Debug, Clone, Copy)]
pub struct MiniGptConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_blocks: usize,
    pub max_position_embeddings: usize,
    pub num_heads: usize,
}

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
pub struct TrivialModel<B: Backend> {
    pub embedding: Embedding<B>,
    pub lm_head: Linear<B>,
}

impl<B: Backend> TrivialModel<B> {
    pub fn new(vocab_size: usize, d_model: usize, device: &B::Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(vocab_size, d_model).init(device),
            lm_head: LinearConfig::new(d_model, vocab_size).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embedding.forward(input); // shape: [batch_size, seq, embedding_dim]
        self.lm_head.forward(x) // shape: [batch_size, seq, vocab_size]
    }
}

#[derive(Module, Debug)]
pub struct SingleAttentionModel<B: Backend> {
    pub embedding: Embedding<B>,
    pub attention: SingleHeadAttention<B>,
    pub lm_head: Linear<B>,
}

impl<B: Backend> SingleAttentionModel<B> {
    pub fn new(vocab_size: usize, d_model: usize, head_dim: usize, device: &B::Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(vocab_size, d_model).init(device),
            attention: SingleHeadAttention::new(d_model, head_dim, device),
            lm_head: LinearConfig::new(head_dim, vocab_size).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.attention.forward(input);
        self.lm_head.forward(x)
    }

    pub fn forward_tokens(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward(self.embedding.forward(input))
    }
}

#[derive(Module, Debug)]
pub struct MultiAttentionModel<B: Backend> {
    pub embedding: Embedding<B>,
    pub attention: MultiHeadAttention<B>,
    pub lm_head: Linear<B>,
}

impl<B: Backend> MultiAttentionModel<B> {
    pub fn new(vocab_size: usize, d_model: usize, num_heads: usize, device: &B::Device) -> Self {
        Self {
            embedding: EmbeddingConfig::new(vocab_size, d_model).init(device),
            attention: MultiHeadAttention::new(d_model, num_heads, device),
            lm_head: LinearConfig::new(d_model, vocab_size).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.attention.forward(input);
        self.lm_head.forward(x)
    }

    pub fn forward_tokens(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward(self.embedding.forward(input))
    }

    pub fn forward_with_weights(&self, input: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let (x, attn_weights) = self.attention.forward_with_weights(input);
        (self.lm_head.forward(x), attn_weights)
    }

    pub fn forward_tokens_with_weights(
        &self,
        input: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        self.forward_with_weights(self.embedding.forward(input))
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
    num_heads: usize,
    head_dim: usize,
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

    fn causal_mask(
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

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
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

/// Contract for the pluggable feed-forward slot inside a transformer [`Block`].
///
/// [`Mlp`] is the dense implementation used by [`MiniGpt`];
/// [`moe::MoeFeedForward`] is the Mixture-of-Experts implementation. The slot
/// is a generic parameter rather than an enum module because Burn serializes
/// enum module records externally tagged, which would change the record tree
/// of every existing dense checkpoint; with a generic defaulted to `Mlp<B>`,
/// pre-MoE `.mpk` files keep loading unchanged.
pub trait FeedForward<B: Backend>: Module<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3>;
}

impl<B: Backend> FeedForward<B> for Mlp<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        Mlp::forward(self, x)
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend, F = Mlp<B>> {
    ln1: LayerNorm<B>,
    attn: MultiHeadAttention<B>,
    ln2: LayerNorm<B>,
    mlp: F,
}

impl<B: Backend> Block<B> {
    pub fn new(d_model: usize, num_heads: usize, device: &B::Device) -> Self {
        Self::new_with_feed_forward(
            d_model,
            num_heads,
            Mlp::new(d_model, 4 * d_model, device),
            device,
        )
    }
}

impl<B: Backend, F: FeedForward<B>> Block<B, F> {
    pub fn new_with_feed_forward(
        d_model: usize,
        num_heads: usize,
        feed_forward: F,
        device: &B::Device,
    ) -> Self {
        Self {
            ln1: LayerNormConfig::new(d_model).init(device),
            attn: MultiHeadAttention::new(d_model, num_heads, device),
            ln2: LayerNormConfig::new(d_model).init(device),
            mlp: feed_forward,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.ln1.forward(x));
        x.clone() + self.mlp.forward(self.ln2.forward(x))
    }

    pub fn forward_with_mask(&self, x: Tensor<B, 3>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward_with_mask(self.ln1.forward(x), mask);
        x.clone() + self.mlp.forward(self.ln2.forward(x))
    }

    pub fn forward_with_weights(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let (attn_output, attn_weights) =
            self.attn.forward_with_weights(self.ln1.forward(x.clone()));
        let x = x + attn_output;
        let x = x.clone() + self.mlp.forward(self.ln2.forward(x));
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
        let x = x + attn_output;
        let x = x.clone() + self.mlp.forward(self.ln2.forward(x));
        (x, attn_weights)
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
        let x = x + attn_output;
        let x = x.clone() + self.mlp.forward(self.ln2.forward(x));
        (x, cache)
    }

    pub fn num_heads(&self) -> usize {
        self.attn.num_heads
    }
}

#[derive(Module, Debug)]
pub struct MiniGpt<B: Backend> {
    token_embed: Embedding<B>,
    position_embed: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_final: LayerNorm<B>,
    lm_head: Linear<B>,
    vocab_size: usize,
    max_position_embeddings: usize,
}

impl<B: Backend> MiniGpt<B> {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_blocks: usize,
        max_position_embeddings: usize,
        num_heads: usize,
        device: &B::Device,
    ) -> Self {
        let token_embed = EmbeddingConfig::new(vocab_size, d_model).init(device);
        let position_embed = EmbeddingConfig::new(max_position_embeddings, d_model).init(device);
        let blocks = (0..num_blocks)
            .map(|_| Block::new(d_model, num_heads, device))
            .collect();
        let ln_final = LayerNormConfig::new(d_model).init(device);
        let lm_head = LinearConfig::new(d_model, vocab_size).init(device);

        Self {
            token_embed,
            position_embed,
            blocks,
            ln_final,
            lm_head,
            vocab_size,
            max_position_embeddings,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    pub fn num_heads(&self) -> usize {
        self.blocks.first().map(Block::num_heads).unwrap_or(0)
    }

    pub fn d_model(&self) -> usize {
        self.token_embed.weight.shape().dims::<2>()[1]
    }

    pub fn block_size(&self) -> usize {
        self.max_position_embeddings
    }

    pub fn forward_tokens(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_b, t] = tokens.dims();
        let positions = Tensor::arange(0..t as i64, &tokens.device()).unsqueeze();

        let tok = self.token_embed.forward(tokens);
        let pos = self.position_embed.forward(positions);
        self.forward(tok + pos)
    }

    pub fn forward_tokens_with_attention(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 4>>) {
        let [_b, t] = tokens.dims();
        let positions = Tensor::arange(0..t as i64, &tokens.device()).unsqueeze();

        let tok = self.token_embed.forward(tokens);
        let pos = self.position_embed.forward(positions);
        self.forward_with_attention(tok + pos)
    }

    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        if let Some(first_block) = self.blocks.first() {
            let [batch_size, seq_len, _] = x.shape().dims();
            let mask = MultiHeadAttention::<B>::causal_mask(
                batch_size,
                first_block.num_heads(),
                seq_len,
                &x.device(),
            );
            for block in &self.blocks {
                x = block.forward_with_mask(x, mask.clone());
            }
        }

        let x = self.ln_final.forward(x);
        self.lm_head.forward(x)
    }

    pub fn forward_with_cache(
        &self,
        tokens: Tensor<B, 2, Int>,
        mut cache: KvCache<B>,
    ) -> (Tensor<B, 3>, KvCache<B>) {
        assert_eq!(
            self.blocks.len(),
            cache.layers.len(),
            "KV cache must have one entry per transformer block"
        );

        let [_batch_size, seq_len] = tokens.dims();
        let cache_len = cache
            .layers
            .first()
            .and_then(|layer| layer.as_ref())
            .map(|c| c.keys.dims()[2])
            .unwrap_or(0);

        let positions = Tensor::arange(
            cache_len as i64..(cache_len + seq_len) as i64,
            &tokens.device(),
        )
        .unsqueeze();
        let tok = self.token_embed.forward(tokens);
        let pos = self.position_embed.forward(positions);
        let mut x = tok + pos;

        for (i, block) in self.blocks.iter().enumerate() {
            let layer_cache = cache.layers[i].take();
            let (new_x, new_layer_cache) = block.forward_with_cache(x, layer_cache);
            x = new_x;
            cache.layers[i] = Some(new_layer_cache);
        }

        let x = self.ln_final.forward(x);
        let logits = self.lm_head.forward(x);
        (logits, cache)
    }

    pub fn forward_with_attention(&self, mut x: Tensor<B, 3>) -> (Tensor<B, 3>, Vec<Tensor<B, 4>>) {
        let mut attentions = Vec::with_capacity(self.blocks.len());
        if let Some(first_block) = self.blocks.first() {
            let [batch_size, seq_len, _] = x.shape().dims();
            let mask = MultiHeadAttention::<B>::causal_mask(
                batch_size,
                first_block.num_heads(),
                seq_len,
                &x.device(),
            );
            for block in &self.blocks {
                let (next_x, attention) = block.forward_with_weights_and_mask(x, mask.clone());
                x = next_x;
                attentions.push(attention);
            }
        }

        let x = self.ln_final.forward(x);
        (self.lm_head.forward(x), attentions)
    }

    pub fn generate(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        device: &B::Device,
    ) -> Result<Vec<usize>, String> {
        self.generate_with_options(prompt, max_new_tokens, device, GenerationOptions::greedy())
    }

    pub fn generate_with_options(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        device: &B::Device,
        options: GenerationOptions,
    ) -> Result<Vec<usize>, String> {
        self.validate_generation_prompt(prompt)?;

        let mut output = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let context_start = output.len().saturating_sub(self.max_position_embeddings);
            let context = &output[context_start..];
            let input: Vec<i64> = context.iter().map(|&token| token as i64).collect();
            let tokens = Tensor::from_data(TensorData::new(input, [1, context.len()]), device);
            let logits = self.forward_tokens(tokens);
            let next_token = Self::select_last_token(logits, options);

            output.push(next_token);
        }

        Ok(output)
    }

    pub fn generate_cached(&self, prompt: Tensor<B, 2, Int>, max_new: usize) -> Vec<usize> {
        self.generate_cached_with_options(prompt, max_new, GenerationOptions::greedy())
    }

    pub fn generate_cached_with_options(
        &self,
        prompt: Tensor<B, 2, Int>,
        max_new: usize,
        options: GenerationOptions,
    ) -> Vec<usize> {
        if max_new == 0 {
            return Vec::new();
        }

        let device = prompt.device();
        let mut cache = KvCache::empty(self.num_layers());
        let (logits, new_cache) = self.forward_with_cache(prompt, cache);
        cache = new_cache;
        let mut next_token = Self::select_last_token(logits, options);
        let mut generated = vec![next_token];

        for _ in 1..max_new {
            let next_input = Tensor::from_data([[next_token as i64]], &device);
            let (logits, new_cache) = self.forward_with_cache(next_input, cache);
            cache = new_cache;
            next_token = Self::select_last_token(logits, options);
            generated.push(next_token);
        }

        generated
    }

    pub fn generate_with_cache(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        device: &B::Device,
    ) -> Result<Vec<usize>, String> {
        self.generate_with_cache_options(
            prompt,
            max_new_tokens,
            device,
            GenerationOptions::greedy(),
        )
    }

    pub fn generate_with_cache_options(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        device: &B::Device,
        options: GenerationOptions,
    ) -> Result<Vec<usize>, String> {
        self.validate_generation_prompt(prompt)?;

        let mut output = prompt.to_vec();
        let mut remaining = max_new_tokens;
        while remaining > 0 {
            let context_start = output.len().saturating_sub(self.max_position_embeddings);
            let context = &output[context_start..];
            let available_positions = self.max_position_embeddings.saturating_sub(context.len());
            let chunk_len = remaining.min(available_positions.max(1));
            let prompt_data: Vec<i64> = context.iter().map(|&token| token as i64).collect();
            let prompt_tensor =
                Tensor::from_data(TensorData::new(prompt_data, [1, context.len()]), device);
            let generated = self.generate_cached_with_options(prompt_tensor, chunk_len, options);

            remaining -= generated.len();
            output.extend(generated);
        }

        Ok(output)
    }

    fn validate_generation_prompt(&self, prompt: &[usize]) -> Result<(), String> {
        if prompt.is_empty() {
            return Err("prompt must contain at least one token".to_string());
        }
        if let Some(token) = prompt.iter().find(|&&token| token >= self.vocab_size) {
            return Err(format!(
                "prompt token {token} is outside the model vocab size {}",
                self.vocab_size
            ));
        }

        Ok(())
    }

    fn select_last_token(logits: Tensor<B, 3>, options: GenerationOptions) -> usize {
        let [_batch_size, seq_len, vocab_size] = logits.shape().dims();
        let logits = logits.into_data().to_vec::<f32>().unwrap();
        let last_position_start = (seq_len - 1) * vocab_size;
        let last_logits = &logits[last_position_start..last_position_start + vocab_size];

        generation::select_token_from_logits(
            last_logits,
            options,
            if options.temperature <= 0.0 {
                0.0
            } else {
                rand::random()
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::data::DataLoader;
    use crate::observability::{EventLogger, LogFormat};
    use burn::backend::Autodiff;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::{Int, Tensor};
    use std::sync::{Arc, Mutex};

    #[test]
    fn training_step_logging_uses_interval_and_always_logs_final_step() {
        assert!(should_log_training_step(0, 25, 10));
        assert!(should_log_training_step(10, 25, 10));
        assert!(should_log_training_step(24, 25, 10));
        assert!(!should_log_training_step(9, 25, 10));
    }

    #[test]
    fn training_step_logging_logs_every_step_when_total_fits_interval() {
        assert!(should_log_training_step(0, 3, 10));
        assert!(should_log_training_step(1, 3, 10));
        assert!(should_log_training_step(2, 3, 10));
    }

    #[test]
    fn training_step_logging_zero_interval_only_logs_final_step() {
        assert!(!should_log_training_step(0, 3, 0));
        assert!(!should_log_training_step(1, 3, 0));
        assert!(should_log_training_step(2, 3, 0));
    }

    #[test]
    fn training_progress_json_log_line_includes_backend_and_losses() {
        type TestBackend = NdArray<f32, i64>;
        let line = training_progress_log_line::<TestBackend>(
            TrainingLogContext {
                backend: "cuda",
                model: "minigpt",
                logger: EventLogger::stdout(TrainingLogFormat::Json),
            },
            10,
            100,
            1.25,
            2.5,
            TrainingThroughput::from_progress(11, 32, 128, 250),
        );
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();

        assert_eq!("training_progress", parsed["event"]);
        assert_eq!("cuda", parsed["backend"]);
        assert_eq!("minigpt", parsed["model"]);
        assert_eq!(10, parsed["step"]);
        assert_eq!(100, parsed["total_steps"]);
        assert_eq!(1.25, parsed["training_loss"]);
        assert_eq!(2.5, parsed["value_loss"]);
        assert_eq!(250, parsed["elapsed_ms"]);
        assert!(parsed["tokens_per_second"].as_f64().unwrap() > 0.0);
        assert!(parsed["steps_per_second"].as_f64().unwrap() > 0.0);
        assert!(parsed["step_ms_mean"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn training_run_emits_final_progress_event() {
        type TestBackend = Autodiff<NdArray<f32, i64>>;
        let device = NdArrayDevice::Cpu;
        let loader = DataLoader {
            tokens: vec![0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4],
            block_size: 2,
            batch_size: 2,
        };
        let value_loader = loader.clone();
        let lines = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&lines);
        let logger = EventLogger::with_sink(LogFormat::Json, move |line| {
            captured.lock().unwrap().push(line);
        });
        let params = TrainingParams::new(
            1e-4,
            3,
            2,
            TrainingLogContext {
                backend: "cpu",
                model: "trivial",
                logger,
            },
        );

        TrivialModel::<TestBackend>::train(&loader, &value_loader, &device, 7, 8, params).unwrap();

        let lines = lines.lock().unwrap();
        let final_progress: serde_json::Value =
            serde_json::from_str(lines.last().expect("expected progress event")).unwrap();
        assert_eq!("training_progress", final_progress["event"]);
        assert_eq!(2, final_progress["step"]);
        assert_eq!(3, final_progress["total_steps"]);
        assert!(final_progress["tokens_per_second"].as_f64().unwrap() > 0.0);
        assert!(final_progress["steps_per_second"].as_f64().unwrap() > 0.0);
        assert!(final_progress["step_ms_mean"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn trivial_model_initializes_embedding_and_lm_head_shapes() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;

        let model = TrivialModel::<TestBackend>::new(17, 8, &device);

        assert_eq!([17, 8], model.embedding.weight.shape().dims());
        assert_eq!([8, 17], model.lm_head.weight.shape().dims());
    }

    #[test]
    fn trivial_model_initializes_lm_head_bias_for_each_vocab_token() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;

        let model = TrivialModel::<TestBackend>::new(17, 8, &device);
        let bias = model
            .lm_head
            .bias
            .as_ref()
            .expect("lm_head should include a bias by default");

        assert_eq!([17], bias.shape().dims());
    }

    #[test]
    fn forward_returns_logits_for_each_token_position() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = TrivialModel::<TestBackend>::new(5, 3, &device);
        let input = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2], [3, 4, 0]], &device);

        let logits = model.forward(input);

        assert_eq!([2, 3, 5], logits.shape().dims());
    }

    #[test]
    fn single_attention_model_returns_logits_for_each_token_position() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = SingleAttentionModel::<TestBackend>::new(5, 8, 4, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);

        let logits = model.forward(input);

        assert_eq!([2, 3, 5], logits.shape().dims());
    }

    #[test]
    fn single_attention_model_embeds_token_inputs() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = SingleAttentionModel::<TestBackend>::new(5, 8, 4, &device);
        let input = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2], [3, 4, 0]], &device);

        let logits = model.forward_tokens(input);

        assert_eq!([2, 3, 5], logits.shape().dims());
    }

    #[test]
    fn multi_attention_model_returns_logits_for_each_token_position() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MultiAttentionModel::<TestBackend>::new(5, 8, 2, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);

        let logits = model.forward(input);

        assert_eq!([2, 3, 5], logits.shape().dims());
    }

    #[test]
    fn multi_attention_model_embeds_token_inputs() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MultiAttentionModel::<TestBackend>::new(5, 8, 2, &device);
        let input = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2], [3, 4, 0]], &device);

        let logits = model.forward_tokens(input);

        assert_eq!([2, 3, 5], logits.shape().dims());
    }

    #[test]
    fn multi_attention_model_can_return_attention_weights() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MultiAttentionModel::<TestBackend>::new(5, 8, 2, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);

        let (logits, weights) = model.forward_with_weights(input);

        assert_eq!([2, 3, 5], logits.shape().dims());
        assert_eq!([2, 2, 3, 3], weights.shape().dims());
    }

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
        assert_eq!([8, 32], block.mlp.fc1.weight.shape().dims());
        assert_eq!([32, 8], block.mlp.fc2.weight.shape().dims());
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

    #[test]
    fn block_built_via_new_uses_dense_mlp_feed_forward() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let block = Block::<TestBackend>::new(8, 2, &device);
        let input = Tensor::<TestBackend, 3>::random(
            [2, 3, 8],
            burn::tensor::Distribution::Default,
            &device,
        );

        // Forward first: Burn params initialize lazily, and cloning an
        // unmaterialized module would re-run the random initializer.
        let via_trait = FeedForward::forward(&block.mlp, input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        // The default feed-forward slot is a plain dense Mlp.
        let standalone: Mlp<TestBackend> = block.mlp.clone();
        let direct = standalone
            .forward(input)
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(direct, via_trait);
    }

    #[test]
    fn block_supports_moe_feed_forward_slot() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let moe = MoeFeedForward::<TestBackend>::new(8, 16, 4, 2, &device);
        let block = Block::new_with_feed_forward(8, 2, moe, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);

        let output = block.forward(input);

        assert_eq!([2, 3, 8], output.shape().dims());
    }

    #[test]
    fn minigpt_loads_checkpoints_saved_with_pre_feedforward_record_layout() {
        type TestBackend = NdArray<f32, i64>;

        // Mirror of the Block/MiniGpt struct layout before the feed-forward
        // slot became generic; saving it reproduces the record tree of
        // checkpoints written by older builds.
        #[derive(Module, Debug)]
        struct LegacyBlock<B: Backend> {
            ln1: LayerNorm<B>,
            attn: MultiHeadAttention<B>,
            ln2: LayerNorm<B>,
            mlp: Mlp<B>,
        }

        #[derive(Module, Debug)]
        struct LegacyMiniGpt<B: Backend> {
            token_embed: Embedding<B>,
            position_embed: Embedding<B>,
            blocks: Vec<LegacyBlock<B>>,
            ln_final: LayerNorm<B>,
            lm_head: Linear<B>,
            vocab_size: usize,
            max_position_embeddings: usize,
        }

        let device = NdArrayDevice::Cpu;
        let path = std::env::temp_dir().join(format!(
            "rusty-gpt-legacy-dense-record-{}",
            std::process::id()
        ));
        let saved_path = path.with_extension("mpk");

        let model = MiniGpt::<TestBackend>::new(7, 8, 2, 6, 2, &device);
        let input = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2], [3, 4, 5]], &device);
        // Materialize the lazily-initialized params before cloning them into
        // the legacy layout; cloning unmaterialized params would re-run the
        // random initializer and the two models would diverge.
        let expected = model
            .forward_tokens(input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let legacy = LegacyMiniGpt {
            token_embed: model.token_embed.clone(),
            position_embed: model.position_embed.clone(),
            blocks: model
                .blocks
                .iter()
                .map(|block| LegacyBlock {
                    ln1: block.ln1.clone(),
                    attn: block.attn.clone(),
                    ln2: block.ln2.clone(),
                    mlp: block.mlp.clone(),
                })
                .collect(),
            ln_final: model.ln_final.clone(),
            lm_head: model.lm_head.clone(),
            vocab_size: model.vocab_size,
            max_position_embeddings: model.max_position_embeddings,
        };
        persistence::save_model(legacy, &path).unwrap();

        let template = MiniGpt::<TestBackend>::new(7, 8, 2, 6, 2, &device);
        let loaded = persistence::load_model(template, &path, &device).unwrap();

        assert_eq!(
            expected,
            loaded
                .forward_tokens(input)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        );

        let _ = std::fs::remove_file(saved_path);
    }

    #[test]
    fn minigpt_initializes_embedding_block_norm_and_head_shapes() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;

        let model = MiniGpt::<TestBackend>::new(17, 8, 3, 12, 2, &device);

        assert_eq!([17, 8], model.token_embed.weight.shape().dims());
        assert_eq!([12, 8], model.position_embed.weight.shape().dims());
        assert_eq!(3, model.blocks.len());
        assert!(
            model
                .blocks
                .iter()
                .all(|block| block.attn.num_heads == 2 && block.attn.head_dim == 4)
        );
        assert_eq!([8], model.ln_final.gamma.shape().dims());
        assert_eq!([8], model.ln_final.beta.as_ref().unwrap().shape().dims());
        assert_eq!([8, 17], model.lm_head.weight.shape().dims());
        assert_eq!([17], model.lm_head.bias.as_ref().unwrap().shape().dims());
    }

    #[test]
    fn minigpt_returns_logits_for_each_token_position() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 2, 6, 2, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);

        let logits = model.forward(input);

        assert_eq!([2, 3, 7], logits.shape().dims());
    }

    #[test]
    fn minigpt_shared_mask_forward_matches_per_block_mask_forward() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 2, 6, 2, &device);
        let input = Tensor::<TestBackend, 3>::zeros([2, 3, 8], &device);

        let shared_mask_logits = model
            .forward(input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut x = input;
        for block in &model.blocks {
            x = block.forward(x);
        }
        let per_block_mask_logits = model
            .lm_head
            .forward(model.ln_final.forward(x))
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(per_block_mask_logits, shared_mask_logits);
    }

    #[test]
    fn minigpt_embeds_token_inputs() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 2, 6, 2, &device);
        let input = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2], [3, 4, 5]], &device);

        let logits = model.forward_tokens(input);

        assert_eq!([2, 3, 7], logits.shape().dims());
    }

    #[test]
    fn minigpt_can_return_attention_for_each_layer() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 2, 6, 2, &device);
        let input = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2]], &device);

        let (logits, attentions) = model.forward_tokens_with_attention(input);

        assert_eq!([1, 3, 7], logits.shape().dims());
        assert_eq!(2, attentions.len());
        assert_eq!([1, 2, 3, 3], attentions[0].shape().dims());
        assert_eq!([1, 2, 3, 3], attentions[1].shape().dims());
    }

    #[test]
    fn minigpt_forward_with_cache_matches_forward_tokens_without_cache() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 2, 6, 2, &device);
        let input = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2]], &device);

        let expected = model
            .forward_tokens(input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let (actual, cache) = model.forward_with_cache(input, KvCache::empty(model.num_layers()));

        assert_eq!([1, 3, 7], actual.shape().dims());
        assert_eq!(2, cache.layers.len());
        for layer_cache in cache.layers {
            let layer_cache = layer_cache.expect("layer cache should be populated");
            assert_eq!([1, 2, 3, 4], layer_cache.keys.shape().dims());
            assert_eq!([1, 2, 3, 4], layer_cache.values.shape().dims());
        }
        assert_eq!(expected, actual.into_data().to_vec::<f32>().unwrap());
    }

    #[test]
    fn minigpt_forward_with_cache_extends_cache_for_incremental_token() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 2, 6, 2, &device);
        let prompt = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2]], &device);
        let next_token = Tensor::<TestBackend, 2, Int>::from_data([[3]], &device);

        let (_prompt_logits, cache) =
            model.forward_with_cache(prompt, KvCache::empty(model.num_layers()));
        let (next_logits, next_cache) = model.forward_with_cache(next_token, cache);

        assert_eq!([1, 1, 7], next_logits.shape().dims());
        assert_eq!(2, next_cache.layers.len());
        for layer_cache in next_cache.layers {
            let layer_cache = layer_cache.expect("layer cache should be populated");
            assert_eq!([1, 2, 4, 4], layer_cache.keys.shape().dims());
            assert_eq!([1, 2, 4, 4], layer_cache.values.shape().dims());
        }
    }

    #[test]
    fn minigpt_forward_with_cache_incremental_logits_match_full_forward_last_token() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 2, 6, 2, &device);
        let prompt = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2]], &device);
        let next_token = Tensor::<TestBackend, 2, Int>::from_data([[3]], &device);
        let full = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2, 3]], &device);

        let (_prompt_logits, cache) =
            model.forward_with_cache(prompt, KvCache::empty(model.num_layers()));
        let (next_logits, _next_cache) = model.forward_with_cache(next_token, cache);
        let full_last_logits = model.forward_tokens(full).slice([0..1, 3..4, 0..7]);

        assert_eq!(
            full_last_logits.into_data().to_vec::<f32>().unwrap(),
            next_logits.into_data().to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn minigpt_supports_zero_transformer_blocks() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 0, 6, 2, &device);
        let input = Tensor::<TestBackend, 3>::zeros([1, 3, 8], &device);

        let logits = model.forward(input);

        assert!(model.blocks.is_empty());
        assert_eq!([1, 3, 7], logits.shape().dims());
    }

    #[test]
    fn minigpt_generate_appends_requested_number_of_tokens() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 4, 2, &device);

        let generated = model.generate(&[0, 1, 2], 3, &device).unwrap();

        assert_eq!(6, generated.len());
        assert_eq!(&[0, 1, 2], &generated[..3]);
        assert!(generated.iter().all(|&token| token < 7));
    }

    #[test]
    fn minigpt_generate_delegates_to_greedy_options() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 4, 2, &device);
        let prompt = [0, 1, 2];

        let generated = model.generate(&prompt, 3, &device).unwrap();
        let generated_with_options = model
            .generate_with_options(&prompt, 3, &device, GenerationOptions::greedy())
            .unwrap();

        assert_eq!(generated_with_options, generated);
    }

    #[test]
    fn minigpt_generate_crops_context_to_position_window() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 2, 2, &device);

        let generated = model.generate(&[0, 1, 2, 3], 1, &device).unwrap();

        assert_eq!(5, generated.len());
        assert_eq!(&[0, 1, 2, 3], &generated[..4]);
    }

    #[test]
    fn minigpt_generate_rejects_empty_prompt() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 4, 2, &device);

        let err = model
            .generate(&[], 1, &device)
            .expect_err("empty prompt should fail");

        assert!(err.contains("prompt must contain at least one token"));
    }

    #[test]
    fn minigpt_option_bearing_generation_preserves_prompt_validation() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 4, 2, &device);

        let uncached_err = model
            .generate_with_options(&[0, 7], 1, &device, GenerationOptions::greedy())
            .expect_err("out-of-vocab prompt should fail");
        let cached_err = model
            .generate_with_cache_options(&[0, 7], 1, &device, GenerationOptions::greedy())
            .expect_err("out-of-vocab prompt should fail");

        assert_eq!(uncached_err, cached_err);
        assert!(uncached_err.contains("outside the model vocab size 7"));
    }

    #[test]
    fn minigpt_generate_cached_returns_requested_number_of_tokens() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device);
        let prompt = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2]], &device);

        let generated = model.generate_cached(prompt, 3);

        assert_eq!(3, generated.len());
        assert!(generated.iter().all(|&token| token < 7));
    }

    #[test]
    fn minigpt_generate_cached_zero_tokens_returns_empty_output() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 4, 2, &device);
        let prompt = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2]], &device);

        let generated = model.generate_cached(prompt, 0);

        assert!(generated.is_empty());
    }

    #[test]
    fn minigpt_generate_cached_matches_uncached_greedy_generation() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device);
        let prompt = [0, 1, 2];
        let prompt_tensor = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2]], &device);
        let options = GenerationOptions::greedy();

        let generated = model.generate_cached_with_options(prompt_tensor, 3, options);
        let uncached = model
            .generate_with_options(&prompt, 3, &device, options)
            .unwrap();

        assert_eq!(uncached[prompt.len()..], generated);
    }

    #[test]
    fn sampling_options_reject_invalid_values() {
        assert!(GenerationOptions::sampling(0.0, None).is_err());
        assert!(GenerationOptions::sampling(1.0, Some(0)).is_err());
        assert_eq!(
            GenerationOptions {
                temperature: 1.0,
                top_k: Some(3),
            },
            GenerationOptions::sampling(1.0, Some(3)).unwrap()
        );
    }

    #[test]
    fn top_k_sampling_restricts_candidates() {
        let logits = [10.0, 9.0, 1.0, 0.0];

        for random_unit in [0.0, 0.25, 0.75, 0.99] {
            let token = generation::sample_from_logits(&logits, 1.0, Some(2), random_unit);

            assert!(token < 2, "top-k sampling selected token {token}");
        }
    }

    #[test]
    fn zero_temperature_sampling_is_greedy() {
        let logits = [1.0, 5.0, 3.0];

        assert_eq!(
            1,
            generation::sample_from_logits(&logits, 0.0, Some(1), 0.99)
        );
    }

    #[test]
    fn minigpt_generate_with_cache_preserves_prompt_and_appends_tokens() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device);

        let generated = model.generate_with_cache(&[0, 1, 2], 3, &device).unwrap();

        assert_eq!(6, generated.len());
        assert_eq!(&[0, 1, 2], &generated[..3]);
        assert!(generated.iter().all(|&token| token < 7));
    }

    #[test]
    fn minigpt_generate_with_cache_handles_full_context_windows() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 4, 2, &device);

        let generated = model
            .generate_with_cache(&[0, 1, 2, 3], 2, &device)
            .unwrap();

        assert_eq!(6, generated.len());
        assert_eq!(&[0, 1, 2, 3], &generated[..4]);
        assert!(generated.iter().all(|&token| token < 7));
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
