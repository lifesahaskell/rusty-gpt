use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Bool, Int, Tensor, TensorData};

use crate::loader::data::DataLoader;

pub mod persistence;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingLogFormat {
    Plain,
    Json,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainingLogContext {
    pub backend: &'static str,
    pub model: &'static str,
    pub format: TrainingLogFormat,
}

impl TrainingLogContext {
    pub fn plain(model: &'static str) -> Self {
        Self {
            backend: "cpu",
            model,
            format: TrainingLogFormat::Plain,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingParams {
    pub learning_rate: f64,
    pub steps: usize,
    pub eval_interval: usize,
    pub grad_clipping: Option<GradientClippingConfig>,
    pub log_context: TrainingLogContext,
}

impl TrainingParams {
    pub fn new(
        learning_rate: f64,
        steps: usize,
        eval_interval: usize,
        log_context: TrainingLogContext,
    ) -> Self {
        Self {
            learning_rate,
            steps,
            eval_interval,
            grad_clipping: None,
            log_context,
        }
    }

    pub fn with_grad_clip_norm(mut self, norm: f32) -> Self {
        self.grad_clipping = Some(GradientClippingConfig::Norm(norm));
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MiniGptConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_blocks: usize,
    pub max_position_embeddings: usize,
    pub num_heads: usize,
}

fn should_log_training_step(step: usize, steps: usize, eval_interval: usize) -> bool {
    if eval_interval == 0 {
        return step + 1 == steps;
    }

    steps <= eval_interval || step.is_multiple_of(eval_interval) || step + 1 == steps
}

fn language_model_loss<B: Backend>(
    loss_fn: &CrossEntropyLoss<B>,
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let [batch_size, seq_len, vocab_size] = logits.shape().dims();
    loss_fn.forward(
        logits.reshape([batch_size * seq_len, vocab_size]),
        targets.reshape([batch_size * seq_len]),
    )
}

fn value_loss<B: Backend>(
    loader: &DataLoader,
    device: &B::Device,
    loss_fn: &CrossEntropyLoss<B>,
    forward: impl FnOnce(Tensor<B, 2, Int>) -> Tensor<B, 3>,
) -> Result<B::FloatElem, String> {
    let (inputs, targets) = loader.next_batch::<B>(device)?;
    Ok(language_model_loss(loss_fn, forward(inputs), targets).into_scalar())
}

fn training_progress_log_line<B: Backend>(
    context: TrainingLogContext,
    step: usize,
    steps: usize,
    training_loss: B::FloatElem,
    value_loss: B::FloatElem,
) -> String {
    match context.format {
        TrainingLogFormat::Plain => {
            format!("Step {step}: training loss = {training_loss:.4}, value loss = {value_loss:.4}")
        }
        TrainingLogFormat::Json => {
            format!(
                r#"{{"event":"training_progress","backend":"{}","model":"{}","step":{},"total_steps":{},"training_loss":{:.6},"value_loss":{:.6}}}"#,
                context.backend, context.model, step, steps, training_loss, value_loss
            )
        }
    }
}

fn log_training_progress<B: Backend>(
    context: TrainingLogContext,
    step: usize,
    steps: usize,
    training_loss: B::FloatElem,
    value_loss: B::FloatElem,
) {
    println!(
        "{}",
        training_progress_log_line::<B>(context, step, steps, training_loss, value_loss)
    );
}

fn train_language_model<B, M>(
    mut model: M,
    loader: &DataLoader,
    value_loader: &DataLoader,
    device: &B::Device,
    params: TrainingParams,
    forward: impl Fn(&M, Tensor<B, 2, Int>) -> Tensor<B, 3>,
) -> Result<M, String>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(params.grad_clipping)
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);

    for step in 0..params.steps {
        let (inputs, targets) = loader.next_batch::<B>(device)?;
        let loss = language_model_loss(&loss_fn, forward(&model, inputs), targets);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(params.learning_rate, model, grads);

        if should_log_training_step(step, params.steps, params.eval_interval) {
            let training_loss = loss.into_scalar();
            let value_loss = value_loss(value_loader, device, &loss_fn, |inputs| {
                forward(&model, inputs)
            })?;
            log_training_progress::<B>(
                params.log_context,
                step,
                params.steps,
                training_loss,
                value_loss,
            );
        }
    }

    Ok(model)
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

impl<B: AutodiffBackend> TrivialModel<B> {
    pub fn train(
        loader: &DataLoader,
        value_loader: &DataLoader,
        device: &B::Device,
        vocab_size: usize,
        d_model: usize,
        params: TrainingParams,
    ) -> Result<Self, String> {
        train_language_model(
            TrivialModel::<B>::new(vocab_size, d_model, device),
            loader,
            value_loader,
            device,
            params,
            |model, inputs| model.forward(inputs),
        )
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

impl<B: AutodiffBackend> SingleAttentionModel<B> {
    pub fn train(
        loader: &DataLoader,
        value_loader: &DataLoader,
        device: &B::Device,
        vocab_size: usize,
        d_model: usize,
        head_dim: usize,
        params: TrainingParams,
    ) -> Result<Self, String> {
        train_language_model(
            SingleAttentionModel::<B>::new(vocab_size, d_model, head_dim, device),
            loader,
            value_loader,
            device,
            params,
            |model, inputs| model.forward_tokens(inputs),
        )
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

impl<B: AutodiffBackend> MultiAttentionModel<B> {
    pub fn train(
        loader: &DataLoader,
        value_loader: &DataLoader,
        device: &B::Device,
        vocab_size: usize,
        d_model: usize,
        num_heads: usize,
        params: TrainingParams,
    ) -> Result<Self, String> {
        train_language_model(
            MultiAttentionModel::<B>::new(vocab_size, d_model, num_heads, device),
            loader,
            value_loader,
            device,
            params,
            |model, inputs| model.forward_tokens(inputs),
        )
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

    pub fn forward_with_weights(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let (output, attn_weights) = self.forward_inner(x, true);
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
        let embed_dim = self.num_heads * self.head_dim;
        let (q, k, v) = self.project_qkv(x);
        let [batch_size, _, seq_len, _] = q.shape().dims();

        let scores = q.matmul(k.swap_dims(2, 3)) / (self.head_dim as f32).sqrt();
        let mask = Self::causal_mask(batch_size, self.num_heads, seq_len, &scores.device());
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

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    ln1: LayerNorm<B>,
    attn: MultiHeadAttention<B>,
    ln2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(d_model: usize, num_heads: usize, device: &B::Device) -> Self {
        Self {
            ln1: LayerNormConfig::new(d_model).init(device),
            attn: MultiHeadAttention::new(d_model, num_heads, device),
            ln2: LayerNormConfig::new(d_model).init(device),
            mlp: Mlp::new(d_model, 4 * d_model, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.ln1.forward(x));
        x.clone() + self.mlp.forward(self.ln2.forward(x))
    }

    pub fn forward_with_weights(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let (attn_output, attn_weights) =
            self.attn.forward_with_weights(self.ln1.forward(x.clone()));
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
        for block in &self.blocks {
            x = block.forward(x);
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
        for block in &self.blocks {
            let (next_x, attention) = block.forward_with_weights(x);
            x = next_x;
            attentions.push(attention);
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
        if prompt.is_empty() {
            return Err("prompt must contain at least one token".to_string());
        }
        if let Some(token) = prompt.iter().find(|&&token| token >= self.vocab_size) {
            return Err(format!(
                "prompt token {token} is outside the model vocab size {}",
                self.vocab_size
            ));
        }

        let mut output = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let context_start = output.len().saturating_sub(self.max_position_embeddings);
            let context = &output[context_start..];
            let input: Vec<i64> = context.iter().map(|&token| token as i64).collect();
            let tokens = Tensor::from_data(TensorData::new(input, [1, context.len()]), device);
            let logits = self.forward_tokens(tokens);
            let [_batch_size, seq_len, vocab_size] = logits.shape().dims();
            let logits = logits.into_data().to_vec::<f32>().unwrap();
            let last_position_start = (seq_len - 1) * vocab_size;
            let next_token = logits[last_position_start..last_position_start + vocab_size]
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(token, _)| token)
                .expect("vocab size should be greater than zero");

            output.push(next_token);
        }

        Ok(output)
    }

    pub fn generate_cached(&self, prompt: Tensor<B, 2, Int>, max_new: usize) -> Vec<usize> {
        if max_new == 0 {
            return Vec::new();
        }

        let device = prompt.device();
        let mut cache = KvCache::empty(self.num_layers());
        let (logits, new_cache) = self.forward_with_cache(prompt, cache);
        cache = new_cache;
        let mut next_token = Self::greedy_last_token(logits);
        let mut generated = vec![next_token];

        for _ in 1..max_new {
            let next_input = Tensor::from_data([[next_token as i64]], &device);
            let (logits, new_cache) = self.forward_with_cache(next_input, cache);
            cache = new_cache;
            next_token = Self::greedy_last_token(logits);
            generated.push(next_token);
        }

        generated
    }

    fn greedy_last_token(logits: Tensor<B, 3>) -> usize {
        let [_batch_size, seq_len, vocab_size] = logits.shape().dims();
        let logits = logits.into_data().to_vec::<f32>().unwrap();
        let last_position_start = (seq_len - 1) * vocab_size;
        logits[last_position_start..last_position_start + vocab_size]
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(token, _)| token)
            .expect("vocab size should be greater than zero")
    }
}

impl<B: AutodiffBackend> MiniGpt<B> {
    pub fn train(
        loader: &DataLoader,
        value_loader: &DataLoader,
        device: &B::Device,
        config: MiniGptConfig,
        params: TrainingParams,
    ) -> Result<Self, String> {
        train_language_model(
            MiniGpt::<B>::new(
                config.vocab_size,
                config.d_model,
                config.num_blocks,
                config.max_position_embeddings,
                config.num_heads,
                device,
            ),
            loader,
            value_loader,
            device,
            params,
            |model, inputs| model.forward_tokens(inputs),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::{Int, Tensor};

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
                format: TrainingLogFormat::Json,
            },
            10,
            100,
            1.25,
            2.5,
        );
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();

        assert_eq!("training_progress", parsed["event"]);
        assert_eq!("cuda", parsed["backend"]);
        assert_eq!("minigpt", parsed["model"]);
        assert_eq!(10, parsed["step"]);
        assert_eq!(100, parsed["total_steps"]);
        assert_eq!(1.25, parsed["training_loss"]);
        assert_eq!(2.5, parsed["value_loss"]);
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

        let generated = model.generate_cached(prompt_tensor, 3);
        let uncached = model.generate(&prompt, 3, &device).unwrap();

        assert_eq!(uncached[prompt.len()..], generated);
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
