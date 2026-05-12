use burn::module::Module;
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Bool, Int, Tensor, TensorData};

use crate::loader::data::DataLoader;

pub mod persistence;

fn should_log_training_step(step: usize, steps: usize, eval_interval: usize) -> bool {
    if eval_interval == 0 {
        return step + 1 == steps;
    }

    steps <= eval_interval || step % eval_interval == 0 || step + 1 == steps
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
        lr: f64,
        steps: usize,
        eval_interval: usize,
    ) -> Result<Self, String> {
        let mut model = TrivialModel::<B>::new(vocab_size, d_model, device);
        let mut optimizer = AdamWConfig::new().init();
        let loss_fn = CrossEntropyLossConfig::new().init(device);

        for step in 0..steps {
            let (inputs, targets) = loader.next_batch::<B>(device)?;
            let logits = model.forward(inputs);
            let loss = language_model_loss(&loss_fn, logits, targets);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);

            if should_log_training_step(step, steps, eval_interval) {
                let training_loss = loss.into_scalar();
                let value_loss = value_loss(value_loader, device, &loss_fn, |inputs| {
                    model.forward(inputs)
                })?;
                println!(
                    "Step {step}: training loss = {training_loss:.4}, value loss = {value_loss:.4}"
                );
            }
        }

        Ok(model)
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
        lr: f64,
        steps: usize,
        eval_interval: usize,
    ) -> Result<Self, String> {
        let mut model = SingleAttentionModel::<B>::new(vocab_size, d_model, head_dim, device);
        let mut optimizer = AdamWConfig::new().init();
        let loss_fn = CrossEntropyLossConfig::new().init(device);

        for step in 0..steps {
            let (inputs, targets) = loader.next_batch::<B>(device)?;
            let logits = model.forward_tokens(inputs);
            let loss = language_model_loss(&loss_fn, logits, targets);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);

            if should_log_training_step(step, steps, eval_interval) {
                let training_loss = loss.into_scalar();
                let value_loss = value_loss(value_loader, device, &loss_fn, |inputs| {
                    model.forward_tokens(inputs)
                })?;
                println!(
                    "Step {step}: training loss = {training_loss:.4}, value loss = {value_loss:.4}"
                );
            }
        }

        Ok(model)
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
        lr: f64,
        steps: usize,
        eval_interval: usize,
    ) -> Result<Self, String> {
        let mut model = MultiAttentionModel::<B>::new(vocab_size, d_model, num_heads, device);
        let mut optimizer = AdamWConfig::new().init();
        let loss_fn = CrossEntropyLossConfig::new().init(device);

        for step in 0..steps {
            let (inputs, targets) = loader.next_batch::<B>(device)?;
            let logits = model.forward_tokens(inputs);
            let loss = language_model_loss(&loss_fn, logits, targets);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);

            if should_log_training_step(step, steps, eval_interval) {
                let training_loss = loss.into_scalar();
                let value_loss = value_loss(value_loader, device, &loss_fn, |inputs| {
                    model.forward_tokens(inputs)
                })?;
                println!(
                    "Step {step}: training loss = {training_loss:.4}, value loss = {value_loss:.4}"
                );
            }
        }

        Ok(model)
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

    fn forward_inner(
        &self,
        x: Tensor<B, 3>,
        return_weights: bool,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 4>>) {
        let embed_dim = self.num_heads * self.head_dim;
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
        let x = x.clone() + self.mlp.forward(self.ln2.forward(x));
        x
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
}

impl<B: AutodiffBackend> MiniGpt<B> {
    pub fn train(
        loader: &DataLoader,
        value_loader: &DataLoader,
        device: &B::Device,
        vocab_size: usize,
        d_model: usize,
        num_blocks: usize,
        max_position_embeddings: usize,
        num_heads: usize,
        lr: f64,
        steps: usize,
        eval_interval: usize,
        grad_clip_norm: f32,
    ) -> Result<Self, String> {
        let mut model = MiniGpt::<B>::new(
            vocab_size,
            d_model,
            num_blocks,
            max_position_embeddings,
            num_heads,
            device,
        );
        let mut optimizer = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(grad_clip_norm)))
            .init();
        let loss_fn = CrossEntropyLossConfig::new().init(device);

        for step in 0..steps {
            let (inputs, targets) = loader.next_batch::<B>(device)?;
            let logits = model.forward_tokens(inputs);
            let loss = language_model_loss(&loss_fn, logits, targets);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);

            if should_log_training_step(step, steps, eval_interval) {
                let training_loss = loss.into_scalar();
                let value_loss = value_loss(value_loader, device, &loss_fn, |inputs| {
                    model.forward_tokens(inputs)
                })?;
                println!(
                    "Step {step}: training loss = {training_loss:.4}, value loss = {value_loss:.4}"
                );
            }
        }

        Ok(model)
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
    #[should_panic(expected = "d_model must be divisible by num_heads")]
    fn multi_head_attention_requires_even_head_split() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;

        let _attention = MultiHeadAttention::<TestBackend>::new(7, 2, &device);
    }
}
