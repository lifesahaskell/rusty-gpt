use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use super::attention::{KvCache, MultiHeadAttention, SingleHeadAttention};
use super::block::{Block, FeedForward};
use super::generation::{self, GenerationOptions};
use super::moe::MoeForwardAux;

#[derive(Debug, Clone, Copy)]
pub struct MiniGptConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_blocks: usize,
    pub max_position_embeddings: usize,
    pub num_heads: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MoeGptConfig {
    pub base: MiniGptConfig,
    pub num_experts: usize,
    pub top_k: usize,
    pub aux_loss_weight: f64,
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

#[derive(Module, Debug)]
pub struct MoeGpt<B: Backend> {
    token_embed: Embedding<B>,
    position_embed: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_final: LayerNorm<B>,
    lm_head: Linear<B>,
    vocab_size: usize,
    max_position_embeddings: usize,
    num_experts: usize,
    top_k: usize,
    aux_loss_weight: f64,
}

impl<B: Backend> MoeGpt<B> {
    pub fn new(config: MoeGptConfig, device: &B::Device) -> Self {
        let token_embed =
            EmbeddingConfig::new(config.base.vocab_size, config.base.d_model).init(device);
        let position_embed =
            EmbeddingConfig::new(config.base.max_position_embeddings, config.base.d_model)
                .init(device);
        let blocks = (0..config.base.num_blocks)
            .map(|_| {
                Block::new_with_feed_forward(
                    config.base.d_model,
                    config.base.num_heads,
                    FeedForward::moe(
                        config.base.d_model,
                        config.num_experts,
                        config.top_k,
                        device,
                    ),
                    device,
                )
            })
            .collect();
        let ln_final = LayerNormConfig::new(config.base.d_model).init(device);
        let lm_head = LinearConfig::new(config.base.d_model, config.base.vocab_size).init(device);

        Self {
            token_embed,
            position_embed,
            blocks,
            ln_final,
            lm_head,
            vocab_size: config.base.vocab_size,
            max_position_embeddings: config.base.max_position_embeddings,
            num_experts: config.num_experts,
            top_k: config.top_k,
            aux_loss_weight: config.aux_loss_weight,
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

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn moe_top_k(&self) -> usize {
        self.top_k
    }

    pub fn aux_loss_weight(&self) -> f64 {
        self.aux_loss_weight
    }

    pub fn forward_tokens(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_tokens_with_aux(tokens).0
    }

    pub fn forward_tokens_with_aux(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Vec<MoeForwardAux<B>>) {
        let [_b, t] = tokens.dims();
        let positions = Tensor::arange(0..t as i64, &tokens.device()).unsqueeze();
        let tok = self.token_embed.forward(tokens);
        let pos = self.position_embed.forward(positions);
        self.forward_with_aux(tok + pos)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_aux(x).0
    }

    pub fn forward_with_aux(&self, mut x: Tensor<B, 3>) -> (Tensor<B, 3>, Vec<MoeForwardAux<B>>) {
        let mut aux_losses = Vec::with_capacity(self.blocks.len());
        if let Some(first_block) = self.blocks.first() {
            let [batch_size, seq_len, _] = x.shape().dims();
            let mask = MultiHeadAttention::<B>::causal_mask(
                batch_size,
                first_block.num_heads(),
                seq_len,
                &x.device(),
            );
            for block in &self.blocks {
                let (next_x, aux) = block.forward_with_mask_and_aux(x, mask.clone());
                x = next_x;
                if let Some(aux) = aux {
                    aux_losses.push(aux);
                }
            }
        }

        let x = self.ln_final.forward(x);
        (self.lm_head.forward(x), aux_losses)
    }

    pub fn forward_tokens_with_attention(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 4>>) {
        self.forward_tokens_with_attention_and_routing(tokens)
            .map_attention()
    }

    pub fn forward_tokens_with_attention_and_routing(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> MoeAttentionRouting<B> {
        let [_b, t] = tokens.dims();
        let positions = Tensor::arange(0..t as i64, &tokens.device()).unsqueeze();
        let tok = self.token_embed.forward(tokens);
        let pos = self.position_embed.forward(positions);
        self.forward_with_attention_and_routing(tok + pos)
    }

    pub fn forward_with_attention_and_routing(
        &self,
        mut x: Tensor<B, 3>,
    ) -> MoeAttentionRouting<B> {
        let mut attentions = Vec::with_capacity(self.blocks.len());
        let mut routing = Vec::with_capacity(self.blocks.len());
        if let Some(first_block) = self.blocks.first() {
            let [batch_size, seq_len, _] = x.shape().dims();
            let mask = MultiHeadAttention::<B>::causal_mask(
                batch_size,
                first_block.num_heads(),
                seq_len,
                &x.device(),
            );
            for block in &self.blocks {
                let (next_x, attention, aux) =
                    block.forward_with_weights_mask_and_aux(x, mask.clone());
                x = next_x;
                attentions.push(attention);
                if let Some(aux) = aux {
                    routing.push(aux);
                }
            }
        }

        let x = self.ln_final.forward(x);
        MoeAttentionRouting {
            logits: self.lm_head.forward(x),
            attentions,
            routing,
        }
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
        (self.lm_head.forward(x), cache)
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
            output.push(Self::select_last_token(logits, options));
        }

        Ok(output)
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

pub struct MoeAttentionRouting<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub attentions: Vec<Tensor<B, 4>>,
    pub routing: Vec<MoeForwardAux<B>>,
}

impl<B: Backend> MoeAttentionRouting<B> {
    fn map_attention(self) -> (Tensor<B, 3>, Vec<Tensor<B, 4>>) {
        (self.logits, self.attentions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::{Int, Tensor};

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
}
