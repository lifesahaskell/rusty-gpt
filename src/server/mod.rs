use std::sync::Arc;

use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router, extract::State};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use crate::model::MiniGpt;
use crate::tokenizer::char::CharTokenizer;

pub type SharedServerState<B> = Arc<ServerState<B>>;

pub struct ServerState<B: Backend> {
    model: MiniGpt<B>,
    tokenizer: CharTokenizer,
    device: B::Device,
}

impl<B: Backend> ServerState<B> {
    pub fn new(model: MiniGpt<B>, tokenizer: CharTokenizer, device: B::Device) -> Self {
        Self {
            model,
            tokenizer,
            device,
        }
    }
}

pub fn router<B>() -> Router<SharedServerState<B>>
where
    B: Backend + Send + Sync + 'static,
    B::Device: Send + Sync + 'static,
{
    Router::new()
        .route("/generate", post(generate::<B>))
        .route("/info", get(info::<B>))
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub generated: String,
    pub tokens: Vec<String>,
    pub attention: Vec<AttentionData>,
}

#[derive(Debug, Serialize)]
pub struct AttentionData {
    pub layer: usize,
    pub head: usize,
    pub weights: Vec<Vec<f64>>,
}

#[derive(Debug, Serialize)]
pub struct InfoResponse {
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub block_size: usize,
}

async fn generate<B>(
    State(state): State<SharedServerState<B>>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)>
where
    B: Backend,
{
    if request.prompt.is_empty() {
        return Err(bad_request("prompt must not be empty"));
    }
    if request.max_tokens == 0 {
        return Err(bad_request("max_tokens must be greater than zero"));
    }
    if request.temperature <= 0.0 {
        return Err(bad_request("temperature must be greater than zero"));
    }

    let prompt_tokens = state
        .tokenizer
        .try_encode(&request.prompt)
        .map_err(|err| (StatusCode::BAD_REQUEST, err))?;
    let generated_tokens = state
        .model
        .generate(&prompt_tokens, request.max_tokens, &state.device)
        .map_err(bad_request)?;
    let attention_tokens = context_window(&generated_tokens, state.model.block_size());
    let attention = attention_for_tokens(&state, attention_tokens)?;
    let generated = state.tokenizer.decode(&generated_tokens);
    let tokens = generated_tokens
        .iter()
        .map(|&token| state.tokenizer.decode(&[token]))
        .collect();

    Ok(Json(GenerateResponse {
        generated,
        tokens,
        attention,
    }))
}

fn bad_request(message: impl Into<String>) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, message.into())
}

async fn info<B>(State(state): State<SharedServerState<B>>) -> Json<InfoResponse>
where
    B: Backend,
{
    Json(InfoResponse {
        vocab_size: state.model.vocab_size(),
        num_layers: state.model.num_layers(),
        num_heads: state.model.num_heads(),
        block_size: state.model.block_size(),
    })
}

fn context_window(tokens: &[usize], block_size: usize) -> &[usize] {
    let start = tokens.len().saturating_sub(block_size);
    &tokens[start..]
}

fn attention_for_tokens<B: Backend>(
    state: &ServerState<B>,
    tokens: &[usize],
) -> Result<Vec<AttentionData>, (StatusCode, String)> {
    let input: Vec<i64> = tokens.iter().map(|&token| token as i64).collect();
    let token_tensor: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(input, [1, tokens.len()]), &state.device);
    let (_logits, attentions) = state.model.forward_tokens_with_attention(token_tensor);

    let mut attention_data = Vec::new();
    for (layer, attention) in attentions.into_iter().enumerate() {
        let [batch_size, num_heads, seq_len, _] = attention.shape().dims();
        if batch_size != 1 {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("expected attention batch size 1, got {batch_size}"),
            ));
        }

        let values = attention.into_data().to_vec::<f32>().map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to serialize attention tensor: {err}"),
            )
        })?;

        for head in 0..num_heads {
            let mut weights = Vec::with_capacity(seq_len);
            for query_pos in 0..seq_len {
                let mut row = Vec::with_capacity(seq_len);
                for key_pos in 0..seq_len {
                    let index = ((head * seq_len + query_pos) * seq_len) + key_pos;
                    row.push(values[index] as f64);
                }
                weights.push(row);
            }

            attention_data.push(AttentionData {
                layer,
                head,
                weights,
            });
        }
    }

    Ok(attention_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    #[test]
    fn context_window_crops_to_block_size() {
        assert_eq!(&[2, 3, 4], context_window(&[0, 1, 2, 3, 4], 3));
    }

    #[test]
    fn server_state_exposes_model_info() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 2, 6, 2, &device);
        let tokenizer = CharTokenizer::from_text("abcdefg");
        let state = ServerState::new(model, tokenizer, device);

        assert_eq!(7, state.model.vocab_size());
        assert_eq!(2, state.model.num_layers());
        assert_eq!(2, state.model.num_heads());
        assert_eq!(6, state.model.block_size());
    }

    #[test]
    fn attention_for_tokens_returns_one_matrix_per_layer_head() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 2, 6, 2, &device);
        let tokenizer = CharTokenizer::from_text("abcdefg");
        let state = ServerState::new(model, tokenizer, device);

        let attention = attention_for_tokens(&state, &[0, 1, 2]).unwrap();

        assert_eq!(4, attention.len());
        assert_eq!(0, attention[0].layer);
        assert_eq!(0, attention[0].head);
        assert_eq!(3, attention[0].weights.len());
        assert_eq!(3, attention[0].weights[0].len());
        assert_eq!(1, attention[3].layer);
        assert_eq!(1, attention[3].head);
    }
}
