use std::sync::Arc;
use std::time::Instant;

use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router, extract::State};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use crate::model::MiniGpt;
use crate::observability::{EventLogger, RuntimeEvent};
use crate::tokenizer::RuntimeTokenizer;

pub type SharedServerState<B> = Arc<ServerState<B>>;

pub struct ServerState<B: Backend> {
    model: MiniGpt<B>,
    tokenizer: RuntimeTokenizer,
    device: B::Device,
    logger: EventLogger,
}

impl<B: Backend> ServerState<B> {
    pub fn new(
        model: MiniGpt<B>,
        tokenizer: RuntimeTokenizer,
        device: B::Device,
        logger: EventLogger,
    ) -> Self {
        Self {
            model,
            tokenizer,
            device,
            logger,
        }
    }

    pub fn model_vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    pub fn model_block_size(&self) -> usize {
        self.model.block_size()
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
    let started_at = Instant::now();
    state.logger.log(RuntimeEvent::GenerateRequestAccepted {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        prompt_chars: request.prompt.chars().count(),
    });

    if request.prompt.is_empty() {
        return Err(logged_bad_request(
            &state.logger,
            "prompt must not be empty",
            started_at,
        ));
    }
    if request.max_tokens == 0 {
        return Err(logged_bad_request(
            &state.logger,
            "max_tokens must be greater than zero",
            started_at,
        ));
    }
    if request.temperature <= 0.0 {
        return Err(logged_bad_request(
            &state.logger,
            "temperature must be greater than zero",
            started_at,
        ));
    }

    let prompt_tokens = state
        .tokenizer
        .try_encode(&request.prompt)
        .map_err(|err| logged_bad_request(&state.logger, err, started_at))?;
    let generated_tokens = state
        .model
        .generate_with_cache(&prompt_tokens, request.max_tokens, &state.device)
        .map_err(|err| logged_bad_request(&state.logger, err, started_at))?;
    let attention_tokens = context_window(&generated_tokens, state.model.block_size());
    let attention = attention_for_tokens(&state, attention_tokens)?;
    let generated = state.tokenizer.decode(&generated_tokens);
    let tokens = generated_tokens
        .iter()
        .map(|&token| state.tokenizer.decode(&[token]))
        .collect();
    state.logger.log(RuntimeEvent::GenerateRequestCompleted {
        status: StatusCode::OK.as_u16(),
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated_tokens.len().saturating_sub(prompt_tokens.len()),
        elapsed_ms: started_at.elapsed().as_millis(),
    });

    Ok(Json(GenerateResponse {
        generated,
        tokens,
        attention,
    }))
}

fn bad_request(message: impl Into<String>) -> (StatusCode, String) {
    (StatusCode::BAD_REQUEST, message.into())
}

fn logged_bad_request(
    logger: &EventLogger,
    message: impl Into<String>,
    started_at: Instant,
) -> (StatusCode, String) {
    let error = bad_request(message);
    logger.log(RuntimeEvent::GenerateRequestRejected {
        status: error.0.as_u16(),
        reason: error.1.clone(),
        elapsed_ms: started_at.elapsed().as_millis(),
    });
    error
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
    use crate::observability::LogFormat;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use std::sync::Mutex;

    #[test]
    fn context_window_crops_to_block_size() {
        assert_eq!(&[2, 3, 4], context_window(&[0, 1, 2, 3, 4], 3));
    }

    #[test]
    fn server_state_exposes_model_info() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 2, 6, 2, &device);
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        let state = ServerState::new(
            model,
            tokenizer,
            device,
            EventLogger::stdout(LogFormat::Plain),
        );

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
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        let state = ServerState::new(
            model,
            tokenizer,
            device,
            EventLogger::stdout(LogFormat::Plain),
        );

        let attention = attention_for_tokens(&state, &[0, 1, 2]).unwrap();

        assert_eq!(4, attention.len());
        assert_eq!(0, attention[0].layer);
        assert_eq!(0, attention[0].head);
        assert_eq!(3, attention[0].weights.len());
        assert_eq!(3, attention[0].weights[0].len());
        assert_eq!(1, attention[3].layer);
        assert_eq!(1, attention[3].head);
    }

    #[tokio::test]
    async fn generate_logs_request_lifecycle() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device);
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        let lines = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&lines);
        let logger = EventLogger::with_sink(LogFormat::Json, move |line| {
            captured.lock().unwrap().push(line);
        });
        let state = Arc::new(ServerState::new(model, tokenizer, device, logger));

        let response = generate(
            State(state),
            Json(GenerateRequest {
                prompt: "ab".to_string(),
                max_tokens: 1,
                temperature: 1.0,
            }),
        )
        .await;

        assert!(response.is_ok());
        let lines = lines.lock().unwrap();
        let accepted: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        let completed: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!("generate_request_accepted", accepted["event"]);
        assert_eq!(2, accepted["prompt_chars"]);
        assert_eq!("generate_request_completed", completed["event"]);
        assert_eq!(200, completed["status"]);
        assert_eq!(2, completed["prompt_tokens"]);
        assert_eq!(1, completed["generated_tokens"]);
    }

    #[tokio::test]
    async fn generate_logs_rejected_request_without_prompt_text() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device);
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        let lines = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&lines);
        let logger = EventLogger::with_sink(LogFormat::Json, move |line| {
            captured.lock().unwrap().push(line);
        });
        let state = Arc::new(ServerState::new(model, tokenizer, device, logger));

        let response = generate(
            State(state),
            Json(GenerateRequest {
                prompt: "".to_string(),
                max_tokens: 1,
                temperature: 1.0,
            }),
        )
        .await;

        assert!(response.is_err());
        let lines = lines.lock().unwrap();
        let rejected: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!("generate_request_rejected", rejected["event"]);
        assert_eq!(400, rejected["status"]);
        assert_eq!("prompt must not be empty", rejected["reason"]);
        assert!(rejected.get("prompt").is_none());
    }
}
