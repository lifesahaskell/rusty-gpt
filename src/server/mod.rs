use std::sync::Arc;
use std::time::Instant;

use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router, extract::State};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use crate::model::{GenerationOptions, MiniGpt};
use crate::observability::{EventLogger, RuntimeEvent};
use crate::tokenizer::RuntimeTokenizer;

pub type SharedServerState<B> = Arc<ServerState<B>>;

pub struct ServerState<B: Backend> {
    model: MiniGpt<B>,
    tokenizer: RuntimeTokenizer,
    device: B::Device,
    logger: EventLogger,
    provenance: ServerProvenance,
}

pub struct ServerProvenance {
    pub started_at: Instant,
    pub checkpoint_source: CheckpointSource,
    pub checkpoint_basename: Option<String>,
    pub checkpoint_sha256: Option<String>,
    pub tokenizer_sha256: Option<String>,
}

impl ServerProvenance {
    pub fn fresh() -> Self {
        Self {
            started_at: Instant::now(),
            checkpoint_source: CheckpointSource::None,
            checkpoint_basename: None,
            checkpoint_sha256: None,
            tokenizer_sha256: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointSource {
    None,
    Explicit,
    Latest,
}

impl CheckpointSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Explicit => "explicit",
            Self::Latest => "latest",
        }
    }
}

impl<B: Backend> ServerState<B> {
    pub fn new(
        model: MiniGpt<B>,
        tokenizer: RuntimeTokenizer,
        device: B::Device,
        logger: EventLogger,
        provenance: ServerProvenance,
    ) -> Self {
        Self {
            model,
            tokenizer,
            device,
            logger,
            provenance,
        }
    }

    pub fn model_vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    pub fn model_block_size(&self) -> usize {
        self.model.block_size()
    }

    pub fn tokenizer_vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    pub fn model_tokenizer_vocab_match(&self) -> bool {
        self.model.vocab_size() == self.tokenizer.vocab_size()
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
        // NOTE: S2-T1 (rate limit) must exempt this route — monitoring probes hit it.
        .route("/health", get(health::<B>))
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
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
    pub tokenizer_vocab_size: usize,
    pub model_tokenizer_vocab_match: bool,
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
    if request.top_k == Some(0) {
        return Err(logged_bad_request(
            &state.logger,
            "top_k must be greater than zero",
            started_at,
        ));
    }

    let prompt_tokens = state
        .tokenizer
        .try_encode(&request.prompt)
        .map_err(|err| logged_bad_request(&state.logger, err, started_at))?;
    let generated_tokens = state
        .model
        .generate_with_cache_options(
            &prompt_tokens,
            request.max_tokens,
            &state.device,
            GenerationOptions::sampling(request.temperature, request.top_k)
                .map_err(|err| logged_bad_request(&state.logger, err, started_at))?,
        )
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
        tokenizer_vocab_size: state.tokenizer.vocab_size(),
        model_tokenizer_vocab_match: state.model_tokenizer_vocab_match(),
    })
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub uptime_seconds: u64,
    pub model: HealthModel,
    pub checkpoint: HealthCheckpoint,
    pub tokenizer: HealthTokenizer,
}

#[derive(Debug, Serialize)]
pub struct HealthTokenizer {
    pub kind: &'static str,
    pub sha256: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct HealthModel {
    pub kind: &'static str,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub block_size: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Serialize)]
pub struct HealthCheckpoint {
    pub loaded: bool,
    pub source: &'static str,
    pub basename: Option<String>,
    pub sha256: Option<String>,
}

async fn health<B>(State(state): State<SharedServerState<B>>) -> Json<HealthResponse>
where
    B: Backend,
{
    let provenance = &state.provenance;
    Json(HealthResponse {
        status: "ok",
        uptime_seconds: provenance.started_at.elapsed().as_secs(),
        model: HealthModel {
            kind: "minigpt",
            embed_dim: state.model.d_model(),
            num_heads: state.model.num_heads(),
            num_layers: state.model.num_layers(),
            block_size: state.model.block_size(),
            vocab_size: state.model.vocab_size(),
        },
        checkpoint: HealthCheckpoint {
            loaded: provenance.checkpoint_source != CheckpointSource::None,
            source: provenance.checkpoint_source.as_str(),
            basename: provenance.checkpoint_basename.clone(),
            sha256: provenance.checkpoint_sha256.clone(),
        },
        tokenizer: HealthTokenizer {
            kind: state.tokenizer.kind(),
            sha256: provenance.tokenizer_sha256.clone(),
        },
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
    fn checkpoint_source_serializes_to_spec_strings() {
        assert_eq!("none", CheckpointSource::None.as_str());
        assert_eq!("explicit", CheckpointSource::Explicit.as_str());
        assert_eq!("latest", CheckpointSource::Latest.as_str());
    }

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
            ServerProvenance::fresh(),
        );

        assert_eq!(7, state.model.vocab_size());
        assert_eq!(2, state.model.num_layers());
        assert_eq!(2, state.model.num_heads());
        assert_eq!(6, state.model.block_size());
        assert_eq!(7, state.tokenizer_vocab_size());
        assert!(state.model_tokenizer_vocab_match());
    }

    #[tokio::test]
    async fn info_exposes_runtime_tokenizer_compatibility() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(8, 8, 1, 6, 2, &device);
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        let state = Arc::new(ServerState::new(
            model,
            tokenizer,
            device,
            EventLogger::stdout(LogFormat::Plain),
            ServerProvenance::fresh(),
        ));

        let Json(response) = info(State(state)).await;

        assert_eq!(8, response.vocab_size);
        assert_eq!(7, response.tokenizer_vocab_size);
        assert!(!response.model_tokenizer_vocab_match);
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
            ServerProvenance::fresh(),
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
        let state = Arc::new(ServerState::new(
            model,
            tokenizer,
            device,
            logger,
            ServerProvenance::fresh(),
        ));

        let response = generate(
            State(state),
            Json(GenerateRequest {
                prompt: "ab".to_string(),
                max_tokens: 1,
                temperature: 1.0,
                top_k: Some(3),
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
        let state = Arc::new(ServerState::new(
            model,
            tokenizer,
            device,
            logger,
            ServerProvenance::fresh(),
        ));

        let response = generate(
            State(state),
            Json(GenerateRequest {
                prompt: "".to_string(),
                max_tokens: 1,
                temperature: 1.0,
                top_k: None,
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

    #[tokio::test]
    async fn health_reports_tokenizer_kind_and_sha256() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device);
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        let provenance = ServerProvenance {
            started_at: Instant::now(),
            checkpoint_source: CheckpointSource::None,
            checkpoint_basename: None,
            checkpoint_sha256: None,
            tokenizer_sha256: Some("deadbeef".to_string()),
        };
        let state = Arc::new(ServerState::new(
            model,
            tokenizer,
            device,
            EventLogger::stdout(LogFormat::Plain),
            provenance,
        ));

        let Json(response) = health(State(state)).await;

        assert_eq!("char", response.tokenizer.kind);
        assert_eq!(Some("deadbeef".to_string()), response.tokenizer.sha256);
    }

    #[tokio::test]
    async fn health_never_exposes_absolute_path() {
        type TestBackend = NdArray<f32, i64>;
        let parent_dir = std::env::temp_dir().join(format!(
            "rusty-gpt-health-disclosure-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&parent_dir).unwrap();
        let checkpoint_path = parent_dir.join("mini_gpt.step-5000.mpk");
        std::fs::write(&checkpoint_path, b"bytes").unwrap();

        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device);
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        let provenance = ServerProvenance {
            started_at: Instant::now(),
            checkpoint_source: CheckpointSource::Latest,
            checkpoint_basename: checkpoint_path
                .file_name()
                .and_then(|name| name.to_str())
                .map(String::from),
            checkpoint_sha256: Some("placeholder-sha".to_string()),
            tokenizer_sha256: None,
        };
        let state = Arc::new(ServerState::new(
            model,
            tokenizer,
            device,
            EventLogger::stdout(LogFormat::Plain),
            provenance,
        ));

        let Json(response) = health(State(state)).await;
        let serialized = serde_json::to_string(&response).unwrap();

        assert!(
            serialized.contains("mini_gpt.step-5000.mpk"),
            "expected basename in response, got: {serialized}"
        );
        let parent_str = parent_dir.to_string_lossy();
        assert!(
            !serialized.contains(parent_str.as_ref()),
            "absolute path leaked into health response: {serialized}"
        );

        let _ = std::fs::remove_file(checkpoint_path);
        let _ = std::fs::remove_dir(parent_dir);
    }

    #[tokio::test]
    async fn health_reports_explicit_checkpoint_sha256_matches_disk() {
        use crate::model::persistence::sha256_file_hex;
        type TestBackend = NdArray<f32, i64>;

        let checkpoint_path = std::env::temp_dir().join(format!(
            "rusty-gpt-health-checkpoint-{}.mpk",
            std::process::id()
        ));
        std::fs::write(&checkpoint_path, b"fake-mpk-bytes-for-test").unwrap();
        let expected_sha = sha256_file_hex(&checkpoint_path).unwrap().unwrap();
        let expected_basename = checkpoint_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();

        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device);
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        let provenance = ServerProvenance {
            started_at: Instant::now(),
            checkpoint_source: CheckpointSource::Explicit,
            checkpoint_basename: Some(expected_basename.clone()),
            checkpoint_sha256: Some(expected_sha.clone()),
            tokenizer_sha256: None,
        };
        let state = Arc::new(ServerState::new(
            model,
            tokenizer,
            device,
            EventLogger::stdout(LogFormat::Plain),
            provenance,
        ));

        let Json(response) = health(State(state)).await;

        assert!(response.checkpoint.loaded);
        assert_eq!("explicit", response.checkpoint.source);
        assert_eq!(Some(expected_basename), response.checkpoint.basename);
        assert_eq!(Some(expected_sha), response.checkpoint.sha256);

        let _ = std::fs::remove_file(checkpoint_path);
    }

    #[tokio::test]
    async fn health_model_shape_matches_constructor_args() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        // MiniGpt::new(vocab_size, d_model, num_layers, block_size, num_heads, device)
        let model = MiniGpt::<TestBackend>::new(11, 16, 3, 7, 4, &device);
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        let state = Arc::new(ServerState::new(
            model,
            tokenizer,
            device,
            EventLogger::stdout(LogFormat::Plain),
            ServerProvenance::fresh(),
        ));

        let Json(response) = health(State(state)).await;

        assert_eq!("minigpt", response.model.kind);
        assert_eq!(11, response.model.vocab_size);
        assert_eq!(16, response.model.embed_dim);
        assert_eq!(3, response.model.num_layers);
        assert_eq!(7, response.model.block_size);
        assert_eq!(4, response.model.num_heads);
    }

    #[tokio::test]
    async fn health_fresh_template_reports_none_source() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device);
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        let state = Arc::new(ServerState::new(
            model,
            tokenizer,
            device,
            EventLogger::stdout(LogFormat::Plain),
            ServerProvenance::fresh(),
        ));

        let Json(response) = health(State(state)).await;

        assert!(!response.checkpoint.loaded);
        assert_eq!("none", response.checkpoint.source);
        assert!(response.checkpoint.basename.is_none());
        assert!(response.checkpoint.sha256.is_none());

        let serialized = serde_json::to_value(&response).unwrap();
        assert!(serialized["checkpoint"]["basename"].is_null());
        assert!(serialized["checkpoint"]["sha256"].is_null());
    }

    #[tokio::test]
    async fn health_reports_status_ok_and_uptime() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device);
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        let state = Arc::new(ServerState::new(
            model,
            tokenizer,
            device,
            EventLogger::stdout(LogFormat::Plain),
            ServerProvenance::fresh(),
        ));

        let Json(response) = health(State(state)).await;

        assert_eq!("ok", response.status);
        assert!(response.uptime_seconds < 5);
    }

    #[tokio::test]
    async fn generate_rejects_zero_top_k() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device);
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        let logger = EventLogger::stdout(LogFormat::Plain);
        let state = Arc::new(ServerState::new(
            model,
            tokenizer,
            device,
            logger,
            ServerProvenance::fresh(),
        ));

        let response = generate(
            State(state),
            Json(GenerateRequest {
                prompt: "ab".to_string(),
                max_tokens: 1,
                temperature: 1.0,
                top_k: Some(0),
            }),
        )
        .await;

        let err = response.expect_err("zero top_k should be rejected");
        assert_eq!(StatusCode::BAD_REQUEST, err.0);
        assert_eq!("top_k must be greater than zero", err.1);
    }
}
