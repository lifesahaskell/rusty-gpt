use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{
    Extension, Json, Router,
    extract::{ConnectInfo, State},
};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use tower_http::limit::RequestBodyLimitLayer;

use crate::model::{GenerationOptions, MiniGpt, MoeForwardAux, MoeGpt};
use crate::observability::{EventLogger, RuntimeEvent};
use crate::tokenizer::RuntimeTokenizer;

pub mod training;

pub use training::{
    DEFAULT_RUNS_DIR, TrainRequest, TrainRunRecord, TrainRunStatus, TrainingRunOutcome,
    TrainingRunner,
};
use training::{TRAIN_REQUEST_BODY_LIMIT_BYTES, TRAINING_RETRY_AFTER_SECONDS, TrainingState};

pub type SharedServerState<B> = Arc<ServerState<B>>;

pub enum ServedModel<B: Backend> {
    MiniGpt(MiniGpt<B>),
    MoeGpt(MoeGpt<B>),
}

impl<B: Backend> From<MiniGpt<B>> for ServedModel<B> {
    fn from(model: MiniGpt<B>) -> Self {
        Self::MiniGpt(model)
    }
}

impl<B: Backend> From<MoeGpt<B>> for ServedModel<B> {
    fn from(model: MoeGpt<B>) -> Self {
        Self::MoeGpt(model)
    }
}

impl<B: Backend> ServedModel<B> {
    fn kind(&self) -> &'static str {
        match self {
            Self::MiniGpt(_) => "minigpt",
            Self::MoeGpt(_) => "moe-gpt",
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::MiniGpt(model) => model.vocab_size(),
            Self::MoeGpt(model) => model.vocab_size(),
        }
    }

    fn block_size(&self) -> usize {
        match self {
            Self::MiniGpt(model) => model.block_size(),
            Self::MoeGpt(model) => model.block_size(),
        }
    }

    fn num_layers(&self) -> usize {
        match self {
            Self::MiniGpt(model) => model.num_layers(),
            Self::MoeGpt(model) => model.num_layers(),
        }
    }

    fn num_heads(&self) -> usize {
        match self {
            Self::MiniGpt(model) => model.num_heads(),
            Self::MoeGpt(model) => model.num_heads(),
        }
    }

    fn d_model(&self) -> usize {
        match self {
            Self::MiniGpt(model) => model.d_model(),
            Self::MoeGpt(model) => model.d_model(),
        }
    }

    fn num_experts(&self) -> usize {
        match self {
            Self::MiniGpt(_) => 0,
            Self::MoeGpt(model) => model.num_experts(),
        }
    }

    fn moe_top_k(&self) -> usize {
        match self {
            Self::MiniGpt(_) => 0,
            Self::MoeGpt(model) => model.moe_top_k(),
        }
    }

    pub fn generate(
        &self,
        prompt: &[usize],
        max_tokens: usize,
        device: &B::Device,
    ) -> Result<Vec<usize>, String> {
        match self {
            Self::MiniGpt(model) => model.generate(prompt, max_tokens, device),
            Self::MoeGpt(model) => model.generate(prompt, max_tokens, device),
        }
    }

    fn generate_with_cache_options(
        &self,
        prompt: &[usize],
        max_tokens: usize,
        device: &B::Device,
        options: GenerationOptions,
    ) -> Result<Vec<usize>, String> {
        match self {
            Self::MiniGpt(model) => {
                model.generate_with_cache_options(prompt, max_tokens, device, options)
            }
            Self::MoeGpt(model) => {
                model.generate_with_cache_options(prompt, max_tokens, device, options)
            }
        }
    }
}

pub struct ServerState<B: Backend> {
    /// The served weights. Behind a lock because a completed `POST /api/train`
    /// run swaps a freshly trained model in underneath live traffic.
    ///
    /// ponytail: one global lock held for the length of a single generation.
    /// Generation is single-model and already serialized by the GPU/CPU it
    /// runs on, so contention here costs nothing today. If concurrent
    /// generation ever matters, swap this for an `ArcSwap`-style pointer so
    /// readers never block on the swap.
    model: Mutex<ServedModel<B>>,
    tokenizer: RuntimeTokenizer,
    device: B::Device,
    logger: EventLogger,
    provenance: ServerProvenance,
    limits: ServerLimits,
    rate_limiter: Mutex<RateLimiter>,
    training: TrainingState<B>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ServerLimits {
    pub max_prompt_bytes: usize,
    pub max_output_tokens: usize,
    pub rate_limit_rps: usize,
    pub rate_limit_burst: usize,
    /// `POST /api/train` `train_steps` cap.
    pub max_train_steps: usize,
    /// `POST /api/train` `learning_rate` cap.
    pub max_train_learning_rate: f64,
}

impl Default for ServerLimits {
    fn default() -> Self {
        Self {
            max_prompt_bytes: 8192,
            max_output_tokens: 512,
            rate_limit_rps: 5,
            rate_limit_burst: 10,
            max_train_steps: 100_000,
            max_train_learning_rate: 1.0,
        }
    }
}

impl ServerLimits {
    pub fn max_request_body_bytes(self) -> usize {
        self.max_prompt_bytes.saturating_add(4096)
    }
}

#[derive(Debug)]
struct RateLimiter {
    buckets: HashMap<IpAddr, TokenBucket>,
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    fn try_acquire(&mut self, peer_ip: IpAddr, limits: ServerLimits, now: Instant) -> RateDecision {
        if limits.rate_limit_rps == 0 {
            return RateDecision::Allowed;
        }

        let bucket = self
            .buckets
            .entry(peer_ip)
            .or_insert_with(|| TokenBucket::new(limits.rate_limit_burst, now));
        bucket.try_acquire(limits, now)
    }
}

#[derive(Debug)]
struct TokenBucket {
    tokens: f64,
    updated_at: Instant,
}

impl TokenBucket {
    fn new(burst: usize, now: Instant) -> Self {
        Self {
            tokens: burst as f64,
            updated_at: now,
        }
    }

    fn try_acquire(&mut self, limits: ServerLimits, now: Instant) -> RateDecision {
        let elapsed = now.saturating_duration_since(self.updated_at);
        self.updated_at = now;
        let refill = elapsed.as_secs_f64() * limits.rate_limit_rps as f64;
        self.tokens = (self.tokens + refill).min(limits.rate_limit_burst as f64);

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            RateDecision::Allowed
        } else {
            let missing = 1.0 - self.tokens;
            let retry_after = (missing / limits.rate_limit_rps as f64).ceil().max(1.0) as u64;
            RateDecision::Limited {
                retry_after_seconds: retry_after,
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum RateDecision {
    Allowed,
    Limited { retry_after_seconds: u64 },
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
    pub fn new<M>(
        model: M,
        tokenizer: RuntimeTokenizer,
        device: B::Device,
        logger: EventLogger,
        provenance: ServerProvenance,
    ) -> Self
    where
        M: Into<ServedModel<B>>,
    {
        Self::new_with_limits(
            model,
            tokenizer,
            device,
            logger,
            provenance,
            ServerLimits::default(),
        )
    }

    pub fn new_with_limits<M>(
        model: M,
        tokenizer: RuntimeTokenizer,
        device: B::Device,
        logger: EventLogger,
        provenance: ServerProvenance,
        limits: ServerLimits,
    ) -> Self
    where
        M: Into<ServedModel<B>>,
    {
        Self {
            model: Mutex::new(model.into()),
            tokenizer,
            device,
            logger,
            provenance,
            limits,
            rate_limiter: Mutex::new(RateLimiter::new()),
            training: TrainingState::default(),
        }
    }

    /// Enable `POST /api/train` with the runner that performs the actual
    /// training, and the directory its run manifests are written to. Without
    /// this the route answers `503`.
    pub fn with_training_runner(
        mut self,
        runner: Arc<dyn TrainingRunner<B>>,
        runs_dir: std::path::PathBuf,
    ) -> Self {
        self.training = TrainingState::with_runner(runner, runs_dir);
        self
    }

    /// Read access to the served model.
    ///
    /// Lock poisoning is recovered rather than propagated: the model is only
    /// ever *replaced* (a single assignment), never mutated in place, so a
    /// panic elsewhere cannot leave it half-written — and `/api/health` must
    /// keep answering after an unrelated handler panics.
    fn model(&self) -> std::sync::MutexGuard<'_, ServedModel<B>> {
        self.model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Swap in newly trained weights. Called once, at the end of a successful
    /// training run — never while the training loop is running.
    pub fn replace_model(&self, model: ServedModel<B>) {
        *self.model() = model;
    }

    /// The most recent training run's record, running or finished. `None`
    /// until the first `POST /api/train` of this process.
    pub fn training_run(&self) -> Option<TrainRunRecord> {
        self.training.current()
    }

    pub fn training_in_progress(&self) -> bool {
        self.training.is_running()
    }

    pub fn model_vocab_size(&self) -> usize {
        self.model().vocab_size()
    }

    pub fn model_block_size(&self) -> usize {
        self.model().block_size()
    }

    pub fn tokenizer_vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    pub fn model_tokenizer_vocab_match(&self) -> bool {
        self.model().vocab_size() == self.tokenizer.vocab_size()
    }
}

pub fn router_with_limits<B>(limits: ServerLimits) -> Router<SharedServerState<B>>
where
    B: Backend + Send + Sync + 'static,
    B::Device: Send + Sync + 'static,
{
    Router::new()
        .route(
            "/generate",
            post(generate::<B>).layer(RequestBodyLimitLayer::new(limits.max_request_body_bytes())),
        )
        // SECURITY: no authentication on this route. It starts a job that
        // consumes the whole box and overwrites `checkpoints/mini_gpt`, so
        // this server is safe on localhost only until auth lands.
        .route(
            "/train",
            post(training::train::<B>)
                .layer(RequestBodyLimitLayer::new(TRAIN_REQUEST_BODY_LIMIT_BYTES)),
        )
        // Read-only, and polled once a second by the UI — kept off the
        // generate rate limiter for the same reason /health is.
        .route("/train/{run_id}/status", get(training::status::<B>))
        // Stopping is a signal, not a payload: no body, so no body limit.
        .route("/train/{run_id}", delete(training::stop_train::<B>))
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing: Option<Vec<RoutingData>>,
}

#[derive(Debug, Serialize)]
pub struct AttentionData {
    pub layer: usize,
    pub head: usize,
    pub weights: Vec<Vec<f64>>,
}

#[derive(Debug, Serialize)]
pub struct RoutingData {
    pub layer: usize,
    pub experts: Vec<Vec<usize>>,
    pub weights: Vec<Vec<f64>>,
}

#[derive(Debug, Serialize)]
pub struct InfoResponse {
    pub model_kind: &'static str,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub block_size: usize,
    pub num_experts: usize,
    pub moe_top_k: usize,
    pub tokenizer_vocab_size: usize,
    pub model_tokenizer_vocab_match: bool,
}

async fn generate<B>(
    State(state): State<SharedServerState<B>>,
    peer: Option<Extension<ConnectInfo<SocketAddr>>>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, GenerateError>
where
    B: Backend + Send + Sync + 'static,
    B::Device: Send + Sync + 'static,
{
    let started_at = Instant::now();
    state.logger.log(RuntimeEvent::GenerateRequestAccepted {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        prompt_chars: request.prompt.chars().count(),
    });

    // A training run owns the model for its duration. Rejecting outright is
    // deliberate: snapshotting a pre-training copy would double the resident
    // model just to keep a degraded endpoint alive.
    if state.training_in_progress() {
        return Err(logged_error(
            &state.logger,
            GenerateError::TrainingInProgress {
                retry_after_seconds: TRAINING_RETRY_AFTER_SECONDS,
            },
            started_at,
        ));
    }
    if request.prompt.is_empty() {
        return Err(logged_bad_request(
            &state.logger,
            "prompt must not be empty",
            started_at,
        ));
    }
    let prompt_bytes = request.prompt.len();
    if prompt_bytes > state.limits.max_prompt_bytes {
        return Err(logged_error(
            &state.logger,
            GenerateError::PromptTooLarge {
                max_bytes: state.limits.max_prompt_bytes,
                actual_bytes: prompt_bytes,
            },
            started_at,
        ));
    }
    if request.max_tokens == 0 || request.max_tokens > state.limits.max_output_tokens {
        return Err(logged_error(
            &state.logger,
            GenerateError::MaxTokensOutOfRange {
                max_allowed: state.limits.max_output_tokens,
                requested: request.max_tokens,
            },
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
    let generation_options = GenerationOptions::sampling(request.temperature, request.top_k)
        .map_err(|err| logged_bad_request(&state.logger, err, started_at))?;

    let peer_ip = peer
        .as_ref()
        .map(|Extension(ConnectInfo(addr))| addr.ip())
        .unwrap_or(IpAddr::from([127, 0, 0, 1]));
    let rate_decision = {
        let mut limiter = state
            .rate_limiter
            .lock()
            .map_err(|_| GenerateError::Internal("rate limiter lock was poisoned".to_string()))?;
        limiter.try_acquire(peer_ip, state.limits, Instant::now())
    };
    if let RateDecision::Limited {
        retry_after_seconds,
    } = rate_decision
    {
        return Err(logged_error(
            &state.logger,
            GenerateError::RateLimited {
                retry_after_seconds,
            },
            started_at,
        ));
    }

    let generated_tokens = state
        .model()
        .generate_with_cache_options(
            &prompt_tokens,
            request.max_tokens,
            &state.device,
            generation_options,
        )
        .map_err(|err| logged_bad_request(&state.logger, err, started_at))?;
    let attention_tokens = context_window(&generated_tokens, state.model().block_size());
    let (attention, routing) = attention_for_tokens(&state, attention_tokens)?;
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
        routing,
    }))
}

#[derive(Debug, Serialize)]
#[serde(tag = "error", rename_all = "snake_case")]
enum GenerateErrorBody {
    PromptTooLarge {
        max_bytes: usize,
        actual_bytes: usize,
    },
    MaxTokensOutOfRange {
        max_allowed: usize,
        requested: usize,
    },
    RateLimited {
        retry_after_seconds: u64,
    },
    TrainingInProgress {
        retry_after_seconds: u64,
    },
    BadRequest {
        message: String,
    },
    Internal {
        message: String,
    },
}

#[derive(Debug)]
enum GenerateError {
    BadRequest(String),
    PromptTooLarge {
        max_bytes: usize,
        actual_bytes: usize,
    },
    MaxTokensOutOfRange {
        max_allowed: usize,
        requested: usize,
    },
    RateLimited {
        retry_after_seconds: u64,
    },
    /// A training run holds the model; try again once it finishes.
    TrainingInProgress {
        retry_after_seconds: u64,
    },
    Internal(String),
}

impl GenerateError {
    fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_)
            | Self::PromptTooLarge { .. }
            | Self::MaxTokensOutOfRange { .. } => StatusCode::BAD_REQUEST,
            Self::RateLimited { .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::TrainingInProgress { .. } => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn reason(&self) -> String {
        match self {
            Self::BadRequest(message) => message.clone(),
            Self::PromptTooLarge {
                max_bytes,
                actual_bytes,
            } => format!("prompt_too_large max_bytes={max_bytes} actual_bytes={actual_bytes}"),
            Self::MaxTokensOutOfRange {
                max_allowed,
                requested,
            } => {
                format!("max_tokens_out_of_range max_allowed={max_allowed} requested={requested}")
            }
            Self::RateLimited {
                retry_after_seconds,
            } => format!("rate_limited retry_after_seconds={retry_after_seconds}"),
            Self::TrainingInProgress {
                retry_after_seconds,
            } => format!("training_in_progress retry_after_seconds={retry_after_seconds}"),
            Self::Internal(message) => message.clone(),
        }
    }

    fn body(&self) -> GenerateErrorBody {
        match self {
            Self::BadRequest(message) => GenerateErrorBody::BadRequest {
                message: message.clone(),
            },
            Self::PromptTooLarge {
                max_bytes,
                actual_bytes,
            } => GenerateErrorBody::PromptTooLarge {
                max_bytes: *max_bytes,
                actual_bytes: *actual_bytes,
            },
            Self::MaxTokensOutOfRange {
                max_allowed,
                requested,
            } => GenerateErrorBody::MaxTokensOutOfRange {
                max_allowed: *max_allowed,
                requested: *requested,
            },
            Self::RateLimited {
                retry_after_seconds,
            } => GenerateErrorBody::RateLimited {
                retry_after_seconds: *retry_after_seconds,
            },
            Self::TrainingInProgress {
                retry_after_seconds,
            } => GenerateErrorBody::TrainingInProgress {
                retry_after_seconds: *retry_after_seconds,
            },
            Self::Internal(message) => GenerateErrorBody::Internal {
                message: message.clone(),
            },
        }
    }
}

impl IntoResponse for GenerateError {
    fn into_response(self) -> Response {
        let status = self.status();
        let retry_after = match self {
            Self::RateLimited {
                retry_after_seconds,
            }
            | Self::TrainingInProgress {
                retry_after_seconds,
            } => Some(retry_after_seconds),
            _ => None,
        };
        let mut response = (status, Json(self.body())).into_response();
        if let Some(seconds) = retry_after
            && let Ok(value) = HeaderValue::from_str(&seconds.to_string())
        {
            response.headers_mut().insert(header::RETRY_AFTER, value);
        }
        response
    }
}

fn bad_request(message: impl Into<String>) -> GenerateError {
    GenerateError::BadRequest(message.into())
}

fn logged_bad_request(
    logger: &EventLogger,
    message: impl Into<String>,
    started_at: Instant,
) -> GenerateError {
    let error = bad_request(message);
    logged_error(logger, error, started_at)
}

fn logged_error(logger: &EventLogger, error: GenerateError, started_at: Instant) -> GenerateError {
    logger.log(RuntimeEvent::GenerateRequestRejected {
        status: error.status().as_u16(),
        reason: error.reason(),
        elapsed_ms: started_at.elapsed().as_millis(),
    });
    error
}

async fn info<B>(State(state): State<SharedServerState<B>>) -> Json<InfoResponse>
where
    B: Backend,
{
    let model = state.model();
    Json(InfoResponse {
        model_kind: model.kind(),
        vocab_size: model.vocab_size(),
        num_layers: model.num_layers(),
        num_heads: model.num_heads(),
        block_size: model.block_size(),
        num_experts: model.num_experts(),
        moe_top_k: model.moe_top_k(),
        tokenizer_vocab_size: state.tokenizer.vocab_size(),
        model_tokenizer_vocab_match: model.vocab_size() == state.tokenizer.vocab_size(),
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
    pub num_experts: usize,
    pub moe_top_k: usize,
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
    let model = state.model();
    Json(HealthResponse {
        status: "ok",
        uptime_seconds: provenance.started_at.elapsed().as_secs(),
        model: HealthModel {
            kind: model.kind(),
            embed_dim: model.d_model(),
            num_heads: model.num_heads(),
            num_layers: model.num_layers(),
            block_size: model.block_size(),
            vocab_size: model.vocab_size(),
            num_experts: model.num_experts(),
            moe_top_k: model.moe_top_k(),
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
) -> Result<(Vec<AttentionData>, Option<Vec<RoutingData>>), GenerateError> {
    let input: Vec<i64> = tokens.iter().map(|&token| token as i64).collect();
    let token_tensor: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(input, [1, tokens.len()]), &state.device);
    let (attentions, routing) = match &*state.model() {
        ServedModel::MiniGpt(model) => {
            let (_logits, attentions) = model.forward_tokens_with_attention(token_tensor);
            (attentions, None)
        }
        ServedModel::MoeGpt(model) => {
            let output = model.forward_tokens_with_attention_and_routing(token_tensor);
            (output.attentions, Some(output.routing))
        }
    };

    let mut attention_data = Vec::new();
    for (layer, attention) in attentions.into_iter().enumerate() {
        let [batch_size, num_heads, seq_len, _] = attention.shape().dims();
        if batch_size != 1 {
            return Err(GenerateError::Internal(format!(
                "expected attention batch size 1, got {batch_size}"
            )));
        }

        let values = attention.into_data().to_vec::<f32>().map_err(|err| {
            GenerateError::Internal(format!("failed to serialize attention tensor: {err}"))
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

    let routing_data = routing
        .map(|routing| serialize_routing(routing))
        .transpose()?;

    Ok((attention_data, routing_data))
}

fn serialize_routing<B: Backend>(
    routing: Vec<MoeForwardAux<B>>,
) -> Result<Vec<RoutingData>, GenerateError> {
    let mut data = Vec::with_capacity(routing.len());
    for (layer, aux) in routing.into_iter().enumerate() {
        let [batch_size, seq_len, top_k] = aux.top_k_indices.shape().dims();
        if batch_size != 1 {
            return Err(GenerateError::Internal(format!(
                "expected routing batch size 1, got {batch_size}"
            )));
        }
        let indices = aux
            .top_k_indices
            .into_data()
            .to_vec::<i64>()
            .map_err(|err| {
                GenerateError::Internal(format!("failed to serialize routing indices: {err}"))
            })?;
        let weights = aux
            .top_k_weights
            .into_data()
            .to_vec::<f32>()
            .map_err(|err| {
                GenerateError::Internal(format!("failed to serialize routing weights: {err}"))
            })?;
        let mut expert_rows = Vec::with_capacity(seq_len);
        let mut weight_rows = Vec::with_capacity(seq_len);
        for token in 0..seq_len {
            let start = token * top_k;
            let end = start + top_k;
            expert_rows.push(
                indices[start..end]
                    .iter()
                    .map(|&idx| idx as usize)
                    .collect(),
            );
            weight_rows.push(
                weights[start..end]
                    .iter()
                    .map(|&weight| weight as f64)
                    .collect(),
            );
        }
        data.push(RoutingData {
            layer,
            experts: expert_rows,
            weights: weight_rows,
        });
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::LogFormat;
    use axum::body::{Body, to_bytes};
    use axum::http::{Method, Request};
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use std::sync::Mutex;
    use tower::ServiceExt;

    type TestBackend = NdArray<f32, i64>;

    fn test_state(limits: ServerLimits) -> Arc<ServerState<TestBackend>> {
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 6, 2, &device);
        let tokenizer = RuntimeTokenizer::char_from_text("abcdefg");
        Arc::new(ServerState::new_with_limits(
            model,
            tokenizer,
            device,
            EventLogger::stdout(LogFormat::Plain),
            ServerProvenance::fresh(),
            limits,
        ))
    }

    async fn post_generate(
        state: Arc<ServerState<TestBackend>>,
        limits: ServerLimits,
        body: serde_json::Value,
    ) -> (StatusCode, serde_json::Value, axum::http::HeaderMap) {
        let response = router_with_limits::<TestBackend>(limits)
            .with_state(state)
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/generate")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, body, headers)
    }

    async fn get_route(
        state: Arc<ServerState<TestBackend>>,
        limits: ServerLimits,
        uri: &str,
    ) -> StatusCode {
        router_with_limits::<TestBackend>(limits)
            .with_state(state)
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
            .status()
    }

    #[test]
    fn checkpoint_source_serializes_to_spec_strings() {
        assert_eq!("none", CheckpointSource::None.as_str());
        assert_eq!("explicit", CheckpointSource::Explicit.as_str());
        assert_eq!("latest", CheckpointSource::Latest.as_str());
    }

    #[test]
    fn rate_limiter_tracks_peer_ips_independently() {
        let limits = ServerLimits {
            rate_limit_rps: 1,
            rate_limit_burst: 1,
            ..ServerLimits::default()
        };
        let now = Instant::now();
        let mut limiter = RateLimiter::new();

        assert_eq!(
            RateDecision::Allowed,
            limiter.try_acquire(IpAddr::from([127, 0, 0, 1]), limits, now)
        );
        assert_eq!(
            RateDecision::Limited {
                retry_after_seconds: 1
            },
            limiter.try_acquire(IpAddr::from([127, 0, 0, 1]), limits, now)
        );
        assert_eq!(
            RateDecision::Allowed,
            limiter.try_acquire(IpAddr::from([127, 0, 0, 2]), limits, now)
        );
    }

    #[tokio::test]
    async fn generate_rejects_prompt_one_byte_over_configured_limit() {
        let limits = ServerLimits {
            max_prompt_bytes: 2,
            rate_limit_rps: 0,
            ..ServerLimits::default()
        };
        let (status, body, _) = post_generate(
            test_state(limits),
            limits,
            serde_json::json!({
                "prompt": "abc",
                "max_tokens": 1,
                "temperature": 1.0,
                "top_k": null
            }),
        )
        .await;

        assert_eq!(StatusCode::BAD_REQUEST, status);
        assert_eq!(
            serde_json::json!({"error":"prompt_too_large","max_bytes":2,"actual_bytes":3}),
            body
        );
    }

    #[tokio::test]
    async fn generate_accepts_prompt_exactly_at_configured_limit() {
        let limits = ServerLimits {
            max_prompt_bytes: 2,
            max_output_tokens: 1,
            rate_limit_rps: 0,
            ..ServerLimits::default()
        };
        let (status, _, _) = post_generate(
            test_state(limits),
            limits,
            serde_json::json!({
                "prompt": "ab",
                "max_tokens": 1,
                "temperature": 1.0,
                "top_k": null
            }),
        )
        .await;

        assert_eq!(StatusCode::OK, status);
    }

    #[tokio::test]
    async fn generate_rejects_max_tokens_above_configured_limit() {
        let limits = ServerLimits {
            max_output_tokens: 1,
            rate_limit_rps: 0,
            ..ServerLimits::default()
        };
        let (status, body, _) = post_generate(
            test_state(limits),
            limits,
            serde_json::json!({
                "prompt": "ab",
                "max_tokens": 2,
                "temperature": 1.0,
                "top_k": null
            }),
        )
        .await;

        assert_eq!(StatusCode::BAD_REQUEST, status);
        assert_eq!(
            serde_json::json!({"error":"max_tokens_out_of_range","max_allowed":1,"requested":2}),
            body
        );
    }

    #[tokio::test]
    async fn generate_rejects_zero_max_tokens_with_structured_error() {
        let limits = ServerLimits {
            max_output_tokens: 1,
            rate_limit_rps: 0,
            ..ServerLimits::default()
        };
        let (status, body, _) = post_generate(
            test_state(limits),
            limits,
            serde_json::json!({
                "prompt": "ab",
                "max_tokens": 0,
                "temperature": 1.0,
                "top_k": null
            }),
        )
        .await;

        assert_eq!(StatusCode::BAD_REQUEST, status);
        assert_eq!(
            serde_json::json!({"error":"max_tokens_out_of_range","max_allowed":1,"requested":0}),
            body
        );
    }

    #[tokio::test]
    async fn generate_accepts_max_tokens_exactly_at_configured_limit() {
        let limits = ServerLimits {
            max_prompt_bytes: 8,
            max_output_tokens: 1,
            rate_limit_rps: 0,
            ..ServerLimits::default()
        };
        let (status, _, _) = post_generate(
            test_state(limits),
            limits,
            serde_json::json!({
                "prompt": "ab",
                "max_tokens": 1,
                "temperature": 1.0,
                "top_k": null
            }),
        )
        .await;

        assert_eq!(StatusCode::OK, status);
    }

    #[tokio::test]
    async fn invalid_cap_requests_do_not_consume_rate_limit_tokens() {
        let limits = ServerLimits {
            max_output_tokens: 1,
            rate_limit_rps: 1,
            rate_limit_burst: 1,
            ..ServerLimits::default()
        };
        let state = test_state(limits);

        for _ in 0..3 {
            let (status, _, _) = post_generate(
                Arc::clone(&state),
                limits,
                serde_json::json!({
                    "prompt": "ab",
                    "max_tokens": 2,
                    "temperature": 1.0,
                    "top_k": null
                }),
            )
            .await;
            assert_eq!(StatusCode::BAD_REQUEST, status);
        }

        let (status, _, _) = post_generate(
            state,
            limits,
            serde_json::json!({
                "prompt": "ab",
                "max_tokens": 1,
                "temperature": 1.0,
                "top_k": null
            }),
        )
        .await;

        assert_eq!(StatusCode::OK, status);
    }

    #[tokio::test]
    async fn tokenizer_rejected_requests_do_not_consume_rate_limit_tokens() {
        let limits = ServerLimits {
            max_output_tokens: 1,
            rate_limit_rps: 1,
            rate_limit_burst: 1,
            ..ServerLimits::default()
        };
        let state = test_state(limits);

        let (status, _, _) = post_generate(
            Arc::clone(&state),
            limits,
            serde_json::json!({
                "prompt": "z",
                "max_tokens": 1,
                "temperature": 1.0,
                "top_k": null
            }),
        )
        .await;
        assert_eq!(StatusCode::BAD_REQUEST, status);

        let (status, _, _) = post_generate(
            state,
            limits,
            serde_json::json!({
                "prompt": "ab",
                "max_tokens": 1,
                "temperature": 1.0,
                "top_k": null
            }),
        )
        .await;
        assert_eq!(StatusCode::OK, status);
    }

    #[tokio::test]
    async fn rate_limit_allows_burst_then_returns_retry_after() {
        let limits = ServerLimits {
            max_output_tokens: 1,
            rate_limit_rps: 1,
            rate_limit_burst: 2,
            ..ServerLimits::default()
        };
        let state = test_state(limits);
        let mut statuses = Vec::new();
        let mut retry_after = None;

        for _ in 0..4 {
            let (status, body, headers) = post_generate(
                Arc::clone(&state),
                limits,
                serde_json::json!({
                    "prompt": "ab",
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "top_k": null
                }),
            )
            .await;
            if status == StatusCode::TOO_MANY_REQUESTS {
                assert_eq!(
                    serde_json::json!({"error":"rate_limited","retry_after_seconds":1}),
                    body
                );
                retry_after = headers.get(header::RETRY_AFTER).cloned();
            }
            statuses.push(status);
        }

        assert_eq!(
            vec![
                StatusCode::OK,
                StatusCode::OK,
                StatusCode::TOO_MANY_REQUESTS,
                StatusCode::TOO_MANY_REQUESTS
            ],
            statuses
        );
        assert_eq!(Some(HeaderValue::from_static("1")), retry_after);
    }

    #[tokio::test]
    async fn info_and_health_are_exempt_from_generate_body_limit_and_rate_limit() {
        let limits = ServerLimits {
            max_prompt_bytes: 1,
            rate_limit_rps: 1,
            rate_limit_burst: 1,
            ..ServerLimits::default()
        };
        let state = test_state(limits);

        assert_eq!(
            StatusCode::OK,
            get_route(Arc::clone(&state), limits, "/info").await
        );
        assert_eq!(StatusCode::OK, get_route(state, limits, "/health").await);
    }

    #[tokio::test]
    async fn generate_route_rejects_request_body_over_prompt_limit_plus_headroom() {
        let limits = ServerLimits {
            max_prompt_bytes: 1,
            rate_limit_rps: 0,
            ..ServerLimits::default()
        };
        let prompt = "a".repeat(limits.max_request_body_bytes());
        let response = router_with_limits::<TestBackend>(limits)
            .with_state(test_state(limits))
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/generate")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "prompt": prompt,
                            "max_tokens": 1,
                            "temperature": 1.0,
                            "top_k": null
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(StatusCode::PAYLOAD_TOO_LARGE, response.status());
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

        assert_eq!(7, state.model().vocab_size());
        assert_eq!(2, state.model().num_layers());
        assert_eq!(2, state.model().num_heads());
        assert_eq!(6, state.model().block_size());
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

        let (attention, routing) = attention_for_tokens(&state, &[0, 1, 2]).unwrap();

        assert!(routing.is_none());
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
            None,
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
            None,
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
            None,
            Json(GenerateRequest {
                prompt: "ab".to_string(),
                max_tokens: 1,
                temperature: 1.0,
                top_k: Some(0),
            }),
        )
        .await;

        let err = response.expect_err("zero top_k should be rejected");
        assert_eq!(StatusCode::BAD_REQUEST, err.status());
        assert_eq!("top_k must be greater than zero", err.reason());
    }
}
