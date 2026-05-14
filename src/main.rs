use anyhow::{Context, Result, bail};
use axum::Router;
use burn::backend::Autodiff;
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "cuda")]
use burn::backend::cuda::CudaDevice;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::backend::Backend;
use rusty_gpt::loader::data::DataLoader;
use rusty_gpt::loader::huggingface;
use rusty_gpt::model::persistence::{
    CheckpointMetadata, CheckpointModelShape, CheckpointTokenizer, CheckpointTrainingMetrics,
    CheckpointTrainingRun, load_model_with_metadata_validation, save_checkpoint_metadata,
    save_model, sha256_file_hex,
};
use rusty_gpt::model::{
    MiniGpt, MiniGptConfig, MultiAttentionModel, SingleAttentionModel, TrainingLogContext,
    TrainingOutcome, TrainingParams, TrivialModel,
};
use rusty_gpt::observability::{EventLogger, LogFormat, RuntimeEvent};
use rusty_gpt::server;
use rusty_gpt::server::ServerState;
use rusty_gpt::tokenizer::RuntimeTokenizer;
use rusty_gpt::utils::{BenchmarkConfig, benchmark_generation, parse_usize_list};
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_INPUT_PATH: &str = "data/input.txt";
const DEFAULT_MINIGPT_CHECKPOINT_PATH: &str = "checkpoints/mini_gpt";
const DEFAULT_BPE_TOKENIZER_PATH: &str = "checkpoints/tokenizer.json";
const BPE_TOKENIZER_ENV: &str = "RUSTY_GPT_BPE_TOKENIZER";
const LOG_FORMAT_ENV: &str = "RUSTY_GPT_LOG_FORMAT";
const BENCHMARK_PROMPT_LENS_ENV: &str = "RUSTY_GPT_BENCHMARK_PROMPT_LENS";
const BENCHMARK_GEN_LENS_ENV: &str = "RUSTY_GPT_BENCHMARK_GEN_LENS";
const BENCHMARK_WARMUPS_ENV: &str = "RUSTY_GPT_BENCHMARK_WARMUPS";
const BENCHMARK_ITERATIONS_ENV: &str = "RUSTY_GPT_BENCHMARK_ITERATIONS";
const DEFAULT_CHECKPOINT_DIR: &str = "checkpoints";
const DEFAULT_SERVER_ADDR: &str = "127.0.0.1:8787";

const BLOCK_SIZE: usize = 128; // context length, start with 64 or 128
const BATCH_SIZE: usize = 32; // start with 32 for quick training and testing, or 64+ for better results at the cost of more memory and slower training
const EMBED_DIM: usize = 128; // model dimensionality, start with 128 or 256
const NUM_HEADS: usize = 4; // number of attention heads, start with 4 or 8
const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS; // dimensionality of each attention head, typically embed_dim / num_heads
const NUM_LAYERS: usize = 4; // number of transformer layers, start with 4 or 8
const DROPOUT: f64 = 0.0; // dropout probability, start with 0.0 for no dropout, or 0.1 for some regularization
const LEARNING_RATE: f64 = 1e-4; // learning rate, start with 1e-4 or 5e-4
const TRAIN_STEPS: usize = 1000; // number of training steps, start with 1000 for a quick demo, or 10000+ for better results
const EVAL_INTERVAL: usize = 100; // how often to evaluate and print training progress, in steps
const GENERATE_TOKENS: usize = 80; // number of tokens to generate in interactive generation mode
const MINIGPT_GRAD_CLIP_NORM: f32 = 1.0; // max gradient norm for minigpt training
const PREFETCH_BATCHES: usize = 2; // number of prepared CPU batches queued ahead of training

#[derive(Debug, Clone, Copy, PartialEq)]
struct Hyperparameters {
    block_size: usize,
    batch_size: usize,
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    num_layers: usize,
    dropout: f64,
    learning_rate: f64,
    train_steps: usize,
    eval_interval: usize,
    generate_tokens: usize,
    minigpt_grad_clip_norm: f32,
    prefetch_batches: usize,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Self {
            block_size: BLOCK_SIZE,
            batch_size: BATCH_SIZE,
            embed_dim: EMBED_DIM,
            num_heads: NUM_HEADS,
            head_dim: HEAD_DIM,
            num_layers: NUM_LAYERS,
            dropout: DROPOUT,
            learning_rate: LEARNING_RATE,
            train_steps: TRAIN_STEPS,
            eval_interval: EVAL_INTERVAL,
            generate_tokens: GENERATE_TOKENS,
            minigpt_grad_clip_norm: MINIGPT_GRAD_CLIP_NORM,
            prefetch_batches: PREFETCH_BATCHES,
        }
    }
}

impl Hyperparameters {
    #[cfg(test)]
    fn from_env() -> Result<Self> {
        Self::from_env_and_overrides(
            &RuntimeEnv::from_process_env(),
            &HyperparameterOverrides::default(),
        )
    }

    fn from_env_and_overrides(
        env: &RuntimeEnv,
        overrides: &HyperparameterOverrides,
    ) -> Result<Self> {
        let mut hyperparameters = Self::default();

        apply_optional_override(
            "RUSTY_GPT_BLOCK_SIZE",
            env.block_size.as_deref(),
            &mut hyperparameters.block_size,
        )?;
        apply_optional_override(
            "RUSTY_GPT_BATCH_SIZE",
            env.batch_size.as_deref(),
            &mut hyperparameters.batch_size,
        )?;
        apply_optional_override(
            "RUSTY_GPT_EMBED_DIM",
            env.embed_dim.as_deref(),
            &mut hyperparameters.embed_dim,
        )?;
        apply_optional_override(
            "RUSTY_GPT_NUM_HEADS",
            env.num_heads.as_deref(),
            &mut hyperparameters.num_heads,
        )?;
        apply_optional_override(
            "RUSTY_GPT_NUM_LAYERS",
            env.num_layers.as_deref(),
            &mut hyperparameters.num_layers,
        )?;
        apply_optional_override(
            "RUSTY_GPT_DROPOUT",
            env.dropout.as_deref(),
            &mut hyperparameters.dropout,
        )?;
        apply_optional_override(
            "RUSTY_GPT_LEARNING_RATE",
            env.learning_rate.as_deref(),
            &mut hyperparameters.learning_rate,
        )?;
        apply_optional_override(
            "RUSTY_GPT_TRAIN_STEPS",
            env.train_steps.as_deref(),
            &mut hyperparameters.train_steps,
        )?;
        apply_optional_override(
            "RUSTY_GPT_EVAL_INTERVAL",
            env.eval_interval.as_deref(),
            &mut hyperparameters.eval_interval,
        )?;
        apply_optional_override(
            "RUSTY_GPT_GENERATE_TOKENS",
            env.generate_tokens.as_deref(),
            &mut hyperparameters.generate_tokens,
        )?;
        apply_optional_override(
            "RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM",
            env.minigpt_grad_clip_norm.as_deref(),
            &mut hyperparameters.minigpt_grad_clip_norm,
        )?;
        apply_optional_override(
            "RUSTY_GPT_PREFETCH_BATCHES",
            env.prefetch_batches.as_deref(),
            &mut hyperparameters.prefetch_batches,
        )?;

        overrides.apply_to(&mut hyperparameters);
        hyperparameters.validate()?;
        Ok(hyperparameters)
    }

    fn validate(&mut self) -> Result<()> {
        if self.block_size == 0 {
            bail!("block_size must be greater than zero");
        }
        if self.batch_size == 0 {
            bail!("batch_size must be greater than zero");
        }
        if self.embed_dim == 0 {
            bail!("embed_dim must be greater than zero");
        }
        if self.num_heads == 0 {
            bail!("num_heads must be greater than zero");
        }
        if self.num_layers == 0 {
            bail!("num_layers must be greater than zero");
        }
        if !self.embed_dim.is_multiple_of(self.num_heads) {
            bail!("embed_dim must be divisible by num_heads");
        }
        if !(0.0..1.0).contains(&self.dropout) {
            bail!("dropout must be >= 0 and < 1");
        }
        if self.learning_rate <= 0.0 {
            bail!("learning_rate must be greater than zero");
        }
        if self.train_steps == 0 {
            bail!("train_steps must be greater than zero");
        }
        if self.generate_tokens == 0 {
            bail!("generate_tokens must be greater than zero");
        }
        if self.minigpt_grad_clip_norm <= 0.0 {
            bail!("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM must be greater than zero");
        }

        self.head_dim = self.embed_dim / self.num_heads;
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
struct HyperparameterOverrides {
    block_size: Option<usize>,
    batch_size: Option<usize>,
    embed_dim: Option<usize>,
    num_heads: Option<usize>,
    num_layers: Option<usize>,
    dropout: Option<f64>,
    learning_rate: Option<f64>,
    train_steps: Option<usize>,
    eval_interval: Option<usize>,
    generate_tokens: Option<usize>,
    minigpt_grad_clip_norm: Option<f32>,
    prefetch_batches: Option<usize>,
}

impl HyperparameterOverrides {
    fn apply_to(&self, hyperparameters: &mut Hyperparameters) {
        if let Some(value) = self.block_size {
            hyperparameters.block_size = value;
        }
        if let Some(value) = self.batch_size {
            hyperparameters.batch_size = value;
        }
        if let Some(value) = self.embed_dim {
            hyperparameters.embed_dim = value;
        }
        if let Some(value) = self.num_heads {
            hyperparameters.num_heads = value;
        }
        if let Some(value) = self.num_layers {
            hyperparameters.num_layers = value;
        }
        if let Some(value) = self.dropout {
            hyperparameters.dropout = value;
        }
        if let Some(value) = self.learning_rate {
            hyperparameters.learning_rate = value;
        }
        if let Some(value) = self.train_steps {
            hyperparameters.train_steps = value;
        }
        if let Some(value) = self.eval_interval {
            hyperparameters.eval_interval = value;
        }
        if let Some(value) = self.generate_tokens {
            hyperparameters.generate_tokens = value;
        }
        if let Some(value) = self.minigpt_grad_clip_norm {
            hyperparameters.minigpt_grad_clip_norm = value;
        }
        if let Some(value) = self.prefetch_batches {
            hyperparameters.prefetch_batches = value;
        }
    }
}

fn apply_optional_override<T>(name: &str, value: Option<&str>, target: &mut T) -> Result<()>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    if let Some(value) = value {
        *target = value
            .parse()
            .with_context(|| format!("invalid {name} value: {value}"))?;
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl BackendChoice {
    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelChoice {
    Trivial,
    SingleAttention,
    MultiAttention,
    MiniGpt,
    Compare,
}

impl ModelChoice {
    fn comparison_models(self) -> Vec<ModelChoice> {
        match self {
            ModelChoice::Compare => vec![
                ModelChoice::Trivial,
                ModelChoice::SingleAttention,
                ModelChoice::MultiAttention,
                ModelChoice::MiniGpt,
            ],
            model => vec![model],
        }
    }

    fn label(self) -> &'static str {
        match self {
            ModelChoice::Trivial => "trivial",
            ModelChoice::SingleAttention => "single-attention",
            ModelChoice::MultiAttention => "multi-attention",
            ModelChoice::MiniGpt => "minigpt",
            ModelChoice::Compare => "compare",
        }
    }

    fn includes_minigpt(self) -> bool {
        matches!(self, ModelChoice::MiniGpt | ModelChoice::Compare)
    }
}

fn main() -> Result<()> {
    let config = parse_runtime_config_with_checkpoint(
        env::args().skip(1),
        RuntimeEnv {
            backend: env::var("RUSTY_GPT_BACKEND").ok(),
            input: env::var("RUSTY_GPT_INPUT").ok(),
            model: env::var("RUSTY_GPT_MODEL").ok(),
            checkpoint: env::var("RUSTY_GPT_MINIGPT_CHECKPOINT").ok(),
            server_addr: env::var("RUSTY_GPT_SERVER_ADDR").ok(),
            log_format: env::var(LOG_FORMAT_ENV).ok(),
            benchmark_prompt_lens: env::var(BENCHMARK_PROMPT_LENS_ENV).ok(),
            benchmark_gen_lens: env::var(BENCHMARK_GEN_LENS_ENV).ok(),
            benchmark_warmups: env::var(BENCHMARK_WARMUPS_ENV).ok(),
            benchmark_iterations: env::var(BENCHMARK_ITERATIONS_ENV).ok(),
            ..RuntimeEnv::from_process_env()
        },
    )?;
    let text = load_input_text(&config.input_path)?;
    let hyperparameters = config.hyperparameters;
    let logger = EventLogger::stdout(config.log_format);
    logger.log(RuntimeEvent::AppConfigured {
        backend: config.backend.label().to_string(),
        model: config.model.label().to_string(),
        input_path: config.input_path.display().to_string(),
        tokenizer_path: minigpt_tokenizer_path(),
        checkpoint_path: config.checkpoint_path.display().to_string(),
        log_format: config.log_format,
        serve: config.serve,
        benchmark_generation: config.benchmark_generation,
    });

    if config.serve {
        return match config.backend {
            BackendChoice::Cpu => run_http_server_with_runtime::<NdArray<f32, i64>>(
                &text,
                hyperparameters,
                ServerRuntimeOptions {
                    server_addr: config.server_addr,
                    checkpoint_path: &config.checkpoint_path,
                    load_checkpoint_enabled: config.load_checkpoint,
                    load_latest_checkpoint_enabled: config.load_latest_checkpoint,
                    backend_label: "cpu",
                    logger,
                },
                &NdArrayDevice::Cpu,
            ),
            #[cfg(feature = "cuda")]
            BackendChoice::Cuda => run_http_server_with_runtime::<Cuda>(
                &text,
                hyperparameters,
                ServerRuntimeOptions {
                    server_addr: config.server_addr,
                    checkpoint_path: &config.checkpoint_path,
                    load_checkpoint_enabled: config.load_checkpoint,
                    load_latest_checkpoint_enabled: config.load_latest_checkpoint,
                    backend_label: "cuda",
                    logger,
                },
                &CudaDevice::default(),
            ),
        };
    }

    match config.backend {
        BackendChoice::Cpu => run_cpu_demo(
            &text,
            hyperparameters,
            CpuDemoOptions {
                model_choice: config.model,
                interactive: config.interactive,
                benchmark_generation: config.benchmark_generation,
                benchmark_config: &config.benchmark_config,
                logger,
                checkpoint_path: &config.checkpoint_path,
                input_source: &config.input_path.display().to_string(),
            },
        ),
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => {
            if config.interactive {
                bail!("interactive generation currently requires --backend cpu");
            }
            let device = CudaDevice::default();
            run_demo::<Cuda>(&text, hyperparameters, config.model, &device, &logger)?;
            run_training_demo::<Autodiff<Cuda>>(
                &text,
                hyperparameters,
                config.model,
                &device,
                &config.checkpoint_path,
                TrainingDemoOptions {
                    backend_label: "cuda",
                    logger,
                    benchmark_generation: config.benchmark_generation,
                    benchmark_config: config.benchmark_config,
                    input_source: config.input_path.display().to_string(),
                },
            )
        }
    }
}

struct CpuDemoOptions<'a> {
    model_choice: ModelChoice,
    interactive: bool,
    benchmark_generation: bool,
    benchmark_config: &'a BenchmarkConfig,
    logger: EventLogger,
    checkpoint_path: &'a Path,
    input_source: &'a str,
}

fn run_cpu_demo(
    text: &str,
    hyperparameters: Hyperparameters,
    options: CpuDemoOptions<'_>,
) -> Result<()> {
    let device = NdArrayDevice::Cpu;
    run_demo::<NdArray<f32, i64>>(
        text,
        hyperparameters,
        options.model_choice,
        &device,
        &options.logger,
    )?;
    if options.interactive {
        if options.benchmark_generation {
            bail!("generation benchmarks cannot run with --interactive-generate");
        }
        if options.model_choice != ModelChoice::MiniGpt {
            bail!("interactive generation requires --model minigpt");
        }
        run_interactive_minigpt_generation::<Autodiff<NdArray<f32, i64>>>(
            text,
            hyperparameters,
            &device,
            options.logger,
            options.checkpoint_path,
        )
    } else {
        run_training_demo::<Autodiff<NdArray<f32, i64>>>(
            text,
            hyperparameters,
            options.model_choice,
            &device,
            options.checkpoint_path,
            TrainingDemoOptions {
                backend_label: "cpu",
                logger: options.logger,
                benchmark_generation: options.benchmark_generation,
                benchmark_config: options.benchmark_config.clone(),
                input_source: options.input_source.to_string(),
            },
        )
    }
}

struct ServerRuntimeOptions<'a> {
    server_addr: SocketAddr,
    checkpoint_path: &'a Path,
    load_checkpoint_enabled: bool,
    load_latest_checkpoint_enabled: bool,
    backend_label: &'static str,
    logger: EventLogger,
}

fn run_http_server_with_runtime<B>(
    text: &str,
    hyperparameters: Hyperparameters,
    options: ServerRuntimeOptions<'_>,
    device: &B::Device,
) -> Result<()>
where
    B: Backend + Send + Sync + 'static,
    B::Device: Send + Sync + 'static,
{
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to start tokio runtime")?;

    runtime.block_on(run_http_server::<B>(text, hyperparameters, options, device))
}

async fn run_http_server<B>(
    _text: &str,
    hyperparameters: Hyperparameters,
    options: ServerRuntimeOptions<'_>,
    device: &B::Device,
) -> Result<()>
where
    B: Backend + Send + Sync + 'static,
    B::Device: Send + Sync + 'static,
{
    let tokenizer = load_minigpt_tokenizer()?;
    let template = new_minigpt::<B>(tokenizer.vocab_size(), hyperparameters, device);
    let model = if options.load_latest_checkpoint_enabled {
        let latest_checkpoint = latest_checkpoint_path(Path::new(DEFAULT_CHECKPOINT_DIR))?;
        load_minigpt_checkpoint(template, &latest_checkpoint, device, &options.logger)?
    } else if options.load_checkpoint_enabled {
        load_minigpt_checkpoint(template, options.checkpoint_path, device, &options.logger)?
    } else {
        template
    };
    let state = Arc::new(ServerState::new(
        model,
        tokenizer,
        device.clone(),
        options.logger.clone(),
    ));
    let vocab_size = state.model_vocab_size();
    let block_size = state.model_block_size();
    let app = Router::new()
        .nest("/api", server::router::<B>())
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind(options.server_addr)
        .await
        .with_context(|| format!("failed to bind HTTP server on {}", options.server_addr))?;

    options.logger.log(RuntimeEvent::ServerStarted {
        addr: options.server_addr.to_string(),
        backend: options.backend_label.to_string(),
        vocab_size,
        block_size,
    });
    axum::serve(listener, app)
        .await
        .context("HTTP server failed")
}

fn run_demo<B: Backend>(
    text: &str,
    hyperparameters: Hyperparameters,
    model_choice: ModelChoice,
    device: &B::Device,
    logger: &EventLogger,
) -> Result<()> {
    let tokenizer = tokenizer_for_model(text, model_choice)?;
    let encoded = tokenizer.encode(text);

    let data_loader = DataLoader {
        tokens: encoded,
        block_size: hyperparameters.block_size,
        batch_size: hyperparameters.batch_size,
    };
    let (x, y) = data_loader
        .next_batch::<B>(device)
        .map_err(anyhow::Error::msg)
        .context("failed to build demo batch")?;
    logger.log(RuntimeEvent::RuntimeBatchPrepared {
        vocab_size: tokenizer.vocab_size(),
        input_chars: text.chars().count(),
        encoded_tokens: data_loader.tokens.len(),
        batch_size: hyperparameters.batch_size,
        block_size: hyperparameters.block_size,
        dropout: hyperparameters.dropout,
    });
    for model_choice in model_choice.comparison_models() {
        let logits = run_model_forward(
            model_choice,
            tokenizer.vocab_size(),
            hyperparameters,
            x.clone(),
            device,
        );
        logger.log(RuntimeEvent::ModelForwardCompleted {
            model: model_choice.label().to_string(),
            logits_shape: logits.shape().dims::<3>(),
            input_shape: x.shape().dims::<2>(),
            target_shape: y.shape().dims::<2>(),
        });
    }

    Ok(())
}

fn run_interactive_minigpt_generation<B: burn::tensor::backend::AutodiffBackend>(
    _text: &str,
    hyperparameters: Hyperparameters,
    device: &B::Device,
    logger: EventLogger,
    checkpoint_path: &Path,
) -> Result<()> {
    let tokenizer = load_minigpt_tokenizer()?;
    let template = new_minigpt::<B>(tokenizer.vocab_size(), hyperparameters, device);
    let model = load_minigpt_checkpoint(template, checkpoint_path, device, &logger)?;

    interactive_generation_loop(&model, &tokenizer, hyperparameters.generate_tokens, device)
}

fn load_minigpt_checkpoint<B: Backend>(
    template: MiniGpt<B>,
    checkpoint_path: &Path,
    device: &B::Device,
    logger: &EventLogger,
) -> Result<MiniGpt<B>> {
    let started_at = Instant::now();
    let expected_shape = CheckpointModelShape {
        vocab_size: template.vocab_size(),
        block_size: template.block_size(),
        embed_dim: template.d_model(),
        num_heads: template.num_heads(),
        num_layers: template.num_layers(),
    };
    let model =
        load_model_with_metadata_validation(template, checkpoint_path, &expected_shape, device)
            .with_context(|| {
                format!(
                    "failed to load minigpt checkpoint from {:?}",
                    checkpoint_path.with_extension("mpk")
                )
            })?;
    logger.log(RuntimeEvent::CheckpointLoaded {
        path: checkpoint_path.with_extension("mpk").display().to_string(),
        elapsed_ms: started_at.elapsed().as_millis(),
    });

    Ok(model)
}

fn tokenizer_for_model(text: &str, model_choice: ModelChoice) -> Result<RuntimeTokenizer> {
    if model_choice.includes_minigpt() {
        load_minigpt_tokenizer()
    } else {
        Ok(RuntimeTokenizer::char_from_text(text))
    }
}

fn load_minigpt_tokenizer() -> Result<RuntimeTokenizer> {
    let tokenizer_path = minigpt_tokenizer_path();
    RuntimeTokenizer::load_bpe(Path::new(&tokenizer_path)).with_context(|| {
        format!(
            "failed to load MiniGPT BPE tokenizer from {tokenizer_path}; train one with `cargo run --bin train-tokenizer -- --corpus data/fafolang.txt --vocab-size 2048 --output {DEFAULT_BPE_TOKENIZER_PATH}`"
        )
    })
}

fn minigpt_tokenizer_path() -> String {
    env::var(BPE_TOKENIZER_ENV).unwrap_or_else(|_| DEFAULT_BPE_TOKENIZER_PATH.to_string())
}

fn latest_checkpoint_path(checkpoint_dir: &Path) -> Result<PathBuf> {
    let mut latest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(checkpoint_dir)
        .with_context(|| format!("failed to read checkpoint directory {:?}", checkpoint_dir))?
    {
        let entry = entry.with_context(|| {
            format!(
                "failed to read checkpoint directory entry in {:?}",
                checkpoint_dir
            )
        })?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("mpk") {
            continue;
        }

        let modified = entry
            .metadata()
            .with_context(|| format!("failed to read checkpoint metadata for {:?}", path))?
            .modified()
            .with_context(|| format!("failed to read checkpoint modified time for {:?}", path))?;
        if latest
            .as_ref()
            .map(|(latest_modified, _)| modified > *latest_modified)
            .unwrap_or(true)
        {
            latest = Some((modified, path.with_extension("")));
        }
    }

    latest
        .map(|(_, path)| path)
        .with_context(|| format!("no .mpk checkpoints found in {:?}", checkpoint_dir))
}

fn interactive_generation_loop<B: Backend>(
    model: &MiniGpt<B>,
    tokenizer: &RuntimeTokenizer,
    generate_tokens: usize,
    device: &B::Device,
) -> Result<()> {
    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let mut stdout = io::stdout();
    let mut line = String::new();

    println!("Enter a prompt, or an empty line / :quit to exit.");
    loop {
        print!("> ");
        stdout.flush().context("failed to flush prompt")?;
        line.clear();
        if stdin
            .read_line(&mut line)
            .context("failed to read prompt")?
            == 0
        {
            break;
        }

        let prompt = line.trim_end_matches(['\r', '\n']);
        if prompt.is_empty() || prompt == ":quit" {
            break;
        }

        let encoded = match tokenizer.try_encode(prompt) {
            Ok(encoded) => encoded,
            Err(err) => {
                println!("{err}");
                continue;
            }
        };
        print_attention_sanity(model, &encoded, device)?;
        let generated = model
            .generate(&encoded, generate_tokens, device)
            .map_err(anyhow::Error::msg)
            .context("failed to generate text")?;

        println!("{}", tokenizer.decode(&generated));
    }

    Ok(())
}

fn print_attention_sanity<B: Backend>(
    model: &MiniGpt<B>,
    prompt: &[usize],
    device: &B::Device,
) -> Result<()> {
    let _generated = model
        .generate(prompt, 1, device)
        .map_err(anyhow::Error::msg)
        .context("failed to generate attention sanity token")?;
    let input: Vec<i64> = prompt.iter().map(|&token| token as i64).collect();
    let tokens = burn::tensor::Tensor::from_data(
        burn::tensor::TensorData::new(input, [1, prompt.len()]),
        device,
    );
    let (_logits, attentions) = model.forward_tokens_with_attention(tokens);

    if let Some(layer_0_attention) = attentions.first() {
        println!(
            "layer 0 attention shape: {:?}",
            layer_0_attention.shape().dims::<4>()
        );
    } else {
        println!("layer 0 attention shape: no transformer layers");
    }

    Ok(())
}

fn run_model_forward<B: Backend>(
    model_choice: ModelChoice,
    vocab_size: usize,
    hyperparameters: Hyperparameters,
    input: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    device: &B::Device,
) -> burn::tensor::Tensor<B, 3> {
    match model_choice {
        ModelChoice::Trivial => {
            let model = TrivialModel::<B>::new(vocab_size, hyperparameters.embed_dim, device);
            model.forward(input)
        }
        ModelChoice::SingleAttention => {
            let model = SingleAttentionModel::<B>::new(
                vocab_size,
                hyperparameters.embed_dim,
                hyperparameters.head_dim,
                device,
            );
            model.forward_tokens(input)
        }
        ModelChoice::MultiAttention => {
            let model = MultiAttentionModel::<B>::new(
                vocab_size,
                hyperparameters.embed_dim,
                hyperparameters.num_heads,
                device,
            );
            model.forward_tokens(input)
        }
        ModelChoice::MiniGpt => {
            let model = new_minigpt::<B>(vocab_size, hyperparameters, device);
            model.forward_tokens(input)
        }
        ModelChoice::Compare => unreachable!("compare should be expanded before forward dispatch"),
    }
}

fn new_minigpt<B: Backend>(
    vocab_size: usize,
    hyperparameters: Hyperparameters,
    device: &B::Device,
) -> MiniGpt<B> {
    MiniGpt::<B>::new(
        vocab_size,
        hyperparameters.embed_dim,
        hyperparameters.num_layers,
        hyperparameters.block_size,
        hyperparameters.num_heads,
        device,
    )
}

fn run_training_demo<B>(
    text: &str,
    hyperparameters: Hyperparameters,
    model_choice: ModelChoice,
    device: &B::Device,
    checkpoint_path: &Path,
    options: TrainingDemoOptions,
) -> Result<()>
where
    B: burn::tensor::backend::AutodiffBackend,
    B::FloatElem: Into<f64>,
{
    if options.benchmark_generation && !model_choice.includes_minigpt() {
        bail!("generation benchmarks require --model minigpt or compare");
    }

    let tokenizer = tokenizer_for_model(text, model_choice)?;
    let encoded = tokenizer.encode(text);
    let (training_tokens, value_tokens, value_block_size) =
        split_training_and_value_tokens(&encoded, hyperparameters.block_size)?;
    let data_loader = DataLoader {
        tokens: training_tokens,
        block_size: hyperparameters.block_size,
        batch_size: hyperparameters.batch_size,
    };
    let value_loader = DataLoader {
        tokens: value_tokens,
        block_size: value_block_size,
        batch_size: hyperparameters.batch_size,
    };

    for model_choice in model_choice.comparison_models() {
        train_model(TrainingRun::<B> {
            model_choice,
            data_loader: &data_loader,
            value_loader: &value_loader,
            device,
            vocab_size: tokenizer.vocab_size(),
            hyperparameters,
            checkpoint_path,
            backend_label: options.backend_label,
            logger: options.logger.clone(),
            benchmark_generation: options.benchmark_generation,
            benchmark_config: options.benchmark_config.clone(),
            input_source: &options.input_source,
        })?;
    }

    Ok(())
}

#[derive(Clone)]
struct TrainingDemoOptions {
    backend_label: &'static str,
    logger: EventLogger,
    benchmark_generation: bool,
    benchmark_config: BenchmarkConfig,
    input_source: String,
}

struct TrainingRun<'a, B: burn::tensor::backend::AutodiffBackend> {
    model_choice: ModelChoice,
    data_loader: &'a DataLoader,
    value_loader: &'a DataLoader,
    device: &'a B::Device,
    vocab_size: usize,
    hyperparameters: Hyperparameters,
    checkpoint_path: &'a Path,
    backend_label: &'static str,
    logger: EventLogger,
    benchmark_generation: bool,
    benchmark_config: BenchmarkConfig,
    input_source: &'a str,
}

fn train_model<B>(run: TrainingRun<'_, B>) -> Result<()>
where
    B: burn::tensor::backend::AutodiffBackend,
    B::FloatElem: Into<f64>,
{
    let log_context = TrainingLogContext {
        backend: run.backend_label,
        model: run.model_choice.label(),
        logger: run.logger.clone(),
    };
    let params = TrainingParams::new(
        run.hyperparameters.learning_rate,
        run.hyperparameters.train_steps,
        run.hyperparameters.eval_interval,
        log_context,
    )
    .with_prefetch_batches(run.hyperparameters.prefetch_batches);

    match run.model_choice {
        ModelChoice::Trivial => {
            run.logger.log(RuntimeEvent::TrainingStarted {
                backend: run.backend_label.to_string(),
                model: run.model_choice.label().to_string(),
                vocab_size: run.vocab_size,
                batch_size: run.hyperparameters.batch_size,
                block_size: run.hyperparameters.block_size,
                total_steps: run.hyperparameters.train_steps,
            });
            let started_at = Instant::now();
            let outcome = TrivialModel::<B>::train(
                run.data_loader,
                run.value_loader,
                run.device,
                run.vocab_size,
                run.hyperparameters.embed_dim,
                params,
            )
            .map_err(anyhow::Error::msg)
            .context("failed to train trivial model")?;
            log_training_completed(&run, started_at, outcome.metrics);
        }
        ModelChoice::SingleAttention => {
            run.logger.log(RuntimeEvent::TrainingStarted {
                backend: run.backend_label.to_string(),
                model: run.model_choice.label().to_string(),
                vocab_size: run.vocab_size,
                batch_size: run.hyperparameters.batch_size,
                block_size: run.hyperparameters.block_size,
                total_steps: run.hyperparameters.train_steps,
            });
            let started_at = Instant::now();
            let outcome = SingleAttentionModel::<B>::train(
                run.data_loader,
                run.value_loader,
                run.device,
                run.vocab_size,
                run.hyperparameters.embed_dim,
                run.hyperparameters.head_dim,
                params,
            )
            .map_err(anyhow::Error::msg)
            .context("failed to train single attention model")?;
            log_training_completed(&run, started_at, outcome.metrics);
        }
        ModelChoice::MultiAttention => {
            run.logger.log(RuntimeEvent::TrainingStarted {
                backend: run.backend_label.to_string(),
                model: run.model_choice.label().to_string(),
                vocab_size: run.vocab_size,
                batch_size: run.hyperparameters.batch_size,
                block_size: run.hyperparameters.block_size,
                total_steps: run.hyperparameters.train_steps,
            });
            let started_at = Instant::now();
            let outcome = MultiAttentionModel::<B>::train(
                run.data_loader,
                run.value_loader,
                run.device,
                run.vocab_size,
                run.hyperparameters.embed_dim,
                run.hyperparameters.num_heads,
                params,
            )
            .map_err(anyhow::Error::msg)
            .context("failed to train multi attention model")?;
            log_training_completed(&run, started_at, outcome.metrics);
        }
        ModelChoice::MiniGpt => {
            run.logger.log(RuntimeEvent::TrainingStarted {
                backend: run.backend_label.to_string(),
                model: run.model_choice.label().to_string(),
                vocab_size: run.vocab_size,
                batch_size: run.hyperparameters.batch_size,
                block_size: run.hyperparameters.block_size,
                total_steps: run.hyperparameters.train_steps,
            });
            let started_at = Instant::now();
            let outcome = MiniGpt::<B>::train(
                run.data_loader,
                run.value_loader,
                run.device,
                MiniGptConfig {
                    vocab_size: run.vocab_size,
                    d_model: run.hyperparameters.embed_dim,
                    num_blocks: run.hyperparameters.num_layers,
                    max_position_embeddings: run.hyperparameters.block_size,
                    num_heads: run.hyperparameters.num_heads,
                },
                params.with_grad_clip_norm(run.hyperparameters.minigpt_grad_clip_norm),
            )
            .map_err(anyhow::Error::msg)
            .context("failed to train minigpt model")?;
            log_training_completed(&run, started_at, outcome.metrics);
            if run.benchmark_generation {
                benchmark_generation(
                    &outcome.model,
                    run.device,
                    &run.benchmark_config,
                    &run.logger,
                )
                .map_err(anyhow::Error::msg)
                .context("failed to benchmark minigpt generation")?;
            }
            save_minigpt_checkpoint(outcome, run.checkpoint_path, &run)?;
        }
        ModelChoice::Compare => unreachable!("compare should be expanded before training dispatch"),
    }

    Ok(())
}

fn log_training_completed<B>(
    run: &TrainingRun<'_, B>,
    started_at: Instant,
    metrics: rusty_gpt::model::TrainingMetrics,
) where
    B: burn::tensor::backend::AutodiffBackend,
{
    run.logger.log(RuntimeEvent::TrainingCompleted {
        backend: run.backend_label.to_string(),
        model: run.model_choice.label().to_string(),
        total_steps: run.hyperparameters.train_steps,
        elapsed_ms: started_at.elapsed().as_millis(),
        final_value_loss: metrics.final_value_loss,
        final_perplexity: metrics.final_perplexity,
    });
}

fn save_minigpt_checkpoint<B: burn::tensor::backend::AutodiffBackend>(
    outcome: TrainingOutcome<MiniGpt<B>>,
    checkpoint_path: &Path,
    run: &TrainingRun<'_, B>,
) -> Result<()> {
    if let Some(parent) = checkpoint_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create checkpoint directory {:?}", parent))?;
    }

    let started_at = Instant::now();
    save_model(outcome.model, checkpoint_path)
        .with_context(|| format!("failed to save minigpt checkpoint to {:?}", checkpoint_path))?;
    save_checkpoint_metadata(checkpoint_path, &checkpoint_metadata(run, outcome.metrics)?)
        .with_context(|| {
            format!(
                "failed to save checkpoint metadata for {:?}",
                checkpoint_path
            )
        })?;
    run.logger.log(RuntimeEvent::CheckpointSaved {
        path: checkpoint_path.with_extension("mpk").display().to_string(),
        elapsed_ms: started_at.elapsed().as_millis(),
    });

    Ok(())
}

fn checkpoint_metadata<B: burn::tensor::backend::AutodiffBackend>(
    run: &TrainingRun<'_, B>,
    metrics: rusty_gpt::model::TrainingMetrics,
) -> Result<CheckpointMetadata> {
    let tokenizer_path = PathBuf::from(minigpt_tokenizer_path());
    Ok(CheckpointMetadata {
        version: 1,
        created_at_utc: chrono::Utc::now().to_rfc3339(),
        git_commit: current_git_commit(),
        input_source: run.input_source.to_string(),
        model_shape: CheckpointModelShape {
            vocab_size: run.vocab_size,
            block_size: run.hyperparameters.block_size,
            embed_dim: run.hyperparameters.embed_dim,
            num_heads: run.hyperparameters.num_heads,
            num_layers: run.hyperparameters.num_layers,
        },
        tokenizer: CheckpointTokenizer {
            path: tokenizer_path.display().to_string(),
            sha256: sha256_file_hex(&tokenizer_path)?,
            vocab_size: run.vocab_size,
        },
        training: CheckpointTrainingRun {
            backend: run.backend_label.to_string(),
            train_tokens: run.data_loader.tokens.len(),
            value_tokens: run.value_loader.tokens.len(),
            batch_size: run.hyperparameters.batch_size,
            learning_rate: run.hyperparameters.learning_rate,
            train_steps: run.hyperparameters.train_steps,
            eval_interval: run.hyperparameters.eval_interval,
            grad_clip_norm: run.hyperparameters.minigpt_grad_clip_norm,
            prefetch_batches: run.hyperparameters.prefetch_batches,
        },
        final_metrics: CheckpointTrainingMetrics {
            final_value_loss: metrics.final_value_loss,
            final_perplexity: metrics.final_perplexity,
        },
    })
}

fn current_git_commit() -> Option<String> {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|commit| commit.trim().to_string())
        .filter(|commit| !commit.is_empty())
}

fn split_training_and_value_tokens(
    tokens: &[usize],
    block_size: usize,
) -> Result<(Vec<usize>, Vec<usize>, usize)> {
    let value_len = tokens.len() / 10;
    if value_len < 2 {
        bail!(
            "not enough input tokens for value loss: last 10% has {value_len} tokens, need at least 2"
        );
    }

    let split_at = tokens.len() - value_len;
    let min_tokens = block_size + 1;
    if split_at < min_tokens {
        bail!(
            "not enough input tokens for training loss: first 90% has {split_at} tokens, need at least {min_tokens}"
        );
    }

    let value_block_size = block_size.min(value_len - 1);
    Ok((
        tokens[..split_at].to_vec(),
        tokens[split_at..].to_vec(),
        value_block_size,
    ))
}

#[derive(Debug, Clone, PartialEq)]
struct RuntimeConfig {
    backend: BackendChoice,
    model: ModelChoice,
    input_path: PathBuf,
    checkpoint_path: PathBuf,
    hyperparameters: Hyperparameters,
    interactive: bool,
    benchmark_generation: bool,
    load_checkpoint: bool,
    load_latest_checkpoint: bool,
    serve: bool,
    server_addr: SocketAddr,
    log_format: LogFormat,
    benchmark_config: BenchmarkConfig,
}

#[derive(Debug, Clone, Default)]
struct RuntimeEnv {
    backend: Option<String>,
    input: Option<String>,
    model: Option<String>,
    checkpoint: Option<String>,
    server_addr: Option<String>,
    log_format: Option<String>,
    benchmark_prompt_lens: Option<String>,
    benchmark_gen_lens: Option<String>,
    benchmark_warmups: Option<String>,
    benchmark_iterations: Option<String>,
    block_size: Option<String>,
    batch_size: Option<String>,
    embed_dim: Option<String>,
    num_heads: Option<String>,
    num_layers: Option<String>,
    dropout: Option<String>,
    learning_rate: Option<String>,
    train_steps: Option<String>,
    eval_interval: Option<String>,
    generate_tokens: Option<String>,
    minigpt_grad_clip_norm: Option<String>,
    prefetch_batches: Option<String>,
}

impl RuntimeEnv {
    fn from_process_env() -> Self {
        Self {
            backend: env::var("RUSTY_GPT_BACKEND").ok(),
            input: env::var("RUSTY_GPT_INPUT").ok(),
            model: env::var("RUSTY_GPT_MODEL").ok(),
            checkpoint: env::var("RUSTY_GPT_MINIGPT_CHECKPOINT").ok(),
            server_addr: env::var("RUSTY_GPT_SERVER_ADDR").ok(),
            log_format: env::var(LOG_FORMAT_ENV).ok(),
            benchmark_prompt_lens: env::var(BENCHMARK_PROMPT_LENS_ENV).ok(),
            benchmark_gen_lens: env::var(BENCHMARK_GEN_LENS_ENV).ok(),
            benchmark_warmups: env::var(BENCHMARK_WARMUPS_ENV).ok(),
            benchmark_iterations: env::var(BENCHMARK_ITERATIONS_ENV).ok(),
            block_size: env::var("RUSTY_GPT_BLOCK_SIZE").ok(),
            batch_size: env::var("RUSTY_GPT_BATCH_SIZE").ok(),
            embed_dim: env::var("RUSTY_GPT_EMBED_DIM").ok(),
            num_heads: env::var("RUSTY_GPT_NUM_HEADS").ok(),
            num_layers: env::var("RUSTY_GPT_NUM_LAYERS").ok(),
            dropout: env::var("RUSTY_GPT_DROPOUT").ok(),
            learning_rate: env::var("RUSTY_GPT_LEARNING_RATE").ok(),
            train_steps: env::var("RUSTY_GPT_TRAIN_STEPS").ok(),
            eval_interval: env::var("RUSTY_GPT_EVAL_INTERVAL").ok(),
            generate_tokens: env::var("RUSTY_GPT_GENERATE_TOKENS").ok(),
            minigpt_grad_clip_norm: env::var("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM").ok(),
            prefetch_batches: env::var("RUSTY_GPT_PREFETCH_BATCHES").ok(),
        }
    }
}

fn load_input_text(path: &Path) -> Result<String> {
    let input = path.as_os_str().to_string_lossy();
    if let Some(text) = huggingface::load_text_from_uri(&input)? {
        return Ok(text);
    }

    fs::read_to_string(path).with_context(|| format!("failed to read input text from {:?}", path))
}

#[cfg(test)]
fn parse_runtime_config<I, S>(
    args: I,
    env_backend: Option<&str>,
    env_input: Option<&str>,
    env_model: Option<&str>,
) -> Result<RuntimeConfig>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    parse_runtime_config_with_checkpoint(
        args,
        RuntimeEnv {
            backend: env_backend.map(str::to_string),
            input: env_input.map(str::to_string),
            model: env_model.map(str::to_string),
            ..RuntimeEnv::default()
        },
    )
}

fn parse_runtime_config_with_checkpoint<I, S>(args: I, env: RuntimeEnv) -> Result<RuntimeConfig>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let args: Vec<String> = args
        .into_iter()
        .map(|arg| arg.as_ref().to_string())
        .collect();
    let mut arg_backend = None;
    let mut arg_input = None;
    let mut arg_model = None;
    let mut arg_checkpoint = None;
    let mut arg_server_addr = None;
    let mut arg_log_format = None;
    let mut arg_benchmark_prompt_lens = None;
    let mut arg_benchmark_gen_lens = None;
    let mut arg_benchmark_warmups = None;
    let mut arg_benchmark_iterations = None;
    let mut hyperparameter_overrides = HyperparameterOverrides::default();
    let mut interactive = false;
    let mut benchmark_generation = false;
    let mut load_checkpoint = false;
    let mut load_latest_checkpoint = false;
    let mut serve = false;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--backend" => {
                let value = args
                    .get(index + 1)
                    .context("--backend requires a value: cpu or cuda")?;
                arg_backend = Some(value.as_str());
                index += 2;
            }
            "--input" => {
                let value = args
                    .get(index + 1)
                    .context("--input requires a path to a text file")?;
                arg_input = Some(value.as_str());
                index += 2;
            }
            "--model" => {
                let value = args
                    .get(index + 1)
                    .context("--model requires a value: trivial, single-attention, multi-attention, minigpt, or compare")?;
                arg_model = Some(value.as_str());
                index += 2;
            }
            "--checkpoint" => {
                let value = args
                    .get(index + 1)
                    .context("--checkpoint requires a value: path to a saved .mpk checkpoint without the extension")?;
                arg_checkpoint = Some(value.as_str());
                index += 2;
            }
            "--server-addr" => {
                let value = args
                    .get(index + 1)
                    .context("--server-addr requires a value like 127.0.0.1:8787")?;
                arg_server_addr = Some(value.as_str());
                index += 2;
            }
            "--log-format" => {
                let value = args
                    .get(index + 1)
                    .context("--log-format requires a value: plain or json")?;
                arg_log_format = Some(value.as_str());
                index += 2;
            }
            "--benchmark-prompt-lens" => {
                let value = args
                    .get(index + 1)
                    .context("--benchmark-prompt-lens requires a comma-separated list")?;
                arg_benchmark_prompt_lens = Some(value.as_str());
                index += 2;
            }
            "--benchmark-gen-lens" => {
                let value = args
                    .get(index + 1)
                    .context("--benchmark-gen-lens requires a comma-separated list")?;
                arg_benchmark_gen_lens = Some(value.as_str());
                index += 2;
            }
            "--benchmark-warmups" => {
                let value = args
                    .get(index + 1)
                    .context("--benchmark-warmups requires an integer")?;
                arg_benchmark_warmups = Some(value.as_str());
                index += 2;
            }
            "--benchmark-iterations" => {
                let value = args
                    .get(index + 1)
                    .context("--benchmark-iterations requires an integer")?;
                arg_benchmark_iterations = Some(value.as_str());
                index += 2;
            }
            "--block-size" => {
                hyperparameter_overrides.block_size =
                    Some(parse_arg_value(&args, index, "--block-size")?);
                index += 2;
            }
            "--batch-size" => {
                hyperparameter_overrides.batch_size =
                    Some(parse_arg_value(&args, index, "--batch-size")?);
                index += 2;
            }
            "--embed-dim" => {
                hyperparameter_overrides.embed_dim =
                    Some(parse_arg_value(&args, index, "--embed-dim")?);
                index += 2;
            }
            "--num-heads" => {
                hyperparameter_overrides.num_heads =
                    Some(parse_arg_value(&args, index, "--num-heads")?);
                index += 2;
            }
            "--num-layers" => {
                hyperparameter_overrides.num_layers =
                    Some(parse_arg_value(&args, index, "--num-layers")?);
                index += 2;
            }
            "--dropout" => {
                hyperparameter_overrides.dropout =
                    Some(parse_arg_value(&args, index, "--dropout")?);
                index += 2;
            }
            "--learning-rate" => {
                hyperparameter_overrides.learning_rate =
                    Some(parse_arg_value(&args, index, "--learning-rate")?);
                index += 2;
            }
            "--train-steps" => {
                hyperparameter_overrides.train_steps =
                    Some(parse_arg_value(&args, index, "--train-steps")?);
                index += 2;
            }
            "--eval-interval" => {
                hyperparameter_overrides.eval_interval =
                    Some(parse_arg_value(&args, index, "--eval-interval")?);
                index += 2;
            }
            "--generate-tokens" => {
                hyperparameter_overrides.generate_tokens =
                    Some(parse_arg_value(&args, index, "--generate-tokens")?);
                index += 2;
            }
            "--grad-clip-norm" => {
                hyperparameter_overrides.minigpt_grad_clip_norm =
                    Some(parse_arg_value(&args, index, "--grad-clip-norm")?);
                index += 2;
            }
            "--prefetch-batches" => {
                hyperparameter_overrides.prefetch_batches =
                    Some(parse_arg_value(&args, index, "--prefetch-batches")?);
                index += 2;
            }
            "--interactive-generate" => {
                interactive = true;
                index += 1;
            }
            "--benchmark-generation" => {
                benchmark_generation = true;
                index += 1;
            }
            "--load-checkpoint" => {
                load_checkpoint = true;
                index += 1;
            }
            "--load-latest-checkpoint" => {
                load_latest_checkpoint = true;
                index += 1;
            }
            "--serve" => {
                serve = true;
                index += 1;
            }
            other => bail!("unsupported argument: {other}"),
        }
    }

    if load_checkpoint && load_latest_checkpoint {
        bail!("--load-checkpoint and --load-latest-checkpoint are mutually exclusive");
    }

    let server_addr_text = arg_server_addr
        .or(env.server_addr.as_deref())
        .unwrap_or(DEFAULT_SERVER_ADDR);
    let server_addr = server_addr_text
        .parse()
        .with_context(|| format!("invalid server address '{server_addr_text}'"))?;
    let backend = parse_backend_name(arg_backend.or(env.backend.as_deref()).unwrap_or("cpu"))?;
    let log_format = match arg_log_format.or(env.log_format.as_deref()) {
        Some(value) => LogFormat::parse(value)?,
        None => default_log_format(backend),
    };
    let mut benchmark_config = BenchmarkConfig::default();
    if let Some(value) = arg_benchmark_prompt_lens.or(env.benchmark_prompt_lens.as_deref()) {
        benchmark_config.prompt_lens = parse_usize_list(value, "benchmark prompt lengths")
            .with_context(|| format!("invalid benchmark prompt lengths '{value}'"))?;
    }
    if let Some(value) = arg_benchmark_gen_lens.or(env.benchmark_gen_lens.as_deref()) {
        benchmark_config.gen_lens = parse_usize_list(value, "benchmark generation lengths")
            .with_context(|| format!("invalid benchmark generation lengths '{value}'"))?;
    }
    if let Some(value) = arg_benchmark_warmups.or(env.benchmark_warmups.as_deref()) {
        benchmark_config.warmups = value
            .parse()
            .with_context(|| format!("invalid benchmark warmups '{value}'"))?;
    }
    if let Some(value) = arg_benchmark_iterations.or(env.benchmark_iterations.as_deref()) {
        benchmark_config.iterations = value
            .parse()
            .with_context(|| format!("invalid benchmark iterations '{value}'"))?;
    }
    benchmark_config.validate()?;
    let hyperparameters = Hyperparameters::from_env_and_overrides(&env, &hyperparameter_overrides)?;

    Ok(RuntimeConfig {
        backend,
        model: parse_model_name(arg_model.or(env.model.as_deref()).unwrap_or("minigpt"))?,
        input_path: PathBuf::from(
            arg_input
                .or(env.input.as_deref())
                .unwrap_or(DEFAULT_INPUT_PATH),
        ),
        checkpoint_path: PathBuf::from(
            arg_checkpoint
                .or(env.checkpoint.as_deref())
                .unwrap_or(DEFAULT_MINIGPT_CHECKPOINT_PATH),
        ),
        hyperparameters,
        interactive,
        benchmark_generation,
        load_checkpoint,
        load_latest_checkpoint,
        serve,
        server_addr,
        log_format,
        benchmark_config,
    })
}

fn default_log_format(backend: BackendChoice) -> LogFormat {
    match backend {
        BackendChoice::Cpu => LogFormat::Plain,
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => LogFormat::Json,
    }
}

fn parse_arg_value<T>(args: &[String], index: usize, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let value = args
        .get(index + 1)
        .with_context(|| format!("{name} requires a value"))?;
    value
        .parse()
        .with_context(|| format!("invalid {name} value: {value}"))
}

fn parse_backend_name(name: &str) -> Result<BackendChoice> {
    match name {
        "cpu" => Ok(BackendChoice::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(BackendChoice::Cuda),
        #[cfg(not(feature = "cuda"))]
        "cuda" => bail!("cuda backend is not compiled in; rebuild with `--features cuda`"),
        other => bail!("unsupported backend '{other}'; expected cpu or cuda"),
    }
}

fn parse_model_name(name: &str) -> Result<ModelChoice> {
    match name {
        "trivial" => Ok(ModelChoice::Trivial),
        "single-attention" => Ok(ModelChoice::SingleAttention),
        "multi-attention" => Ok(ModelChoice::MultiAttention),
        "minigpt" | "mini-gpt" => Ok(ModelChoice::MiniGpt),
        "compare" => Ok(ModelChoice::Compare),
        other => bail!(
            "unsupported model '{other}'; expected trivial, single-attention, multi-attention, minigpt, or compare"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_defaults_to_cpu() {
        let config = parse_runtime_config(Vec::<String>::new(), None, None, None).unwrap();

        assert_eq!(BackendChoice::Cpu, config.backend);
        assert_eq!(ModelChoice::MiniGpt, config.model);
        assert_eq!(PathBuf::from(DEFAULT_INPUT_PATH), config.input_path);
        assert_eq!(
            PathBuf::from(DEFAULT_MINIGPT_CHECKPOINT_PATH),
            config.checkpoint_path
        );
        assert!(!config.serve);
        assert!(!config.benchmark_generation);
        assert!(!config.load_checkpoint);
        assert!(!config.load_latest_checkpoint);
        assert_eq!(LogFormat::Plain, config.log_format);
        assert_eq!(BenchmarkConfig::default(), config.benchmark_config);
        assert_eq!(
            DEFAULT_SERVER_ADDR.parse::<SocketAddr>().unwrap(),
            config.server_addr
        );
    }

    #[test]
    fn log_format_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--log-format".to_string(), "json".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(LogFormat::Json, config.log_format);
    }

    #[test]
    fn log_format_arg_takes_precedence_over_env() {
        let config = parse_runtime_config_with_checkpoint(
            ["--log-format".to_string(), "plain".to_string()],
            RuntimeEnv {
                log_format: Some("json".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .unwrap();

        assert_eq!(LogFormat::Plain, config.log_format);
    }

    #[test]
    fn invalid_log_format_returns_clear_error() {
        let err = parse_runtime_config(
            ["--log-format".to_string(), "yaml".to_string()],
            None,
            None,
            None,
        )
        .expect_err("invalid log format should fail");

        assert!(err.to_string().contains("unsupported log format"));
    }

    #[test]
    fn benchmark_config_can_be_selected_from_args() {
        let config = parse_runtime_config(
            [
                "--benchmark-prompt-lens".to_string(),
                "2,4".to_string(),
                "--benchmark-gen-lens".to_string(),
                "1,3".to_string(),
                "--benchmark-warmups".to_string(),
                "2".to_string(),
                "--benchmark-iterations".to_string(),
                "7".to_string(),
            ],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(vec![2, 4], config.benchmark_config.prompt_lens);
        assert_eq!(vec![1, 3], config.benchmark_config.gen_lens);
        assert_eq!(2, config.benchmark_config.warmups);
        assert_eq!(7, config.benchmark_config.iterations);
    }

    #[test]
    fn invalid_benchmark_config_returns_clear_error() {
        let err = parse_runtime_config(
            ["--benchmark-iterations".to_string(), "0".to_string()],
            None,
            None,
            None,
        )
        .expect_err("zero iterations should fail");

        assert!(
            err.to_string()
                .contains("benchmark iterations must be greater than zero")
        );
    }

    #[test]
    fn hyperparameters_default_to_main_constants() {
        let hyperparameters = Hyperparameters::default();

        assert_eq!(BLOCK_SIZE, hyperparameters.block_size);
        assert_eq!(BATCH_SIZE, hyperparameters.batch_size);
        assert_eq!(EMBED_DIM, hyperparameters.embed_dim);
        assert_eq!(NUM_HEADS, hyperparameters.num_heads);
        assert_eq!(HEAD_DIM, hyperparameters.head_dim);
        assert_eq!(NUM_LAYERS, hyperparameters.num_layers);
        assert_eq!(DROPOUT, hyperparameters.dropout);
        assert_eq!(LEARNING_RATE, hyperparameters.learning_rate);
        assert_eq!(TRAIN_STEPS, hyperparameters.train_steps);
        assert_eq!(EVAL_INTERVAL, hyperparameters.eval_interval);
        assert_eq!(GENERATE_TOKENS, hyperparameters.generate_tokens);
        assert_eq!(
            MINIGPT_GRAD_CLIP_NORM,
            hyperparameters.minigpt_grad_clip_norm
        );
        assert_eq!(PREFETCH_BATCHES, hyperparameters.prefetch_batches);
    }

    #[test]
    fn hyperparameters_can_override_training_settings_from_env() {
        // SAFETY: This unit test mutates process environment while not relying on
        // other environment-sensitive code in parallel.
        unsafe {
            env::set_var("RUSTY_GPT_TRAIN_STEPS", "7");
            env::set_var("RUSTY_GPT_EVAL_INTERVAL", "3");
            env::set_var("RUSTY_GPT_GENERATE_TOKENS", "11");
            env::set_var("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM", "0.5");
            env::set_var("RUSTY_GPT_PREFETCH_BATCHES", "4");
        }

        let hyperparameters = Hyperparameters::from_env().unwrap();

        assert_eq!(7, hyperparameters.train_steps);
        assert_eq!(3, hyperparameters.eval_interval);
        assert_eq!(11, hyperparameters.generate_tokens);
        assert_eq!(0.5, hyperparameters.minigpt_grad_clip_norm);
        assert_eq!(4, hyperparameters.prefetch_batches);

        // SAFETY: See note above.
        unsafe {
            env::remove_var("RUSTY_GPT_TRAIN_STEPS");
            env::remove_var("RUSTY_GPT_EVAL_INTERVAL");
            env::remove_var("RUSTY_GPT_GENERATE_TOKENS");
            env::remove_var("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM");
            env::remove_var("RUSTY_GPT_PREFETCH_BATCHES");
        }
    }

    #[test]
    fn hyperparameter_args_take_precedence_over_env() {
        let config = parse_runtime_config_with_checkpoint(
            [
                "--block-size".to_string(),
                "64".to_string(),
                "--batch-size".to_string(),
                "4".to_string(),
                "--embed-dim".to_string(),
                "32".to_string(),
                "--num-heads".to_string(),
                "4".to_string(),
                "--num-layers".to_string(),
                "2".to_string(),
                "--learning-rate".to_string(),
                "0.001".to_string(),
            ],
            RuntimeEnv {
                block_size: Some("128".to_string()),
                batch_size: Some("16".to_string()),
                embed_dim: Some("64".to_string()),
                num_heads: Some("8".to_string()),
                num_layers: Some("3".to_string()),
                learning_rate: Some("0.01".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .unwrap();

        assert_eq!(64, config.hyperparameters.block_size);
        assert_eq!(4, config.hyperparameters.batch_size);
        assert_eq!(32, config.hyperparameters.embed_dim);
        assert_eq!(4, config.hyperparameters.num_heads);
        assert_eq!(8, config.hyperparameters.head_dim);
        assert_eq!(2, config.hyperparameters.num_layers);
        assert_eq!(0.001, config.hyperparameters.learning_rate);
    }

    #[test]
    fn hyperparameters_reject_invalid_head_split() {
        let err = parse_runtime_config(
            [
                "--embed-dim".to_string(),
                "30".to_string(),
                "--num-heads".to_string(),
                "8".to_string(),
            ],
            None,
            None,
            None,
        )
        .expect_err("invalid head split should fail");

        assert!(
            err.to_string()
                .contains("embed_dim must be divisible by num_heads")
        );
    }

    #[test]
    fn hyperparameters_reject_non_positive_minigpt_gradient_clip_norm() {
        // SAFETY: This unit test mutates process environment while not relying on
        // other environment-sensitive code in parallel.
        unsafe {
            env::set_var("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM", "0");
        }

        let err = Hyperparameters::from_env()
            .expect_err("non-positive gradient clipping norm should fail");

        assert!(
            err.to_string()
                .contains("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM must be greater than zero")
        );

        // SAFETY: See note above.
        unsafe {
            env::remove_var("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM");
        }
    }

    #[test]
    fn split_training_and_value_tokens_uses_last_ten_percent_for_value() {
        let tokens: Vec<usize> = (0..100).collect();

        let (training_tokens, value_tokens, value_block_size) =
            split_training_and_value_tokens(&tokens, 8).unwrap();

        assert_eq!((0..90).collect::<Vec<_>>(), training_tokens);
        assert_eq!((90..100).collect::<Vec<_>>(), value_tokens);
        assert_eq!(8, value_block_size);
    }

    #[test]
    fn split_training_and_value_tokens_allows_shorter_value_block() {
        let tokens: Vec<usize> = (0..230).collect();

        let (_training_tokens, value_tokens, value_block_size) =
            split_training_and_value_tokens(&tokens, 128).unwrap();

        assert_eq!((207..230).collect::<Vec<_>>(), value_tokens);
        assert_eq!(22, value_block_size);
    }

    #[test]
    fn minigpt_training_saves_checkpoint_mpk_file() {
        type TestBackend = Autodiff<NdArray<f32, i64>>;
        let device = NdArrayDevice::Cpu;
        let checkpoint_path = std::env::temp_dir()
            .join("rusty-gpt-checkpoint-tests")
            .join(format!("mini-gpt-{}", std::process::id()));
        let saved_path = checkpoint_path.with_extension("mpk");
        let metadata_path = checkpoint_path.with_extension("metadata.json");
        let _ = fs::remove_file(&saved_path);
        let _ = fs::remove_file(&metadata_path);

        let hyperparameters = Hyperparameters {
            block_size: 4,
            batch_size: 2,
            embed_dim: 8,
            num_heads: 2,
            head_dim: 4,
            num_layers: 1,
            dropout: 0.0,
            learning_rate: 1e-4,
            train_steps: 1,
            eval_interval: 0,
            generate_tokens: 4,
            minigpt_grad_clip_norm: 1.0,
            prefetch_batches: 2,
        };
        let text = "abcdefghijklmnopqrstuvwxyz ".repeat(8);

        // SAFETY: This test mutates process environment before running any
        // threaded code and restores it before returning.
        unsafe {
            env::set_var(BPE_TOKENIZER_ENV, "tests/fixtures/tokenizer.json");
        }

        run_training_demo::<TestBackend>(
            &text,
            hyperparameters,
            ModelChoice::MiniGpt,
            &device,
            &checkpoint_path,
            TrainingDemoOptions {
                backend_label: "cpu",
                logger: EventLogger::stdout(LogFormat::Plain),
                benchmark_generation: false,
                benchmark_config: BenchmarkConfig::default(),
                input_source: "test".to_string(),
            },
        )
        .unwrap();

        // SAFETY: See note above.
        unsafe {
            env::remove_var(BPE_TOKENIZER_ENV);
        }

        assert!(
            saved_path.is_file(),
            "expected training to save {:?}",
            saved_path
        );
        assert!(
            metadata_path.is_file(),
            "expected training to save {:?}",
            metadata_path
        );

        let _ = fs::remove_file(saved_path);
        let _ = fs::remove_file(metadata_path);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn backend_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--backend".to_string(), "cuda".to_string()],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(BackendChoice::Cuda, config.backend);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn backend_cuda_arg_requires_feature() {
        let err = parse_runtime_config(
            ["--backend".to_string(), "cuda".to_string()],
            None,
            None,
            None,
        )
        .expect_err("cuda backend should fail when feature is disabled");

        assert!(err.to_string().contains("cuda backend is not compiled in"));
    }

    #[test]
    fn backend_can_be_selected_from_env() {
        let config = parse_runtime_config(Vec::<String>::new(), Some("cpu"), None, None).unwrap();

        assert_eq!(BackendChoice::Cpu, config.backend);
    }

    #[test]
    fn backend_arg_takes_precedence_over_env() {
        let config = parse_runtime_config(
            ["--backend".to_string(), "cpu".to_string()],
            Some("cuda"),
            None,
            None,
        )
        .unwrap();

        assert_eq!(BackendChoice::Cpu, config.backend);
    }

    #[test]
    fn input_path_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--input".to_string(), "data/custom.txt".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(PathBuf::from("data/custom.txt"), config.input_path);
    }

    #[test]
    fn input_arg_takes_precedence_over_env() {
        let config = parse_runtime_config(
            ["--input".to_string(), "data/from-arg.txt".to_string()],
            None,
            Some("data/from-env.txt"),
            None,
        )
        .unwrap();

        assert_eq!(PathBuf::from("data/from-arg.txt"), config.input_path);
    }

    #[test]
    fn checkpoint_path_can_be_selected_from_args() {
        let config = parse_runtime_config_with_checkpoint(
            ["--checkpoint".to_string(), "checkpoints/custom".to_string()],
            RuntimeEnv::default(),
        )
        .unwrap();

        assert_eq!(PathBuf::from("checkpoints/custom"), config.checkpoint_path);
    }

    #[test]
    fn checkpoint_arg_takes_precedence_over_env() {
        let config = parse_runtime_config_with_checkpoint(
            [
                "--checkpoint".to_string(),
                "checkpoints/from-arg".to_string(),
            ],
            RuntimeEnv {
                checkpoint: Some("checkpoints/from-env".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .unwrap();

        assert_eq!(
            PathBuf::from("checkpoints/from-arg"),
            config.checkpoint_path
        );
    }

    #[test]
    fn missing_checkpoint_arg_value_returns_clear_error() {
        let err = parse_runtime_config_with_checkpoint(
            ["--checkpoint".to_string()],
            RuntimeEnv::default(),
        )
        .expect_err("missing checkpoint value should fail");

        assert!(err.to_string().contains("--checkpoint requires a value"));
    }

    #[test]
    fn missing_backend_arg_value_returns_clear_error() {
        let err = parse_runtime_config(["--backend".to_string()], None, None, None)
            .expect_err("missing backend value should fail");

        assert!(err.to_string().contains("--backend requires a value"));
    }

    #[test]
    fn missing_input_arg_value_returns_clear_error() {
        let err = parse_runtime_config(["--input".to_string()], None, None, None)
            .expect_err("missing input value should fail");

        assert!(err.to_string().contains("--input requires a path"));
    }

    #[test]
    fn invalid_backend_returns_clear_error() {
        let err = parse_runtime_config(
            ["--backend".to_string(), "metal".to_string()],
            None,
            None,
            None,
        )
        .expect_err("invalid backend should fail");

        assert!(err.to_string().contains("unsupported backend"));
    }

    #[test]
    fn model_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--model".to_string(), "single-attention".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::SingleAttention, config.model);
    }

    #[test]
    fn multi_attention_model_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--model".to_string(), "multi-attention".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::MultiAttention, config.model);
    }

    #[test]
    fn minigpt_model_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--model".to_string(), "minigpt".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::MiniGpt, config.model);
    }

    #[test]
    fn mini_gpt_alias_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--model".to_string(), "mini-gpt".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::MiniGpt, config.model);
    }

    #[test]
    fn interactive_generation_can_be_selected_from_args() {
        let config = parse_runtime_config(
            [
                "--model".to_string(),
                "minigpt".to_string(),
                "--interactive-generate".to_string(),
            ],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::MiniGpt, config.model);
        assert!(config.interactive);
    }

    #[test]
    fn generation_benchmark_can_be_selected_from_args() {
        let config =
            parse_runtime_config(["--benchmark-generation".to_string()], None, None, None).unwrap();

        assert!(config.benchmark_generation);
    }

    #[test]
    fn checkpoint_loading_can_be_selected_from_args() {
        let config =
            parse_runtime_config(["--load-checkpoint".to_string()], None, None, None).unwrap();

        assert!(config.load_checkpoint);
    }

    #[test]
    fn latest_checkpoint_loading_can_be_selected_from_args() {
        let config =
            parse_runtime_config(["--load-latest-checkpoint".to_string()], None, None, None)
                .unwrap();

        assert!(config.load_latest_checkpoint);
    }

    #[test]
    fn checkpoint_loading_modes_are_mutually_exclusive() {
        let err = parse_runtime_config(
            [
                "--load-checkpoint".to_string(),
                "--load-latest-checkpoint".to_string(),
            ],
            None,
            None,
            None,
        )
        .expect_err("checkpoint loading modes should conflict");

        assert!(
            err.to_string()
                .contains("--load-checkpoint and --load-latest-checkpoint are mutually exclusive")
        );
    }

    #[test]
    fn latest_checkpoint_path_uses_newest_mpk_file_without_extension() {
        let dir = std::env::temp_dir().join(format!(
            "rusty-gpt-latest-checkpoint-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let old = dir.join("old.mpk");
        let new = dir.join("new.mpk");
        fs::write(&old, b"old").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        fs::write(&new, b"new").unwrap();

        let latest = latest_checkpoint_path(&dir).unwrap();

        assert_eq!(dir.join("new"), latest);

        let _ = fs::remove_file(old);
        let _ = fs::remove_file(new);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn generation_benchmark_requires_minigpt_training_model() {
        type TestBackend = Autodiff<NdArray<f32, i64>>;
        let device = NdArrayDevice::Cpu;
        let checkpoint_path = std::env::temp_dir().join("rusty-gpt-unused-benchmark-checkpoint");
        let hyperparameters = Hyperparameters {
            block_size: 4,
            batch_size: 2,
            embed_dim: 8,
            num_heads: 2,
            head_dim: 4,
            num_layers: 1,
            dropout: 0.0,
            learning_rate: 1e-4,
            train_steps: 1,
            eval_interval: 0,
            generate_tokens: 4,
            minigpt_grad_clip_norm: 1.0,
            prefetch_batches: 0,
        };

        let err = run_training_demo::<TestBackend>(
            &"abcdefghijklmnopqrstuvwxyz ".repeat(8),
            hyperparameters,
            ModelChoice::Trivial,
            &device,
            &checkpoint_path,
            TrainingDemoOptions {
                backend_label: "cpu",
                logger: EventLogger::stdout(LogFormat::Plain),
                benchmark_generation: true,
                benchmark_config: BenchmarkConfig::default(),
                input_source: "test".to_string(),
            },
        )
        .expect_err("benchmarking should require minigpt or compare");

        assert!(
            err.to_string()
                .contains("generation benchmarks require --model minigpt or compare")
        );
    }

    #[test]
    fn compare_model_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--model".to_string(), "compare".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::Compare, config.model);
        assert_eq!(
            vec![
                ModelChoice::Trivial,
                ModelChoice::SingleAttention,
                ModelChoice::MultiAttention,
                ModelChoice::MiniGpt
            ],
            config.model.comparison_models()
        );
    }

    #[test]
    fn model_can_be_selected_from_env() {
        let config =
            parse_runtime_config(Vec::<String>::new(), None, None, Some("single-attention"))
                .unwrap();

        assert_eq!(ModelChoice::SingleAttention, config.model);
    }

    #[test]
    fn model_arg_takes_precedence_over_env() {
        let config = parse_runtime_config(
            ["--model".to_string(), "trivial".to_string()],
            None,
            None,
            Some("single-attention"),
        )
        .unwrap();

        assert_eq!(ModelChoice::Trivial, config.model);
    }

    #[test]
    fn missing_model_arg_value_returns_clear_error() {
        let err = parse_runtime_config(["--model".to_string()], None, None, None)
            .expect_err("missing model value should fail");

        assert!(err.to_string().contains("--model requires a value"));
    }

    #[test]
    fn invalid_model_returns_clear_error() {
        let err = parse_runtime_config(
            ["--model".to_string(), "large".to_string()],
            None,
            None,
            None,
        )
        .expect_err("invalid model should fail");

        assert!(err.to_string().contains("unsupported model"));
    }
}
