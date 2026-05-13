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
use rusty_gpt::model::persistence::{load_model, save_model};
use rusty_gpt::model::{
    MiniGpt, MiniGptConfig, MultiAttentionModel, SingleAttentionModel, TrainingLogContext,
    TrainingLogFormat, TrainingParams, TrivialModel,
};
use rusty_gpt::server;
use rusty_gpt::server::ServerState;
use rusty_gpt::tokenizer::RuntimeTokenizer;
use rusty_gpt::utils::benchmark_generation;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DEFAULT_INPUT_PATH: &str = "data/input.txt";
const DEFAULT_MINIGPT_CHECKPOINT_PATH: &str = "checkpoints/mini_gpt";
const DEFAULT_BPE_TOKENIZER_PATH: &str = "checkpoints/tokenizer.json";
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

#[derive(Debug, Clone, Copy)]
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
    fn from_env() -> Result<Self> {
        let mut hyperparameters = Self::default();

        apply_env_override("RUSTY_GPT_TRAIN_STEPS", &mut hyperparameters.train_steps)?;
        apply_env_override(
            "RUSTY_GPT_EVAL_INTERVAL",
            &mut hyperparameters.eval_interval,
        )?;
        apply_env_override(
            "RUSTY_GPT_GENERATE_TOKENS",
            &mut hyperparameters.generate_tokens,
        )?;
        apply_env_override(
            "RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM",
            &mut hyperparameters.minigpt_grad_clip_norm,
        )?;
        apply_env_override(
            "RUSTY_GPT_PREFETCH_BATCHES",
            &mut hyperparameters.prefetch_batches,
        )?;

        if hyperparameters.minigpt_grad_clip_norm <= 0.0 {
            bail!("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM must be greater than zero");
        }

        Ok(hyperparameters)
    }
}

fn apply_env_override<T>(name: &str, target: &mut T) -> Result<()>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    if let Ok(value) = env::var(name) {
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
        env::var("RUSTY_GPT_BACKEND").ok().as_deref(),
        env::var("RUSTY_GPT_INPUT").ok().as_deref(),
        env::var("RUSTY_GPT_MODEL").ok().as_deref(),
        env::var("RUSTY_GPT_MINIGPT_CHECKPOINT").ok().as_deref(),
        env::var("RUSTY_GPT_SERVER_ADDR").ok().as_deref(),
    )?;
    let text = load_input_text(&config.input_path)?;
    let hyperparameters = Hyperparameters::from_env()?;

    if config.serve {
        return match config.backend {
            BackendChoice::Cpu => run_http_server_with_runtime::<NdArray<f32, i64>>(
                &text,
                hyperparameters,
                config.server_addr,
                &config.checkpoint_path,
                config.load_checkpoint,
                config.load_latest_checkpoint,
                &NdArrayDevice::Cpu,
            ),
            #[cfg(feature = "cuda")]
            BackendChoice::Cuda => run_http_server_with_runtime::<Cuda>(
                &text,
                hyperparameters,
                config.server_addr,
                &config.checkpoint_path,
                config.load_checkpoint,
                config.load_latest_checkpoint,
                &CudaDevice::default(),
            ),
        };
    }

    match config.backend {
        BackendChoice::Cpu => run_cpu_demo(
            &text,
            hyperparameters,
            config.model,
            config.interactive,
            config.benchmark_generation,
            &config.checkpoint_path,
        ),
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => {
            if config.interactive {
                bail!("interactive generation currently requires --backend cpu");
            }
            let device = CudaDevice::default();
            run_demo::<Cuda>(&text, hyperparameters, config.model, &device)?;
            run_training_demo::<Autodiff<Cuda>>(
                &text,
                hyperparameters,
                config.model,
                &device,
                &config.checkpoint_path,
                TrainingDemoOptions {
                    backend_label: "cuda",
                    log_format: TrainingLogFormat::Json,
                    benchmark_generation: config.benchmark_generation,
                },
            )
        }
    }
}

fn run_cpu_demo(
    text: &str,
    hyperparameters: Hyperparameters,
    model_choice: ModelChoice,
    interactive: bool,
    benchmark_generation: bool,
    checkpoint_path: &Path,
) -> Result<()> {
    let device = NdArrayDevice::Cpu;
    run_demo::<NdArray<f32, i64>>(text, hyperparameters, model_choice, &device)?;
    if interactive {
        if benchmark_generation {
            bail!("generation benchmarks cannot run with --interactive-generate");
        }
        if model_choice != ModelChoice::MiniGpt {
            bail!("interactive generation requires --model minigpt");
        }
        run_interactive_minigpt_generation::<Autodiff<NdArray<f32, i64>>>(
            text,
            hyperparameters,
            &device,
            checkpoint_path,
        )
    } else {
        run_training_demo::<Autodiff<NdArray<f32, i64>>>(
            text,
            hyperparameters,
            model_choice,
            &device,
            checkpoint_path,
            TrainingDemoOptions {
                backend_label: "cpu",
                log_format: TrainingLogFormat::Plain,
                benchmark_generation,
            },
        )
    }
}

fn run_http_server_with_runtime<B>(
    text: &str,
    hyperparameters: Hyperparameters,
    server_addr: SocketAddr,
    checkpoint_path: &Path,
    load_checkpoint_enabled: bool,
    load_latest_checkpoint_enabled: bool,
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

    runtime.block_on(run_http_server::<B>(
        text,
        hyperparameters,
        server_addr,
        checkpoint_path,
        load_checkpoint_enabled,
        load_latest_checkpoint_enabled,
        device,
    ))
}

async fn run_http_server<B>(
    _text: &str,
    hyperparameters: Hyperparameters,
    server_addr: SocketAddr,
    checkpoint_path: &Path,
    load_checkpoint_enabled: bool,
    load_latest_checkpoint_enabled: bool,
    device: &B::Device,
) -> Result<()>
where
    B: Backend + Send + Sync + 'static,
    B::Device: Send + Sync + 'static,
{
    let tokenizer = load_minigpt_tokenizer()?;
    let template = new_minigpt::<B>(tokenizer.vocab_size(), hyperparameters, device);
    let model = if load_latest_checkpoint_enabled {
        let latest_checkpoint = latest_checkpoint_path(Path::new(DEFAULT_CHECKPOINT_DIR))?;
        load_minigpt_checkpoint(template, &latest_checkpoint, device)?
    } else if load_checkpoint_enabled {
        load_minigpt_checkpoint(template, checkpoint_path, device)?
    } else {
        template
    };
    let state = Arc::new(ServerState::new(model, tokenizer, device.clone()));
    let app = Router::new()
        .nest("/api", server::router::<B>())
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(server_addr)
        .await
        .with_context(|| format!("failed to bind HTTP server on {server_addr}"))?;

    println!("Serving GPT API on http://{server_addr}");
    axum::serve(listener, app)
        .await
        .context("HTTP server failed")
}

fn run_demo<B: Backend>(
    text: &str,
    hyperparameters: Hyperparameters,
    model_choice: ModelChoice,
    device: &B::Device,
) -> Result<()> {
    let tokenizer = tokenizer_for_model(text, model_choice)?;
    let encoded = tokenizer.encode(text);
    println!("Vocab size: {}", tokenizer.vocab_size());
    println!("Input chars: {}", text.chars().count());
    println!(
        "Hyperparameters: block_size={}, batch_size={}, embed_dim={}, num_heads={}, head_dim={}, num_layers={}, dropout={}, lr={}, train_steps={}, eval_interval={}, generate_tokens={}, minigpt_grad_clip_norm={}, prefetch_batches={}",
        hyperparameters.block_size,
        hyperparameters.batch_size,
        hyperparameters.embed_dim,
        hyperparameters.num_heads,
        hyperparameters.head_dim,
        hyperparameters.num_layers,
        hyperparameters.dropout,
        hyperparameters.learning_rate,
        hyperparameters.train_steps,
        hyperparameters.eval_interval,
        hyperparameters.generate_tokens,
        hyperparameters.minigpt_grad_clip_norm,
        hyperparameters.prefetch_batches
    );
    let preview_len = encoded.len().min(80);
    println!(
        "Decoded preview: {:?}",
        tokenizer.decode(&encoded[..preview_len])
    );

    let data_loader = DataLoader {
        tokens: encoded,
        block_size: hyperparameters.block_size,
        batch_size: hyperparameters.batch_size,
    };
    let (x, y) = data_loader
        .next_batch::<B>(device)
        .map_err(anyhow::Error::msg)
        .context("failed to build demo batch")?;
    for model_choice in model_choice.comparison_models() {
        let logits = run_model_forward(
            model_choice,
            tokenizer.vocab_size(),
            hyperparameters,
            x.clone(),
            device,
        );
        println!(
            "{} logits shape: {:?}",
            model_choice.label(),
            logits.shape().dims::<3>()
        );
    }
    println!("x shape: {:?}", x.shape().dims::<2>());
    println!("y shape: {:?}", y.shape().dims::<2>());

    Ok(())
}

fn run_interactive_minigpt_generation<B: burn::tensor::backend::AutodiffBackend>(
    _text: &str,
    hyperparameters: Hyperparameters,
    device: &B::Device,
    checkpoint_path: &Path,
) -> Result<()> {
    let tokenizer = load_minigpt_tokenizer()?;
    let template = new_minigpt::<B>(tokenizer.vocab_size(), hyperparameters, device);
    let model = load_minigpt_checkpoint(template, checkpoint_path, device)?;

    interactive_generation_loop(&model, &tokenizer, hyperparameters.generate_tokens, device)
}

fn load_minigpt_checkpoint<B: Backend>(
    template: MiniGpt<B>,
    checkpoint_path: &Path,
    device: &B::Device,
) -> Result<MiniGpt<B>> {
    let model = load_model(template, checkpoint_path, device).with_context(|| {
        format!(
            "failed to load minigpt checkpoint from {:?}",
            checkpoint_path.with_extension("mpk")
        )
    })?;
    println!(
        "Loaded minigpt checkpoint from {:?}",
        checkpoint_path.with_extension("mpk")
    );

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
    RuntimeTokenizer::load_bpe(Path::new(DEFAULT_BPE_TOKENIZER_PATH)).with_context(|| {
        format!(
            "failed to load default MiniGPT BPE tokenizer from {DEFAULT_BPE_TOKENIZER_PATH}; train one with `cargo run --bin train-tokenizer -- --corpus data/fafolang.txt --vocab-size 2048 --output {DEFAULT_BPE_TOKENIZER_PATH}`"
        )
    })
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

fn run_training_demo<B: burn::tensor::backend::AutodiffBackend>(
    text: &str,
    hyperparameters: Hyperparameters,
    model_choice: ModelChoice,
    device: &B::Device,
    checkpoint_path: &Path,
    options: TrainingDemoOptions,
) -> Result<()> {
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
        println!("Training {} model", model_choice.label());
        train_model(TrainingRun::<B> {
            model_choice,
            data_loader: &data_loader,
            value_loader: &value_loader,
            device,
            vocab_size: tokenizer.vocab_size(),
            hyperparameters,
            checkpoint_path,
            backend_label: options.backend_label,
            log_format: options.log_format,
            benchmark_generation: options.benchmark_generation,
        })?;
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct TrainingDemoOptions {
    backend_label: &'static str,
    log_format: TrainingLogFormat,
    benchmark_generation: bool,
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
    log_format: TrainingLogFormat,
    benchmark_generation: bool,
}

fn train_model<B: burn::tensor::backend::AutodiffBackend>(run: TrainingRun<'_, B>) -> Result<()> {
    let log_context = TrainingLogContext {
        backend: run.backend_label,
        model: run.model_choice.label(),
        format: run.log_format,
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
            let _model = TrivialModel::<B>::train(
                run.data_loader,
                run.value_loader,
                run.device,
                run.vocab_size,
                run.hyperparameters.embed_dim,
                params,
            )
            .map_err(anyhow::Error::msg)
            .context("failed to train trivial model")?;
        }
        ModelChoice::SingleAttention => {
            let _model = SingleAttentionModel::<B>::train(
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
        }
        ModelChoice::MultiAttention => {
            let _model = MultiAttentionModel::<B>::train(
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
        }
        ModelChoice::MiniGpt => {
            let model = MiniGpt::<B>::train(
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
            if run.benchmark_generation {
                benchmark_generation(&model, run.device);
            }
            save_minigpt_checkpoint(model, run.checkpoint_path)?;
        }
        ModelChoice::Compare => unreachable!("compare should be expanded before training dispatch"),
    }

    Ok(())
}

fn save_minigpt_checkpoint<B: burn::tensor::backend::AutodiffBackend>(
    model: MiniGpt<B>,
    checkpoint_path: &Path,
) -> Result<()> {
    if let Some(parent) = checkpoint_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create checkpoint directory {:?}", parent))?;
    }

    save_model(model, checkpoint_path)
        .with_context(|| format!("failed to save minigpt checkpoint to {:?}", checkpoint_path))?;
    println!(
        "Saved minigpt checkpoint to {:?}",
        checkpoint_path.with_extension("mpk")
    );

    Ok(())
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeConfig {
    backend: BackendChoice,
    model: ModelChoice,
    input_path: PathBuf,
    checkpoint_path: PathBuf,
    interactive: bool,
    benchmark_generation: bool,
    load_checkpoint: bool,
    load_latest_checkpoint: bool,
    serve: bool,
    server_addr: SocketAddr,
}

fn load_input_text(path: &Path) -> Result<String> {
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
    parse_runtime_config_with_checkpoint(args, env_backend, env_input, env_model, None, None)
}

fn parse_runtime_config_with_checkpoint<I, S>(
    args: I,
    env_backend: Option<&str>,
    env_input: Option<&str>,
    env_model: Option<&str>,
    env_checkpoint: Option<&str>,
    env_server_addr: Option<&str>,
) -> Result<RuntimeConfig>
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
        .or(env_server_addr)
        .unwrap_or(DEFAULT_SERVER_ADDR);
    let server_addr = server_addr_text
        .parse()
        .with_context(|| format!("invalid server address '{server_addr_text}'"))?;

    Ok(RuntimeConfig {
        backend: parse_backend_name(arg_backend.or(env_backend).unwrap_or("cpu"))?,
        model: parse_model_name(arg_model.or(env_model).unwrap_or("trivial"))?,
        input_path: PathBuf::from(arg_input.or(env_input).unwrap_or(DEFAULT_INPUT_PATH)),
        checkpoint_path: PathBuf::from(
            arg_checkpoint
                .or(env_checkpoint)
                .unwrap_or(DEFAULT_MINIGPT_CHECKPOINT_PATH),
        ),
        interactive,
        benchmark_generation,
        load_checkpoint,
        load_latest_checkpoint,
        serve,
        server_addr,
    })
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
        assert_eq!(ModelChoice::Trivial, config.model);
        assert_eq!(PathBuf::from(DEFAULT_INPUT_PATH), config.input_path);
        assert_eq!(
            PathBuf::from(DEFAULT_MINIGPT_CHECKPOINT_PATH),
            config.checkpoint_path
        );
        assert!(!config.serve);
        assert!(!config.benchmark_generation);
        assert!(!config.load_checkpoint);
        assert!(!config.load_latest_checkpoint);
        assert_eq!(
            DEFAULT_SERVER_ADDR.parse::<SocketAddr>().unwrap(),
            config.server_addr
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
        let _ = fs::remove_file(&saved_path);

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

        run_training_demo::<TestBackend>(
            &text,
            hyperparameters,
            ModelChoice::MiniGpt,
            &device,
            &checkpoint_path,
            TrainingDemoOptions {
                backend_label: "cpu",
                log_format: TrainingLogFormat::Plain,
                benchmark_generation: false,
            },
        )
        .unwrap();

        assert!(
            saved_path.is_file(),
            "expected training to save {:?}",
            saved_path
        );

        let _ = fs::remove_file(saved_path);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn backend_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--backend".to_string(), "cuda".to_string()],
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
            None,
            None,
            None,
            None,
            None,
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
            None,
            None,
            None,
            Some("checkpoints/from-env"),
            None,
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
            None,
            None,
            None,
            None,
            None,
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
                log_format: TrainingLogFormat::Plain,
                benchmark_generation: true,
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
