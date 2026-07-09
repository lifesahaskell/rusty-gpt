use anyhow::{Context, Result, bail};
use axum::Router;
use burn::backend::Autodiff;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::backend::Backend;
use rusty_gpt::loader::data::DataLoader;
use rusty_gpt::model::persistence::sha256_file_hex;
use rusty_gpt::model::{MiniGpt, MultiAttentionModel, SingleAttentionModel, TrivialModel};
use rusty_gpt::observability::{EventLogger, RuntimeEvent};
use rusty_gpt::server;
use rusty_gpt::server::{CheckpointSource, ServerLimits, ServerProvenance, ServerState};
use rusty_gpt::tokenizer::RuntimeTokenizer;
use rusty_gpt::utils::BenchmarkConfig;
use std::io::{self, BufRead, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::runtime_assets::{
    DEFAULT_CHECKPOINT_DIR, latest_checkpoint_path, load_minigpt_checkpoint,
    load_minigpt_tokenizer, minigpt_tokenizer_path, tokenizer_for_model,
};
use crate::runtime_config::{Hyperparameters, ModelChoice};
use crate::runtime_training::{TrainingDemoOptions, run_training_demo};

pub(crate) struct CpuDemoOptions<'a> {
    pub(crate) model_choice: ModelChoice,
    pub(crate) interactive: bool,
    pub(crate) benchmark_generation: bool,
    pub(crate) benchmark_config: &'a BenchmarkConfig,
    pub(crate) logger: EventLogger,
    pub(crate) checkpoint_path: &'a Path,
    pub(crate) input_source: &'a str,
    /// `--resume-from` checkpoint (already confined to `checkpoints/`), or
    /// `None` for a fresh run. Only used on the MiniGPT training path.
    pub(crate) resume_from: Option<&'a Path>,
}

pub(crate) fn run_cpu_demo(
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
                resume_from: options.resume_from.map(Path::to_path_buf),
            },
        )
    }
}

pub(crate) struct ServerRuntimeOptions<'a> {
    pub(crate) server_addr: SocketAddr,
    pub(crate) checkpoint_path: &'a Path,
    pub(crate) load_checkpoint_enabled: bool,
    pub(crate) load_latest_checkpoint_enabled: bool,
    pub(crate) backend_label: &'static str,
    pub(crate) logger: EventLogger,
    pub(crate) limits: ServerLimits,
}

pub(crate) fn run_http_server_with_runtime<B>(
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
    let started_at = Instant::now();
    let tokenizer = load_minigpt_tokenizer()?;
    let template = new_minigpt::<B>(tokenizer.vocab_size(), hyperparameters, device);
    let (model, checkpoint_source, checkpoint_path): (_, _, Option<PathBuf>) =
        if options.load_latest_checkpoint_enabled {
            let latest = latest_checkpoint_path(Path::new(DEFAULT_CHECKPOINT_DIR))?;
            let loaded = load_minigpt_checkpoint(template, &latest, device, &options.logger)?;
            (loaded, CheckpointSource::Latest, Some(latest))
        } else if options.load_checkpoint_enabled {
            let explicit = options.checkpoint_path.to_path_buf();
            let loaded = load_minigpt_checkpoint(template, &explicit, device, &options.logger)?;
            (loaded, CheckpointSource::Explicit, Some(explicit))
        } else {
            (template, CheckpointSource::None, None)
        };
    let (checkpoint_basename, checkpoint_sha256) = match &checkpoint_path {
        Some(path) => {
            let mpk = path.with_extension("mpk");
            let basename = mpk
                .file_name()
                .and_then(|name| name.to_str())
                .map(String::from);
            (basename, sha256_file_hex(&mpk)?)
        }
        None => (None, None),
    };
    let tokenizer_sha256 = sha256_file_hex(Path::new(&minigpt_tokenizer_path()))?;
    let provenance = ServerProvenance {
        started_at,
        checkpoint_source,
        checkpoint_basename,
        checkpoint_sha256,
        tokenizer_sha256,
    };
    let state = Arc::new(ServerState::new_with_limits(
        model,
        tokenizer,
        device.clone(),
        options.logger.clone(),
        provenance,
        options.limits,
    ));
    let vocab_size = state.model_vocab_size();
    let block_size = state.model_block_size();
    let app = Router::new()
        .nest("/api", server::router_with_limits::<B>(options.limits))
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
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .context("HTTP server failed")
}

pub(crate) fn run_demo<B: Backend>(
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
