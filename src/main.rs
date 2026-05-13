mod loader;
pub mod model;
pub mod server;
mod tokenizer;

use crate::loader::data::DataLoader;
use crate::model::persistence::load_model;
use crate::model::{
    MiniGpt, MultiAttentionModel, SingleAttentionModel, TrainingLogContext, TrainingLogFormat,
    TrivialModel,
};
use crate::server::ServerState;
use crate::tokenizer::char::CharTokenizer;
use anyhow::{Context, Result, bail};
use axum::Router;
use burn::backend::Autodiff;
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "cuda")]
use burn::backend::cuda::CudaDevice;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::backend::Backend;
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DEFAULT_INPUT_PATH: &str = "data/input.txt";
const DEFAULT_MINIGPT_CHECKPOINT_PATH: &str = "checkpoints/mini_gpt";
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
        }
    }
}

impl Hyperparameters {
    fn from_env() -> Result<Self> {
        let mut hyperparameters = Self::default();

        if let Ok(train_steps) = env::var("RUSTY_GPT_TRAIN_STEPS") {
            hyperparameters.train_steps = train_steps
                .parse()
                .with_context(|| format!("invalid RUSTY_GPT_TRAIN_STEPS value: {train_steps}"))?;
        }
        if let Ok(eval_interval) = env::var("RUSTY_GPT_EVAL_INTERVAL") {
            hyperparameters.eval_interval = eval_interval.parse().with_context(|| {
                format!("invalid RUSTY_GPT_EVAL_INTERVAL value: {eval_interval}")
            })?;
        }
        if let Ok(generate_tokens) = env::var("RUSTY_GPT_GENERATE_TOKENS") {
            hyperparameters.generate_tokens = generate_tokens.parse().with_context(|| {
                format!("invalid RUSTY_GPT_GENERATE_TOKENS value: {generate_tokens}")
            })?;
        }
        if let Ok(grad_clip_norm) = env::var("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM") {
            hyperparameters.minigpt_grad_clip_norm = grad_clip_norm.parse().with_context(|| {
                format!("invalid RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM value: {grad_clip_norm}")
            })?;
        }
        if hyperparameters.minigpt_grad_clip_norm <= 0.0 {
            bail!("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM must be greater than zero");
        }

        Ok(hyperparameters)
    }
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
                &NdArrayDevice::Cpu,
            ),
            #[cfg(feature = "cuda")]
            BackendChoice::Cuda => run_http_server_with_runtime::<Cuda>(
                &text,
                hyperparameters,
                config.server_addr,
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
                "cuda",
                TrainingLogFormat::Json,
            )
        }
    }
}

fn run_cpu_demo(
    text: &str,
    hyperparameters: Hyperparameters,
    model_choice: ModelChoice,
    interactive: bool,
    checkpoint_path: &Path,
) -> Result<()> {
    let device = NdArrayDevice::Cpu;
    run_demo::<NdArray<f32, i64>>(text, hyperparameters, model_choice, &device)?;
    if interactive {
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
            "cpu",
            TrainingLogFormat::Plain,
        )
    }
}

fn run_http_server_with_runtime<B>(
    text: &str,
    hyperparameters: Hyperparameters,
    server_addr: SocketAddr,
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
        device,
    ))
}

async fn run_http_server<B>(
    text: &str,
    hyperparameters: Hyperparameters,
    server_addr: SocketAddr,
    device: &B::Device,
) -> Result<()>
where
    B: Backend + Send + Sync + 'static,
    B::Device: Send + Sync + 'static,
{
    let tokenizer = CharTokenizer::from_text(text);
    let model = MiniGpt::<B>::new(
        tokenizer.vocab_size(),
        hyperparameters.embed_dim,
        hyperparameters.num_layers,
        hyperparameters.block_size,
        hyperparameters.num_heads,
        device,
    );
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
    let tokenizer = CharTokenizer::from_text(text);
    let encoded = tokenizer.encode(text);
    println!("Vocab size: {}", tokenizer.vocab_size());
    println!("Input chars: {}", text.chars().count());
    println!(
        "Hyperparameters: block_size={}, batch_size={}, embed_dim={}, num_heads={}, head_dim={}, num_layers={}, dropout={}, lr={}, train_steps={}, eval_interval={}, generate_tokens={}, minigpt_grad_clip_norm={}",
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
        hyperparameters.minigpt_grad_clip_norm
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
    text: &str,
    hyperparameters: Hyperparameters,
    device: &B::Device,
    checkpoint_path: &Path,
) -> Result<()> {
    let tokenizer = CharTokenizer::from_text(text);
    let template = MiniGpt::<B>::new(
        tokenizer.vocab_size(),
        hyperparameters.embed_dim,
        hyperparameters.num_layers,
        hyperparameters.block_size,
        hyperparameters.num_heads,
        device,
    );
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

    interactive_generation_loop(&model, &tokenizer, hyperparameters.generate_tokens, device)
}

fn interactive_generation_loop<B: Backend>(
    model: &MiniGpt<B>,
    tokenizer: &CharTokenizer,
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
            let model = MiniGpt::<B>::new(
                vocab_size,
                hyperparameters.embed_dim,
                hyperparameters.num_layers,
                hyperparameters.block_size,
                hyperparameters.num_heads,
                device,
            );
            model.forward_tokens(input)
        }
        ModelChoice::Compare => unreachable!("compare should be expanded before forward dispatch"),
    }
}

fn run_training_demo<B: burn::tensor::backend::AutodiffBackend>(
    text: &str,
    hyperparameters: Hyperparameters,
    model_choice: ModelChoice,
    device: &B::Device,
    backend_label: &'static str,
    log_format: TrainingLogFormat,
) -> Result<()> {
    let tokenizer = CharTokenizer::from_text(text);
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
        train_model::<B>(
            model_choice,
            &data_loader,
            &value_loader,
            device,
            tokenizer.vocab_size(),
            hyperparameters,
            backend_label,
            log_format,
        )?;
    }

    Ok(())
}

fn train_model<B: burn::tensor::backend::AutodiffBackend>(
    model_choice: ModelChoice,
    data_loader: &DataLoader,
    value_loader: &DataLoader,
    device: &B::Device,
    vocab_size: usize,
    hyperparameters: Hyperparameters,
    backend_label: &'static str,
    log_format: TrainingLogFormat,
) -> Result<()> {
    let log_context = TrainingLogContext {
        backend: backend_label,
        model: model_choice.label(),
        format: log_format,
    };

    match model_choice {
        ModelChoice::Trivial => {
            let _model = TrivialModel::<B>::train(
                data_loader,
                value_loader,
                device,
                vocab_size,
                hyperparameters.embed_dim,
                hyperparameters.learning_rate,
                hyperparameters.train_steps,
                hyperparameters.eval_interval,
                log_context,
            )
            .map_err(anyhow::Error::msg)
            .context("failed to train trivial model")?;
        }
        ModelChoice::SingleAttention => {
            let _model = SingleAttentionModel::<B>::train(
                data_loader,
                value_loader,
                device,
                vocab_size,
                hyperparameters.embed_dim,
                hyperparameters.head_dim,
                hyperparameters.learning_rate,
                hyperparameters.train_steps,
                hyperparameters.eval_interval,
                log_context,
            )
            .map_err(anyhow::Error::msg)
            .context("failed to train single attention model")?;
        }
        ModelChoice::MultiAttention => {
            let _model = MultiAttentionModel::<B>::train(
                data_loader,
                value_loader,
                device,
                vocab_size,
                hyperparameters.embed_dim,
                hyperparameters.num_heads,
                hyperparameters.learning_rate,
                hyperparameters.train_steps,
                hyperparameters.eval_interval,
                log_context,
            )
            .map_err(anyhow::Error::msg)
            .context("failed to train multi attention model")?;
        }
        ModelChoice::MiniGpt => {
            let _model = MiniGpt::<B>::train(
                data_loader,
                value_loader,
                device,
                vocab_size,
                hyperparameters.embed_dim,
                hyperparameters.num_layers,
                hyperparameters.block_size,
                hyperparameters.num_heads,
                hyperparameters.learning_rate,
                hyperparameters.train_steps,
                hyperparameters.eval_interval,
                hyperparameters.minigpt_grad_clip_norm,
                log_context,
            )
            .map_err(anyhow::Error::msg)
            .context("failed to train minigpt model")?;
        }
        ModelChoice::Compare => unreachable!("compare should be expanded before training dispatch"),
    }

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
            "--serve" => {
                serve = true;
                index += 1;
            }
            other => bail!("unsupported argument: {other}"),
        }
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
        }

        let hyperparameters = Hyperparameters::from_env().unwrap();

        assert_eq!(7, hyperparameters.train_steps);
        assert_eq!(3, hyperparameters.eval_interval);
        assert_eq!(11, hyperparameters.generate_tokens);
        assert_eq!(0.5, hyperparameters.minigpt_grad_clip_norm);

        // SAFETY: See note above.
        unsafe {
            env::remove_var("RUSTY_GPT_TRAIN_STEPS");
            env::remove_var("RUSTY_GPT_EVAL_INTERVAL");
            env::remove_var("RUSTY_GPT_GENERATE_TOKENS");
            env::remove_var("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM");
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
