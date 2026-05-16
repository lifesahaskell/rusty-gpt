use anyhow::{Context, Result, bail};
use rusty_gpt::loader::InputSource;
use rusty_gpt::observability::LogFormat;
use rusty_gpt::utils::{BenchmarkConfig, parse_usize_list};
use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;

pub(crate) const DEFAULT_INPUT_PATH: &str = "data/input.txt";
pub(crate) const DEFAULT_MINIGPT_CHECKPOINT_PATH: &str = "checkpoints/mini_gpt";
pub(crate) const DEFAULT_SERVER_ADDR: &str = "127.0.0.1:8787";
const LOG_FORMAT_ENV: &str = "RUSTY_GPT_LOG_FORMAT";
const BENCHMARK_PROMPT_LENS_ENV: &str = "RUSTY_GPT_BENCHMARK_PROMPT_LENS";
const BENCHMARK_GEN_LENS_ENV: &str = "RUSTY_GPT_BENCHMARK_GEN_LENS";
const BENCHMARK_WARMUPS_ENV: &str = "RUSTY_GPT_BENCHMARK_WARMUPS";
const BENCHMARK_ITERATIONS_ENV: &str = "RUSTY_GPT_BENCHMARK_ITERATIONS";

pub(crate) const BLOCK_SIZE: usize = 128;
pub(crate) const BATCH_SIZE: usize = 32;
pub(crate) const EMBED_DIM: usize = 128;
pub(crate) const NUM_HEADS: usize = 4;
pub(crate) const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS;
pub(crate) const NUM_LAYERS: usize = 4;
pub(crate) const DROPOUT: f64 = 0.1;
pub(crate) const LEARNING_RATE: f64 = 1e-4;
pub(crate) const TRAIN_STEPS: usize = 1000;
pub(crate) const EVAL_INTERVAL: usize = 100;
pub(crate) const GENERATE_TOKENS: usize = 80;
pub(crate) const MINIGPT_GRAD_CLIP_NORM: f32 = 1.0;
pub(crate) const PREFETCH_BATCHES: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Hyperparameters {
    pub(crate) block_size: usize,
    pub(crate) batch_size: usize,
    pub(crate) embed_dim: usize,
    pub(crate) num_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) num_layers: usize,
    pub(crate) dropout: f64,
    pub(crate) learning_rate: f64,
    pub(crate) train_steps: usize,
    pub(crate) eval_interval: usize,
    pub(crate) generate_tokens: usize,
    pub(crate) minigpt_grad_clip_norm: f32,
    pub(crate) prefetch_batches: usize,
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
    pub(crate) fn from_env() -> Result<Self> {
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
pub(crate) enum BackendChoice {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl BackendChoice {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelChoice {
    Trivial,
    SingleAttention,
    MultiAttention,
    MiniGpt,
    Compare,
}

impl ModelChoice {
    pub(crate) fn comparison_models(self) -> Vec<ModelChoice> {
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

    pub(crate) fn label(self) -> &'static str {
        match self {
            ModelChoice::Trivial => "trivial",
            ModelChoice::SingleAttention => "single-attention",
            ModelChoice::MultiAttention => "multi-attention",
            ModelChoice::MiniGpt => "minigpt",
            ModelChoice::Compare => "compare",
        }
    }

    pub(crate) fn includes_minigpt(self) -> bool {
        matches!(self, ModelChoice::MiniGpt | ModelChoice::Compare)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RuntimeConfig {
    pub(crate) backend: BackendChoice,
    pub(crate) model: ModelChoice,
    pub(crate) input_path: PathBuf,
    pub(crate) input_source: InputSource,
    pub(crate) checkpoint_path: PathBuf,
    pub(crate) hyperparameters: Hyperparameters,
    pub(crate) interactive: bool,
    pub(crate) benchmark_generation: bool,
    pub(crate) load_checkpoint: bool,
    pub(crate) load_latest_checkpoint: bool,
    pub(crate) serve: bool,
    pub(crate) server_addr: SocketAddr,
    pub(crate) log_format: LogFormat,
    pub(crate) benchmark_config: BenchmarkConfig,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct RuntimeEnv {
    pub(crate) backend: Option<String>,
    pub(crate) input: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) checkpoint: Option<String>,
    pub(crate) server_addr: Option<String>,
    pub(crate) log_format: Option<String>,
    pub(crate) benchmark_prompt_lens: Option<String>,
    pub(crate) benchmark_gen_lens: Option<String>,
    pub(crate) benchmark_warmups: Option<String>,
    pub(crate) benchmark_iterations: Option<String>,
    pub(crate) block_size: Option<String>,
    pub(crate) batch_size: Option<String>,
    pub(crate) embed_dim: Option<String>,
    pub(crate) num_heads: Option<String>,
    pub(crate) num_layers: Option<String>,
    pub(crate) dropout: Option<String>,
    pub(crate) learning_rate: Option<String>,
    pub(crate) train_steps: Option<String>,
    pub(crate) eval_interval: Option<String>,
    pub(crate) generate_tokens: Option<String>,
    pub(crate) minigpt_grad_clip_norm: Option<String>,
    pub(crate) prefetch_batches: Option<String>,
}

impl RuntimeEnv {
    pub(crate) fn from_process_env() -> Self {
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

#[cfg(test)]
pub(crate) fn parse_runtime_config<I, S>(
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

pub(crate) fn parse_runtime_config_with_checkpoint<I, S>(
    args: I,
    env: RuntimeEnv,
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

    let raw_input = arg_input
        .or(env.input.as_deref())
        .unwrap_or(DEFAULT_INPUT_PATH);
    let input_source = InputSource::parse(raw_input)
        .with_context(|| format!("invalid --input value '{raw_input}'"))?;
    let input_path = match &input_source {
        InputSource::Local(path) => path.clone(),
        InputSource::HuggingFace { .. } => PathBuf::from(input_source.display()),
    };

    Ok(RuntimeConfig {
        backend,
        model: parse_model_name(arg_model.or(env.model.as_deref()).unwrap_or("minigpt"))?,
        input_path,
        input_source,
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

    fn parse_args(args: &[&str]) -> Result<RuntimeConfig> {
        parse_args_with_env(args, RuntimeEnv::default())
    }

    fn parse_args_with_env(args: &[&str], env: RuntimeEnv) -> Result<RuntimeConfig> {
        parse_runtime_config_with_checkpoint(args.iter().copied(), env)
    }

    fn expect_parse_error(args: &[&str], expected: &str) {
        let err = parse_args(args).expect_err("runtime config parsing should fail");
        assert!(
            err.to_string().contains(expected),
            "expected error to contain '{expected}', got '{err}'"
        );
    }

    #[test]
    fn rejects_invalid_hyperparameter_invariants_from_cli_overrides() {
        let cases = [
            (
                &["--block-size", "0"][..],
                "block_size must be greater than zero",
            ),
            (
                &["--batch-size", "0"][..],
                "batch_size must be greater than zero",
            ),
            (
                &["--embed-dim", "0"][..],
                "embed_dim must be greater than zero",
            ),
            (
                &["--num-heads", "0"][..],
                "num_heads must be greater than zero",
            ),
            (
                &["--num-layers", "0"][..],
                "num_layers must be greater than zero",
            ),
            (
                &["--train-steps", "0"][..],
                "train_steps must be greater than zero",
            ),
            (
                &["--generate-tokens", "0"][..],
                "generate_tokens must be greater than zero",
            ),
            (&["--dropout", "1.0"][..], "dropout must be >= 0 and < 1"),
            (&["--dropout", "-0.1"][..], "dropout must be >= 0 and < 1"),
            (
                &["--learning-rate", "0"][..],
                "learning_rate must be greater than zero",
            ),
            (
                &["--learning-rate", "-0.01"][..],
                "learning_rate must be greater than zero",
            ),
        ];

        for (args, expected) in cases {
            expect_parse_error(args, expected);
        }
    }

    #[test]
    fn rejects_invalid_hyperparameter_invariants_from_env() {
        let cases: [(&str, fn(&mut RuntimeEnv), &str); 9] = [
            (
                "block_size",
                |env| env.block_size = Some("0".to_string()),
                "block_size must be greater than zero",
            ),
            (
                "batch_size",
                |env| env.batch_size = Some("0".to_string()),
                "batch_size must be greater than zero",
            ),
            (
                "embed_dim",
                |env| env.embed_dim = Some("0".to_string()),
                "embed_dim must be greater than zero",
            ),
            (
                "num_heads",
                |env| env.num_heads = Some("0".to_string()),
                "num_heads must be greater than zero",
            ),
            (
                "num_layers",
                |env| env.num_layers = Some("0".to_string()),
                "num_layers must be greater than zero",
            ),
            (
                "train_steps",
                |env| env.train_steps = Some("0".to_string()),
                "train_steps must be greater than zero",
            ),
            (
                "generate_tokens",
                |env| env.generate_tokens = Some("0".to_string()),
                "generate_tokens must be greater than zero",
            ),
            (
                "dropout",
                |env| env.dropout = Some("1.0".to_string()),
                "dropout must be >= 0 and < 1",
            ),
            (
                "learning_rate",
                |env| env.learning_rate = Some("0".to_string()),
                "learning_rate must be greater than zero",
            ),
        ];

        for (name, set_env, expected) in cases {
            let mut env = RuntimeEnv::default();
            set_env(&mut env);

            let err = match parse_args_with_env(&[], env) {
                Ok(_) => panic!("{name} env override should fail"),
                Err(err) => err,
            };
            assert!(
                err.to_string().contains(expected),
                "expected {name} error to contain '{expected}', got '{err}'"
            );
        }
    }

    #[test]
    fn rejects_invalid_numeric_parse_from_env_and_cli_overrides() {
        let mut env = RuntimeEnv {
            block_size: Some("many".to_string()),
            ..RuntimeEnv::default()
        };
        let err = parse_args_with_env(&[], env.clone())
            .expect_err("invalid env hyperparameter value should fail");
        assert!(
            err.to_string()
                .contains("invalid RUSTY_GPT_BLOCK_SIZE value: many")
        );

        env.block_size = None;
        env.dropout = Some("often".to_string());
        let err = parse_args_with_env(&[], env)
            .expect_err("invalid env float hyperparameter value should fail");
        assert!(
            err.to_string()
                .contains("invalid RUSTY_GPT_DROPOUT value: often")
        );

        expect_parse_error(
            &["--block-size", "many"],
            "invalid --block-size value: many",
        );
        expect_parse_error(&["--dropout", "often"], "invalid --dropout value: often");
    }

    #[test]
    fn rejects_missing_parser_owned_flag_values() {
        let cases = [
            (&["--server-addr"][..], "--server-addr requires a value"),
            (&["--log-format"][..], "--log-format requires a value"),
            (
                &["--benchmark-prompt-lens"][..],
                "--benchmark-prompt-lens requires a comma-separated list",
            ),
            (
                &["--benchmark-gen-lens"][..],
                "--benchmark-gen-lens requires a comma-separated list",
            ),
            (
                &["--benchmark-warmups"][..],
                "--benchmark-warmups requires an integer",
            ),
            (
                &["--benchmark-iterations"][..],
                "--benchmark-iterations requires an integer",
            ),
            (&["--block-size"][..], "--block-size requires a value"),
            (&["--dropout"][..], "--dropout requires a value"),
            (&["--learning-rate"][..], "--learning-rate requires a value"),
        ];

        for (args, expected) in cases {
            expect_parse_error(args, expected);
        }
    }

    #[test]
    fn rejects_invalid_parser_owned_flag_values() {
        let cases = [
            (
                &["--server-addr", "localhost"][..],
                "invalid server address 'localhost'",
            ),
            (&["--log-format", "yaml"][..], "unsupported log format"),
            (
                &["--benchmark-prompt-lens", "1,two"][..],
                "invalid benchmark prompt lengths '1,two'",
            ),
            (
                &["--benchmark-gen-lens", "0"][..],
                "invalid benchmark generation lengths '0'",
            ),
            (
                &["--benchmark-warmups", "many"][..],
                "invalid benchmark warmups 'many'",
            ),
            (
                &["--benchmark-iterations", "many"][..],
                "invalid benchmark iterations 'many'",
            ),
            (
                &["--benchmark-iterations", "0"][..],
                "benchmark iterations must be greater than zero",
            ),
            (
                &["--block-size", "many"][..],
                "invalid --block-size value: many",
            ),
            (
                &["--num-heads", "many"][..],
                "invalid --num-heads value: many",
            ),
            (
                &["--dropout", "often"][..],
                "invalid --dropout value: often",
            ),
        ];

        for (args, expected) in cases {
            expect_parse_error(args, expected);
        }
    }

    #[test]
    fn serve_can_be_selected_from_args() {
        let config = parse_args(&["--serve"]).unwrap();

        assert!(config.serve);
    }

    #[test]
    fn server_addr_can_be_selected_from_env() {
        let config = parse_args_with_env(
            &[],
            RuntimeEnv {
                server_addr: Some("127.0.0.1:9000".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .unwrap();

        assert_eq!(
            "127.0.0.1:9000".parse::<SocketAddr>().unwrap(),
            config.server_addr
        );
    }

    #[test]
    fn server_addr_cli_override_takes_precedence_over_env() {
        let config = parse_args_with_env(
            &["--server-addr", "127.0.0.1:9001"],
            RuntimeEnv {
                server_addr: Some("127.0.0.1:9000".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .unwrap();

        assert_eq!(
            "127.0.0.1:9001".parse::<SocketAddr>().unwrap(),
            config.server_addr
        );
    }
}
