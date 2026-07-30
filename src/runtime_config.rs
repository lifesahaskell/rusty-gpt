use crate::runtime_assets::DEFAULT_CHECKPOINT_DIR;
use anyhow::{Context, Result, bail};
use clap::Parser;
use clap::error::ErrorKind;
use rusty_gpt::loader::InputSource;
use rusty_gpt::loader::data::SamplingPolicy;
use rusty_gpt::model::LearningRateSchedule;
use rusty_gpt::observability::LogFormat;
use rusty_gpt::utils::{BenchmarkConfig, parse_usize_list};
use std::env;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

pub(crate) const DEFAULT_INPUT_PATH: &str = "data/input.txt";
pub(crate) const DEFAULT_MINIGPT_CHECKPOINT_PATH: &str = "checkpoints/mini_gpt";
pub(crate) const DEFAULT_SERVER_ADDR: &str = "127.0.0.1:8787";
pub(crate) const DEFAULT_MAX_PROMPT_BYTES: usize = 8192;
pub(crate) const DEFAULT_MAX_OUTPUT_TOKENS: usize = 512;
pub(crate) const DEFAULT_RATE_LIMIT_RPS: usize = 5;
pub(crate) const DEFAULT_RATE_LIMIT_BURST: usize = 10;
const LOG_FORMAT_ENV: &str = "RUSTY_GPT_LOG_FORMAT";
const MAX_PROMPT_BYTES_ENV: &str = "RUSTY_GPT_MAX_PROMPT_BYTES";
const MAX_OUTPUT_TOKENS_ENV: &str = "RUSTY_GPT_MAX_OUTPUT_TOKENS";
const RATE_LIMIT_RPS_ENV: &str = "RUSTY_GPT_RATE_LIMIT_RPS";
const RATE_LIMIT_BURST_ENV: &str = "RUSTY_GPT_RATE_LIMIT_BURST";
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
pub(crate) const LR_WARMUP_STEPS: usize = 0;
pub(crate) const TRAIN_STEPS: usize = 1000;
pub(crate) const EVAL_INTERVAL: usize = 100;
pub(crate) const GENERATE_TOKENS: usize = 80;
pub(crate) const MINIGPT_GRAD_CLIP_NORM: f32 = 1.0;
pub(crate) const MOE_EXPERTS: usize = 4;
pub(crate) const MOE_TOP_K: usize = 2;
pub(crate) const MOE_AUX_LOSS_WEIGHT: f64 = 0.01;
pub(crate) const PREFETCH_BATCHES: usize = 2;
/// Default cadence (in training steps) for mid-run MiniGPT checkpoints.
/// `0` disables periodic saves, preserving the historical behaviour of
/// saving only at the end of `train_steps`.
pub(crate) const CHECKPOINT_INTERVAL: usize = 0;
/// Default retention window for periodic checkpoints: keep the most recent
/// K snapshots, prune the rest. The final end-of-run save and any SIGINT
/// `interrupted-step-*` save are never pruned.
pub(crate) const CHECKPOINT_KEEP: usize = 3;

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
    pub(crate) learning_rate_schedule: LearningRateSchedule,
    pub(crate) lr_warmup_steps: usize,
    pub(crate) sampling_policy: SamplingPolicy,
    pub(crate) train_steps: usize,
    pub(crate) eval_interval: usize,
    pub(crate) generate_tokens: usize,
    pub(crate) minigpt_grad_clip_norm: f32,
    pub(crate) moe_experts: usize,
    pub(crate) moe_top_k: usize,
    pub(crate) moe_aux_loss_weight: f64,
    pub(crate) prefetch_batches: usize,
    pub(crate) checkpoint_interval: usize,
    pub(crate) checkpoint_keep: usize,
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
            learning_rate_schedule: LearningRateSchedule::Constant,
            lr_warmup_steps: LR_WARMUP_STEPS,
            sampling_policy: SamplingPolicy::RandomWindow,
            train_steps: TRAIN_STEPS,
            eval_interval: EVAL_INTERVAL,
            generate_tokens: GENERATE_TOKENS,
            minigpt_grad_clip_norm: MINIGPT_GRAD_CLIP_NORM,
            moe_experts: MOE_EXPERTS,
            moe_top_k: MOE_TOP_K,
            moe_aux_loss_weight: MOE_AUX_LOSS_WEIGHT,
            prefetch_batches: PREFETCH_BATCHES,
            checkpoint_interval: CHECKPOINT_INTERVAL,
            checkpoint_keep: CHECKPOINT_KEEP,
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
        if let Some(value) = env.learning_rate_schedule.as_deref() {
            hyperparameters.learning_rate_schedule = parse_lr_schedule(value)?;
        }
        apply_optional_override(
            "RUSTY_GPT_LR_WARMUP_STEPS",
            env.lr_warmup_steps.as_deref(),
            &mut hyperparameters.lr_warmup_steps,
        )?;
        if let Some(value) = env.sampling_policy.as_deref() {
            hyperparameters.sampling_policy = parse_sampling_policy(value)?;
        }
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
            "RUSTY_GPT_MOE_EXPERTS",
            env.moe_experts.as_deref(),
            &mut hyperparameters.moe_experts,
        )?;
        apply_optional_override(
            "RUSTY_GPT_MOE_TOP_K",
            env.moe_top_k.as_deref(),
            &mut hyperparameters.moe_top_k,
        )?;
        apply_optional_override(
            "RUSTY_GPT_MOE_AUX_LOSS_WEIGHT",
            env.moe_aux_loss_weight.as_deref(),
            &mut hyperparameters.moe_aux_loss_weight,
        )?;
        apply_optional_override(
            "RUSTY_GPT_PREFETCH_BATCHES",
            env.prefetch_batches.as_deref(),
            &mut hyperparameters.prefetch_batches,
        )?;
        apply_optional_override(
            "RUSTY_GPT_CHECKPOINT_INTERVAL",
            env.checkpoint_interval.as_deref(),
            &mut hyperparameters.checkpoint_interval,
        )?;
        apply_optional_override(
            "RUSTY_GPT_CHECKPOINT_KEEP",
            env.checkpoint_keep.as_deref(),
            &mut hyperparameters.checkpoint_keep,
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
        if self.lr_warmup_steps > self.train_steps {
            bail!("lr_warmup_steps must be <= train_steps");
        }
        if self.generate_tokens == 0 {
            bail!("generate_tokens must be greater than zero");
        }
        if self.minigpt_grad_clip_norm <= 0.0 {
            bail!("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM must be greater than zero");
        }
        if self.moe_experts == 0 {
            bail!("moe_experts must be greater than zero");
        }
        if self.moe_top_k == 0 {
            bail!("moe_top_k must be greater than zero");
        }
        if self.moe_top_k > self.moe_experts {
            bail!("moe_top_k must be <= moe_experts");
        }
        if self.moe_aux_loss_weight < 0.0 {
            bail!("moe_aux_loss_weight must be >= 0");
        }
        if self.checkpoint_interval != 0 && self.checkpoint_keep == 0 {
            bail!(
                "checkpoint_keep must be greater than zero when checkpoint_interval is non-zero (use --checkpoint-interval 0 to disable periodic checkpoints entirely)"
            );
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
    learning_rate_schedule: Option<LearningRateSchedule>,
    lr_warmup_steps: Option<usize>,
    sampling_policy: Option<SamplingPolicy>,
    train_steps: Option<usize>,
    eval_interval: Option<usize>,
    generate_tokens: Option<usize>,
    minigpt_grad_clip_norm: Option<f32>,
    moe_experts: Option<usize>,
    moe_top_k: Option<usize>,
    moe_aux_loss_weight: Option<f64>,
    prefetch_batches: Option<usize>,
    checkpoint_interval: Option<usize>,
    checkpoint_keep: Option<usize>,
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
        if let Some(value) = self.learning_rate_schedule {
            hyperparameters.learning_rate_schedule = value;
        }
        if let Some(value) = self.lr_warmup_steps {
            hyperparameters.lr_warmup_steps = value;
        }
        if let Some(value) = self.sampling_policy {
            hyperparameters.sampling_policy = value;
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
        if let Some(value) = self.moe_experts {
            hyperparameters.moe_experts = value;
        }
        if let Some(value) = self.moe_top_k {
            hyperparameters.moe_top_k = value;
        }
        if let Some(value) = self.moe_aux_loss_weight {
            hyperparameters.moe_aux_loss_weight = value;
        }
        if let Some(value) = self.prefetch_batches {
            hyperparameters.prefetch_batches = value;
        }
        if let Some(value) = self.checkpoint_interval {
            hyperparameters.checkpoint_interval = value;
        }
        if let Some(value) = self.checkpoint_keep {
            hyperparameters.checkpoint_keep = value;
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
    MoeGpt,
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
                ModelChoice::MoeGpt,
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
            ModelChoice::MoeGpt => "moe-gpt",
            ModelChoice::Compare => "compare",
        }
    }

    pub(crate) fn includes_generation_model(self) -> bool {
        matches!(
            self,
            ModelChoice::MiniGpt | ModelChoice::MoeGpt | ModelChoice::Compare
        )
    }

    pub(crate) fn uses_bpe_tokenizer(self) -> bool {
        self.includes_generation_model()
    }

    pub(crate) fn is_checkpoint_backed(self) -> bool {
        matches!(self, ModelChoice::MiniGpt | ModelChoice::MoeGpt)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RuntimeConfig {
    pub(crate) backend: BackendChoice,
    pub(crate) model: ModelChoice,
    pub(crate) input_path: PathBuf,
    pub(crate) input_source: InputSource,
    pub(crate) checkpoint_path: PathBuf,
    /// Checkpoint to resume MiniGPT training from (`--resume-from`). Confined
    /// to `checkpoints/` exactly like `--checkpoint`. `None` for a fresh run.
    pub(crate) resume_from: Option<PathBuf>,
    pub(crate) hyperparameters: Hyperparameters,
    pub(crate) interactive: bool,
    pub(crate) benchmark_generation: bool,
    pub(crate) load_checkpoint: bool,
    pub(crate) load_latest_checkpoint: bool,
    pub(crate) serve: bool,
    pub(crate) server_addr: SocketAddr,
    pub(crate) max_prompt_bytes: usize,
    pub(crate) max_output_tokens: usize,
    pub(crate) rate_limit_rps: usize,
    pub(crate) rate_limit_burst: usize,
    pub(crate) log_format: LogFormat,
    pub(crate) benchmark_config: BenchmarkConfig,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct RuntimeEnv {
    pub(crate) backend: Option<String>,
    pub(crate) input: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) checkpoint: Option<String>,
    pub(crate) resume_from: Option<String>,
    pub(crate) server_addr: Option<String>,
    pub(crate) max_prompt_bytes: Option<String>,
    pub(crate) max_output_tokens: Option<String>,
    pub(crate) rate_limit_rps: Option<String>,
    pub(crate) rate_limit_burst: Option<String>,
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
    pub(crate) learning_rate_schedule: Option<String>,
    pub(crate) lr_warmup_steps: Option<String>,
    pub(crate) sampling_policy: Option<String>,
    pub(crate) train_steps: Option<String>,
    pub(crate) eval_interval: Option<String>,
    pub(crate) generate_tokens: Option<String>,
    pub(crate) minigpt_grad_clip_norm: Option<String>,
    pub(crate) moe_experts: Option<String>,
    pub(crate) moe_top_k: Option<String>,
    pub(crate) moe_aux_loss_weight: Option<String>,
    pub(crate) prefetch_batches: Option<String>,
    pub(crate) checkpoint_interval: Option<String>,
    pub(crate) checkpoint_keep: Option<String>,
}

impl RuntimeEnv {
    pub(crate) fn from_process_env() -> Self {
        Self {
            backend: env::var("RUSTY_GPT_BACKEND").ok(),
            input: env::var("RUSTY_GPT_INPUT").ok(),
            model: env::var("RUSTY_GPT_MODEL").ok(),
            checkpoint: env::var("RUSTY_GPT_MINIGPT_CHECKPOINT").ok(),
            resume_from: env::var("RUSTY_GPT_RESUME_FROM").ok(),
            server_addr: env::var("RUSTY_GPT_SERVER_ADDR").ok(),
            max_prompt_bytes: env::var(MAX_PROMPT_BYTES_ENV).ok(),
            max_output_tokens: env::var(MAX_OUTPUT_TOKENS_ENV).ok(),
            rate_limit_rps: env::var(RATE_LIMIT_RPS_ENV).ok(),
            rate_limit_burst: env::var(RATE_LIMIT_BURST_ENV).ok(),
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
            learning_rate_schedule: env::var("RUSTY_GPT_LR_SCHEDULE").ok(),
            lr_warmup_steps: env::var("RUSTY_GPT_LR_WARMUP_STEPS").ok(),
            sampling_policy: env::var("RUSTY_GPT_SAMPLING_POLICY").ok(),
            train_steps: env::var("RUSTY_GPT_TRAIN_STEPS").ok(),
            eval_interval: env::var("RUSTY_GPT_EVAL_INTERVAL").ok(),
            generate_tokens: env::var("RUSTY_GPT_GENERATE_TOKENS").ok(),
            minigpt_grad_clip_norm: env::var("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM").ok(),
            moe_experts: env::var("RUSTY_GPT_MOE_EXPERTS").ok(),
            moe_top_k: env::var("RUSTY_GPT_MOE_TOP_K").ok(),
            moe_aux_loss_weight: env::var("RUSTY_GPT_MOE_AUX_LOSS_WEIGHT").ok(),
            prefetch_batches: env::var("RUSTY_GPT_PREFETCH_BATCHES").ok(),
            checkpoint_interval: env::var("RUSTY_GPT_CHECKPOINT_INTERVAL").ok(),
            checkpoint_keep: env::var("RUSTY_GPT_CHECKPOINT_KEEP").ok(),
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

/// Command-line surface. Every field is `Option`/`bool` with no clap default,
/// because defaults are resolved *downstream* in the CLI → env → default chain
/// that `RuntimeConfig` and `Hyperparameters` already implement. A clap
/// `default_value` here would make "not passed" indistinguishable from
/// "passed the default", and silently beat the `RUSTY_GPT_*` env vars.
///
/// Env vars are deliberately **not** wired through clap's `env` feature. They
/// are read into [`RuntimeEnv`] instead so tests can inject them as a value
/// rather than mutating the process environment — see the `ENV_MUTATION_LOCK`
/// note in `main.rs`. Env names appear in the help text only.
#[derive(Parser, Debug)]
#[command(
    name = "rusty-gpt",
    version,
    // Without this, clap reads `--dropout -0.5` as the unknown flag `-0` and
    // rejects it before `Hyperparameters::validate` ever sees the value. The
    // domain errors ("dropout must be >= 0 and < 1", "learning rate must be
    // greater than zero") are the ones users need, so negative numbers have to
    // parse as values.
    allow_negative_numbers = true,
    about = "Train and serve a GPT from scratch on the Burn framework",
    long_about = "Every flag below has a RUSTY_GPT_* environment variable counterpart \
                  (named in each flag's help). Both forms work; the flag wins when both are set. \
                  See docs/configuration.md for the authoritative reference."
)]
struct Cli {
    /// Compute backend: cpu or cuda [env: RUSTY_GPT_BACKEND] [default: cpu]
    #[arg(long)]
    backend: Option<String>,
    /// Corpus path or hf:// dataset URI [env: RUSTY_GPT_INPUT] [default: data/input.txt]
    #[arg(long)]
    input: Option<String>,
    /// trivial, single-attention, multi-attention, minigpt, moe-gpt, compare [env: RUSTY_GPT_MODEL]
    #[arg(long)]
    model: Option<String>,
    /// Checkpoint path without .mpk, confined to checkpoints/ [env: RUSTY_GPT_MINIGPT_CHECKPOINT]
    #[arg(long)]
    checkpoint: Option<String>,
    /// Resume training from this checkpoint; requires minigpt or moe-gpt [env: RUSTY_GPT_RESUME_FROM]
    #[arg(long)]
    resume_from: Option<String>,
    /// HTTP server bind address [env: RUSTY_GPT_SERVER_ADDR] [default: 127.0.0.1:8787]
    #[arg(long)]
    server_addr: Option<String>,
    /// plain or json; defaults to plain on cpu, json on cuda [env: RUSTY_GPT_LOG_FORMAT]
    #[arg(long)]
    log_format: Option<String>,

    /// POST /api/generate prompt byte cap [env: RUSTY_GPT_MAX_PROMPT_BYTES] [default: 8192]
    #[arg(long)]
    max_prompt_bytes: Option<usize>,
    /// POST /api/generate max_tokens cap [env: RUSTY_GPT_MAX_OUTPUT_TOKENS] [default: 512]
    #[arg(long)]
    max_output_tokens: Option<usize>,
    /// Generate refill rate; 0 disables rate limiting [env: RUSTY_GPT_RATE_LIMIT_RPS] [default: 5]
    #[arg(long)]
    rate_limit_rps: Option<usize>,
    /// Generate burst capacity [env: RUSTY_GPT_RATE_LIMIT_BURST] [default: 10]
    #[arg(long)]
    rate_limit_burst: Option<usize>,

    /// Comma-separated prompt lengths [env: RUSTY_GPT_BENCHMARK_PROMPT_LENS]
    #[arg(long)]
    benchmark_prompt_lens: Option<String>,
    /// Comma-separated generation lengths [env: RUSTY_GPT_BENCHMARK_GEN_LENS]
    #[arg(long)]
    benchmark_gen_lens: Option<String>,
    /// Warmup iterations before timing [env: RUSTY_GPT_BENCHMARK_WARMUPS]
    #[arg(long)]
    benchmark_warmups: Option<String>,
    /// Timed iterations per case [env: RUSTY_GPT_BENCHMARK_ITERATIONS]
    #[arg(long)]
    benchmark_iterations: Option<String>,

    /// Context length [env: RUSTY_GPT_BLOCK_SIZE] [default: 128]
    #[arg(long)]
    block_size: Option<usize>,
    /// Training batch size [env: RUSTY_GPT_BATCH_SIZE] [default: 32]
    #[arg(long)]
    batch_size: Option<usize>,
    /// Model width; must divide by num-heads [env: RUSTY_GPT_EMBED_DIM] [default: 128]
    #[arg(long)]
    embed_dim: Option<usize>,
    /// Attention heads [env: RUSTY_GPT_NUM_HEADS] [default: 4]
    #[arg(long)]
    num_heads: Option<usize>,
    /// Transformer block count [env: RUSTY_GPT_NUM_LAYERS] [default: 4]
    #[arg(long)]
    num_layers: Option<usize>,
    /// Dropout probability, >= 0 and < 1 [env: RUSTY_GPT_DROPOUT]
    #[arg(long)]
    dropout: Option<f64>,
    /// Base optimizer learning rate [env: RUSTY_GPT_LEARNING_RATE] [default: 1e-4]
    #[arg(long)]
    learning_rate: Option<f64>,
    /// constant, warmup-cosine, or warmup-linear [env: RUSTY_GPT_LR_SCHEDULE]
    #[arg(long)]
    lr_schedule: Option<String>,
    /// Warmup length; must be <= train-steps [env: RUSTY_GPT_LR_WARMUP_STEPS]
    #[arg(long)]
    lr_warmup_steps: Option<usize>,
    /// random-window, sequential, or shuffled-chunks [env: RUSTY_GPT_SAMPLING_POLICY]
    #[arg(long)]
    sampling_policy: Option<String>,
    /// Training steps [env: RUSTY_GPT_TRAIN_STEPS] [default: 1000]
    #[arg(long)]
    train_steps: Option<usize>,
    /// Eval cadence; 0 logs only the final step [env: RUSTY_GPT_EVAL_INTERVAL] [default: 100]
    #[arg(long)]
    eval_interval: Option<usize>,
    /// Interactive generation token count [env: RUSTY_GPT_GENERATE_TOKENS] [default: 80]
    #[arg(long)]
    generate_tokens: Option<usize>,
    /// Gradient clipping norm [env: RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM] [default: 1.0]
    #[arg(long)]
    grad_clip_norm: Option<f32>,
    /// MoE experts per block [env: RUSTY_GPT_MOE_EXPERTS] [default: 4]
    #[arg(long)]
    moe_experts: Option<usize>,
    /// Experts selected per token [env: RUSTY_GPT_MOE_TOP_K] [default: 2]
    #[arg(long)]
    moe_top_k: Option<usize>,
    /// Load-balancing aux loss weight [env: RUSTY_GPT_MOE_AUX_LOSS_WEIGHT] [default: 0.01]
    #[arg(long)]
    moe_aux_loss_weight: Option<f64>,
    /// CPU batch prefetch depth [env: RUSTY_GPT_PREFETCH_BATCHES] [default: 2]
    #[arg(long)]
    prefetch_batches: Option<usize>,
    /// Save a snapshot every N steps; 0 disables [env: RUSTY_GPT_CHECKPOINT_INTERVAL] [default: 0]
    #[arg(long)]
    checkpoint_interval: Option<usize>,
    /// Periodic snapshot retention window [env: RUSTY_GPT_CHECKPOINT_KEEP] [default: 3]
    #[arg(long)]
    checkpoint_keep: Option<usize>,

    /// Chat with a loaded checkpoint; requires --backend cpu
    #[arg(long)]
    interactive_generate: bool,
    /// Run naive-vs-cached generation benchmarks
    #[arg(long)]
    benchmark_generation: bool,
    /// Load weights from --checkpoint before serving
    #[arg(long)]
    load_checkpoint: bool,
    /// Load the newest .mpk in checkpoints/ before serving
    #[arg(long)]
    load_latest_checkpoint: bool,
    /// Start the HTTP API instead of the demo/training run
    #[arg(long)]
    serve: bool,
}

pub(crate) fn parse_runtime_config_with_checkpoint<I, S>(
    args: I,
    env: RuntimeEnv,
) -> Result<RuntimeConfig>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    // clap expects argv[0]; every caller passes bare flags (main.rs uses
    // `env::args().skip(1)`), so synthesize the program name here.
    let cli = match Cli::try_parse_from(
        std::iter::once("rusty-gpt".to_string())
            .chain(args.into_iter().map(|arg| arg.as_ref().to_string())),
    ) {
        Ok(cli) => cli,
        // `try_parse_from` reports `--help` and `--version` as errors. They are
        // not: they belong on stdout with exit code 0, which `Error::exit` does.
        // Everything else becomes an ordinary anyhow error so the existing
        // message assertions and `main`'s error path keep working.
        Err(err)
            if matches!(
                err.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) =>
        {
            err.exit()
        }
        Err(err) => return Err(err.into()),
    };

    let arg_backend = cli.backend.as_deref();
    let arg_input = cli.input.as_deref();
    let arg_model = cli.model.as_deref();
    let arg_checkpoint = cli.checkpoint.as_deref();
    let arg_resume_from = cli.resume_from.as_deref();
    let arg_server_addr = cli.server_addr.as_deref();
    let arg_log_format = cli.log_format.as_deref();
    let arg_max_prompt_bytes = cli.max_prompt_bytes;
    let arg_max_output_tokens = cli.max_output_tokens;
    let arg_rate_limit_rps = cli.rate_limit_rps;
    let arg_rate_limit_burst = cli.rate_limit_burst;
    let arg_benchmark_prompt_lens = cli.benchmark_prompt_lens.as_deref();
    let arg_benchmark_gen_lens = cli.benchmark_gen_lens.as_deref();
    let arg_benchmark_warmups = cli.benchmark_warmups.as_deref();
    let arg_benchmark_iterations = cli.benchmark_iterations.as_deref();
    let interactive = cli.interactive_generate;
    let benchmark_generation = cli.benchmark_generation;
    let load_checkpoint = cli.load_checkpoint;
    let load_latest_checkpoint = cli.load_latest_checkpoint;
    let serve = cli.serve;

    // `--lr-schedule` and `--sampling-policy` keep their hand-written parsers
    // rather than a clap `value_parser`, so their error text stays identical
    // and `LearningRateSchedule`/`SamplingPolicy` need no clap derive.
    let hyperparameter_overrides = HyperparameterOverrides {
        block_size: cli.block_size,
        batch_size: cli.batch_size,
        embed_dim: cli.embed_dim,
        num_heads: cli.num_heads,
        num_layers: cli.num_layers,
        dropout: cli.dropout,
        learning_rate: cli.learning_rate,
        learning_rate_schedule: cli
            .lr_schedule
            .as_deref()
            .map(parse_lr_schedule)
            .transpose()?,
        lr_warmup_steps: cli.lr_warmup_steps,
        sampling_policy: cli
            .sampling_policy
            .as_deref()
            .map(parse_sampling_policy)
            .transpose()?,
        train_steps: cli.train_steps,
        eval_interval: cli.eval_interval,
        generate_tokens: cli.generate_tokens,
        minigpt_grad_clip_norm: cli.grad_clip_norm,
        moe_experts: cli.moe_experts,
        moe_top_k: cli.moe_top_k,
        moe_aux_loss_weight: cli.moe_aux_loss_weight,
        prefetch_batches: cli.prefetch_batches,
        checkpoint_interval: cli.checkpoint_interval,
        checkpoint_keep: cli.checkpoint_keep,
    };

    if load_checkpoint && load_latest_checkpoint {
        bail!("--load-checkpoint and --load-latest-checkpoint are mutually exclusive");
    }

    let server_addr_text = arg_server_addr
        .or(env.server_addr.as_deref())
        .unwrap_or(DEFAULT_SERVER_ADDR);
    let server_addr = server_addr_text
        .parse()
        .with_context(|| format!("invalid server address '{server_addr_text}'"))?;
    let max_prompt_bytes = parse_config_usize(
        arg_max_prompt_bytes,
        env.max_prompt_bytes.as_deref(),
        DEFAULT_MAX_PROMPT_BYTES,
        MAX_PROMPT_BYTES_ENV,
    )?;
    let max_output_tokens = parse_config_usize(
        arg_max_output_tokens,
        env.max_output_tokens.as_deref(),
        DEFAULT_MAX_OUTPUT_TOKENS,
        MAX_OUTPUT_TOKENS_ENV,
    )?;
    let rate_limit_rps = parse_config_usize(
        arg_rate_limit_rps,
        env.rate_limit_rps.as_deref(),
        DEFAULT_RATE_LIMIT_RPS,
        RATE_LIMIT_RPS_ENV,
    )?;
    let rate_limit_burst = parse_config_usize(
        arg_rate_limit_burst,
        env.rate_limit_burst.as_deref(),
        DEFAULT_RATE_LIMIT_BURST,
        RATE_LIMIT_BURST_ENV,
    )?;
    validate_server_limits(
        max_prompt_bytes,
        max_output_tokens,
        rate_limit_rps,
        rate_limit_burst,
    )?;
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

    let checkpoint_path = match arg_checkpoint.or(env.checkpoint.as_deref()) {
        Some(path) => validate_checkpoint_path(path, Path::new(DEFAULT_CHECKPOINT_DIR))?,
        None => PathBuf::from(DEFAULT_MINIGPT_CHECKPOINT_PATH),
    };

    let model = parse_model_name(arg_model.or(env.model.as_deref()).unwrap_or("minigpt"))?;

    // `--resume-from` is confined to `checkpoints/` exactly like `--checkpoint`,
    // and is only meaningful for the checkpoint-backed MiniGPT model.
    let resume_from = match arg_resume_from.or(env.resume_from.as_deref()) {
        Some(path) => {
            if !model.is_checkpoint_backed() {
                bail!(
                    "--resume-from requires --model minigpt or moe-gpt; got --model {}",
                    model.label()
                );
            }
            Some(validate_checkpoint_path(
                path,
                Path::new(DEFAULT_CHECKPOINT_DIR),
            )?)
        }
        None => None,
    };

    Ok(RuntimeConfig {
        backend,
        model,
        input_path,
        input_source,
        checkpoint_path,
        resume_from,
        hyperparameters,
        interactive,
        benchmark_generation,
        load_checkpoint,
        load_latest_checkpoint,
        serve,
        server_addr,
        max_prompt_bytes,
        max_output_tokens,
        rate_limit_rps,
        rate_limit_burst,
        log_format,
        benchmark_config,
    })
}

fn parse_config_usize(
    arg_value: Option<usize>,
    env_value: Option<&str>,
    default: usize,
    env_name: &str,
) -> Result<usize> {
    if let Some(value) = arg_value {
        return Ok(value);
    }
    if let Some(value) = env_value {
        return value
            .parse()
            .with_context(|| format!("invalid {env_name} value: {value}"));
    }
    Ok(default)
}

fn validate_server_limits(
    max_prompt_bytes: usize,
    max_output_tokens: usize,
    rate_limit_rps: usize,
    rate_limit_burst: usize,
) -> Result<()> {
    if max_prompt_bytes == 0 {
        bail!("max_prompt_bytes must be greater than zero");
    }
    if max_output_tokens == 0 {
        bail!("max_output_tokens must be greater than zero");
    }
    if rate_limit_rps != 0 && rate_limit_burst == 0 {
        bail!("rate_limit_burst must be greater than zero when rate_limit_rps is non-zero");
    }
    Ok(())
}

pub(crate) fn validate_checkpoint_path(input: &str, root: &Path) -> Result<PathBuf> {
    let root_input = root;
    let root = root_input.canonicalize().with_context(|| {
        format!(
            "checkpoint path must be inside checkpoints/ (got: {input}, failed to resolve root: {})",
            root_input.display()
        )
    })?;
    let input_path = Path::new(input);
    let candidate = if input_path.is_absolute() || input_path.starts_with(root_input) {
        input_path.to_path_buf()
    } else {
        root_input.join(input_path)
    };
    let file_name = candidate.file_name().with_context(|| {
        format!("checkpoint path must be inside checkpoints/ (got: {input}, resolved: <none>)")
    })?;
    let parent = candidate.parent().unwrap_or(&root);
    let canonical_parent = parent.canonicalize().with_context(|| {
        format!(
            "checkpoint path must be inside checkpoints/ (got: {input}, failed to resolve parent: {})",
            parent.display()
        )
    })?;
    let resolved = canonical_parent.join(file_name);

    if !resolved.starts_with(&root) {
        bail!(
            "checkpoint path must be inside checkpoints/ (got: {input}, resolved: {})",
            resolved.display()
        );
    }

    Ok(resolved)
}

fn default_log_format(backend: BackendChoice) -> LogFormat {
    match backend {
        BackendChoice::Cpu => LogFormat::Plain,
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => LogFormat::Json,
    }
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
        "moe-gpt" | "moegpt" => Ok(ModelChoice::MoeGpt),
        "compare" => Ok(ModelChoice::Compare),
        other => bail!(
            "unsupported model '{other}'; expected trivial, single-attention, multi-attention, minigpt, moe-gpt, or compare"
        ),
    }
}

fn parse_lr_schedule(value: &str) -> Result<LearningRateSchedule> {
    match value {
        "constant" => Ok(LearningRateSchedule::Constant),
        "warmup-cosine" => Ok(LearningRateSchedule::WarmupCosine),
        "warmup-linear" => Ok(LearningRateSchedule::WarmupLinear),
        other => bail!(
            "unsupported lr schedule '{other}'; expected constant, warmup-cosine, or warmup-linear"
        ),
    }
}

fn parse_sampling_policy(value: &str) -> Result<SamplingPolicy> {
    match value {
        "random-window" => Ok(SamplingPolicy::RandomWindow),
        "sequential" => Ok(SamplingPolicy::Sequential),
        "shuffled-chunks" => Ok(SamplingPolicy::ShuffledChunks),
        other => bail!(
            "unsupported sampling policy '{other}'; expected random-window, sequential, or shuffled-chunks"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs as unix_fs;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!(
            "rusty-gpt-runtime-config-{name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn assert_checkpoint_error_contains(err: anyhow::Error) {
        assert!(
            err.to_string()
                .contains("checkpoint path must be inside checkpoints/"),
            "expected checkpoint confinement error, got '{err}'"
        );
    }

    #[test]
    fn checkpoint_validator_accepts_relative_path_inside_root() {
        let dir = unique_temp_dir("relative");
        let root = dir.join("checkpoints");
        fs::create_dir_all(root.join("nested")).unwrap();

        let path = validate_checkpoint_path("nested/mini_gpt", &root).unwrap();

        assert_eq!(root.join("nested/mini_gpt"), path);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn checkpoint_validator_accepts_absolute_path_inside_root() {
        let dir = unique_temp_dir("absolute");
        let root = dir.join("checkpoints");
        fs::create_dir_all(root.join("nested")).unwrap();
        let input = root.join("nested/mini_gpt");

        let path = validate_checkpoint_path(input.to_str().unwrap(), &root).unwrap();

        assert_eq!(input, path);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn checkpoint_validator_maps_bare_name_inside_root() {
        let dir = unique_temp_dir("bare-name");
        let root = dir.join("checkpoints");
        fs::create_dir_all(&root).unwrap();

        let path = validate_checkpoint_path("mini_gpt", &root).unwrap();

        assert_eq!(root.join("mini_gpt"), path);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn checkpoint_validator_rejects_parent_traversal_outside_root() {
        let dir = unique_temp_dir("traversal");
        let root = dir.join("checkpoints");
        fs::create_dir_all(&root).unwrap();

        let err = validate_checkpoint_path("../secret", &root)
            .expect_err("parent traversal should be rejected");

        assert_checkpoint_error_contains(err);
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_validator_rejects_symlink_parent_outside_root() {
        let dir = unique_temp_dir("symlink");
        let root = dir.join("checkpoints");
        let outside = dir.join("outside");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        unix_fs::symlink(&outside, root.join("linked-outside")).unwrap();

        let err = validate_checkpoint_path("linked-outside/mini_gpt", &root)
            .expect_err("symlink escaping the checkpoint root should be rejected");

        assert_checkpoint_error_contains(err);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn checkpoint_validator_accepts_nonexistent_file_inside_existing_parent() {
        let dir = unique_temp_dir("nonexistent-file");
        let root = dir.join("checkpoints");
        fs::create_dir_all(root.join("nested")).unwrap();

        let path = validate_checkpoint_path("nested/new_checkpoint", &root).unwrap();

        assert_eq!(root.join("nested/new_checkpoint"), path);
        assert!(!path.exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn checkpoint_env_path_is_validated() {
        fs::create_dir_all(DEFAULT_CHECKPOINT_DIR).unwrap();
        let config = parse_args_with_env(
            &[],
            RuntimeEnv {
                checkpoint: Some("mini_gpt".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .unwrap();

        assert!(config.checkpoint_path.ends_with("checkpoints/mini_gpt"));
    }

    #[test]
    fn checkpoint_cli_path_rejects_parent_traversal() {
        fs::create_dir_all(DEFAULT_CHECKPOINT_DIR).unwrap();
        let err = parse_args(&["--checkpoint", "../secret"])
            .expect_err("cli checkpoint traversal should fail");

        assert_checkpoint_error_contains(err);
    }

    #[test]
    fn checkpoint_env_path_rejects_parent_traversal() {
        fs::create_dir_all(DEFAULT_CHECKPOINT_DIR).unwrap();
        let err = parse_args_with_env(
            &[],
            RuntimeEnv {
                checkpoint: Some("../secret".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .expect_err("env checkpoint traversal should fail");

        assert_checkpoint_error_contains(err);
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
                &["--train-steps", "10", "--lr-warmup-steps", "11"][..],
                "lr_warmup_steps must be <= train_steps",
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

    /// A single env-driven invariant-violation case used by
    /// [`rejects_invalid_hyperparameter_invariants_from_env`]: a human-readable
    /// label, a mutator that injects the bad value into a [`RuntimeEnv`], and
    /// the expected substring of the resulting validation error.
    type EnvInvariantCase = (&'static str, fn(&mut RuntimeEnv), &'static str);

    #[test]
    fn rejects_invalid_hyperparameter_invariants_from_env() {
        let cases: [EnvInvariantCase; 10] = [
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
                "lr_warmup_steps",
                |env| {
                    env.train_steps = Some("10".to_string());
                    env.lr_warmup_steps = Some("11".to_string());
                },
                "lr_warmup_steps must be <= train_steps",
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
            "invalid value 'many' for '--block-size",
        );
        expect_parse_error(
            &["--dropout", "often"],
            "invalid value 'often' for '--dropout",
        );
    }

    #[test]
    fn parses_experiment_runtime_flags() {
        let config = parse_args(&[
            "--lr-schedule",
            "warmup-cosine",
            "--lr-warmup-steps",
            "5",
            "--sampling-policy",
            "shuffled-chunks",
        ])
        .unwrap();

        assert_eq!(
            LearningRateSchedule::WarmupCosine,
            config.hyperparameters.learning_rate_schedule
        );
        assert_eq!(5, config.hyperparameters.lr_warmup_steps);
        assert_eq!(
            SamplingPolicy::ShuffledChunks,
            config.hyperparameters.sampling_policy
        );
    }

    #[test]
    fn rejects_missing_parser_owned_flag_values() {
        // clap owns missing-value errors now. The behaviour under test is
        // unchanged - a flag given without its value is rejected, naming the
        // flag - only the wording moved from hand-written text to clap's.
        for flag in [
            "--server-addr",
            "--log-format",
            "--benchmark-prompt-lens",
            "--benchmark-gen-lens",
            "--benchmark-warmups",
            "--benchmark-iterations",
            "--block-size",
            "--dropout",
            "--learning-rate",
            "--lr-schedule",
            "--sampling-policy",
        ] {
            expect_parse_error(&[flag], &format!("a value is required for '{flag}"));
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
                "invalid value 'many' for '--block-size",
            ),
            (
                &["--num-heads", "many"][..],
                "invalid value 'many' for '--num-heads",
            ),
            (
                &["--dropout", "often"][..],
                "invalid value 'often' for '--dropout",
            ),
            (
                &["--lr-schedule", "banana"][..],
                "unsupported lr schedule 'banana'",
            ),
            (
                &["--sampling-policy", "banana"][..],
                "unsupported sampling policy 'banana'",
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
    fn server_limits_default_to_security_caps() {
        let config = parse_args(&[]).unwrap();

        assert_eq!(DEFAULT_MAX_PROMPT_BYTES, config.max_prompt_bytes);
        assert_eq!(DEFAULT_MAX_OUTPUT_TOKENS, config.max_output_tokens);
        assert_eq!(DEFAULT_RATE_LIMIT_RPS, config.rate_limit_rps);
        assert_eq!(DEFAULT_RATE_LIMIT_BURST, config.rate_limit_burst);
    }

    #[test]
    fn server_limits_can_be_selected_from_env() {
        let config = parse_args_with_env(
            &[],
            RuntimeEnv {
                max_prompt_bytes: Some("1024".to_string()),
                max_output_tokens: Some("64".to_string()),
                rate_limit_rps: Some("2".to_string()),
                rate_limit_burst: Some("4".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .unwrap();

        assert_eq!(1024, config.max_prompt_bytes);
        assert_eq!(64, config.max_output_tokens);
        assert_eq!(2, config.rate_limit_rps);
        assert_eq!(4, config.rate_limit_burst);
    }

    #[test]
    fn server_limit_cli_overrides_take_precedence_over_env() {
        let config = parse_args_with_env(
            &[
                "--max-prompt-bytes",
                "2048",
                "--max-output-tokens",
                "128",
                "--rate-limit-rps",
                "3",
                "--rate-limit-burst",
                "6",
            ],
            RuntimeEnv {
                max_prompt_bytes: Some("1024".to_string()),
                max_output_tokens: Some("64".to_string()),
                rate_limit_rps: Some("2".to_string()),
                rate_limit_burst: Some("4".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .unwrap();

        assert_eq!(2048, config.max_prompt_bytes);
        assert_eq!(128, config.max_output_tokens);
        assert_eq!(3, config.rate_limit_rps);
        assert_eq!(6, config.rate_limit_burst);
    }

    #[test]
    fn zero_rate_limit_rps_disables_limiter_without_requiring_burst() {
        let config = parse_args(&["--rate-limit-rps", "0", "--rate-limit-burst", "0"]).unwrap();

        assert_eq!(0, config.rate_limit_rps);
        assert_eq!(0, config.rate_limit_burst);
    }

    #[test]
    fn rejects_invalid_server_limit_invariants() {
        let cases = [
            (
                &["--max-prompt-bytes", "0"][..],
                "max_prompt_bytes must be greater than zero",
            ),
            (
                &["--max-output-tokens", "0"][..],
                "max_output_tokens must be greater than zero",
            ),
            (
                &["--rate-limit-rps", "1", "--rate-limit-burst", "0"][..],
                "rate_limit_burst must be greater than zero",
            ),
        ];

        for (args, expected) in cases {
            expect_parse_error(args, expected);
        }
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

    #[test]
    fn resume_from_defaults_to_none() {
        let config = parse_args(&[]).unwrap();

        assert_eq!(None, config.resume_from);
    }

    #[test]
    fn resume_from_arg_is_confined_to_checkpoints() {
        fs::create_dir_all(DEFAULT_CHECKPOINT_DIR).unwrap();
        let config =
            parse_args(&["--model", "minigpt", "--resume-from", "mini_gpt.step-4"]).unwrap();

        assert!(
            config
                .resume_from
                .as_ref()
                .expect("resume_from should be set")
                .ends_with("checkpoints/mini_gpt.step-4")
        );
    }

    #[test]
    fn resume_from_can_be_selected_from_env() {
        fs::create_dir_all(DEFAULT_CHECKPOINT_DIR).unwrap();
        let config = parse_args_with_env(
            &[],
            RuntimeEnv {
                resume_from: Some("mini_gpt.step-4".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .unwrap();

        assert!(
            config
                .resume_from
                .as_ref()
                .expect("resume_from should be set")
                .ends_with("checkpoints/mini_gpt.step-4")
        );
    }

    #[test]
    fn resume_from_rejects_parent_traversal() {
        fs::create_dir_all(DEFAULT_CHECKPOINT_DIR).unwrap();
        let err = parse_args(&["--model", "minigpt", "--resume-from", "../secret"])
            .expect_err("resume-from traversal should fail");

        assert_checkpoint_error_contains(err);
    }

    #[test]
    fn resume_from_rejects_non_minigpt_model_at_parse_time() {
        // Meaningless combination: only checkpoint-backed models can resume.
        expect_parse_error(
            &["--model", "trivial", "--resume-from", "mini_gpt.step-4"],
            "--resume-from requires --model minigpt",
        );
        expect_parse_error(
            &["--model", "compare", "--resume-from", "mini_gpt.step-4"],
            "--resume-from requires --model minigpt",
        );
    }
}
