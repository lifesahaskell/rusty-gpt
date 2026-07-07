use burn::module::AutodiffModule;
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};

use crate::loader::data::{BatchPrefetcher, DataLoader, TokenBatch};
use crate::observability::{EventLogger, LogFormat, RuntimeEvent};

use super::{MiniGpt, MiniGptConfig, MultiAttentionModel, SingleAttentionModel, TrivialModel};

pub type TrainingLogFormat = LogFormat;

/// Learning-rate schedule applied across a training run. `Constant` reproduces
/// the historical behaviour (the optimizer sees `base_lr` at every step);
/// `Cosine` adds an optional linear warmup followed by cosine decay toward a
/// floor. See [`learning_rate_at_step`] for the exact formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LrSchedule {
    #[default]
    Constant,
    Cosine,
}

impl LrSchedule {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Cosine => "cosine",
        }
    }
}

impl std::fmt::Display for LrSchedule {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Error returned when parsing an [`LrSchedule`] from an unrecognized string.
/// Implements [`std::error::Error`] so it satisfies the `FromStr` bound used by
/// the `RUSTY_GPT_*` env/CLI override plumbing in `runtime_config`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LrScheduleParseError(String);

impl std::fmt::Display for LrScheduleParseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "unsupported lr schedule '{}'; expected constant or cosine",
            self.0
        )
    }
}

impl std::error::Error for LrScheduleParseError {}

impl std::str::FromStr for LrSchedule {
    type Err = LrScheduleParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "constant" => Ok(Self::Constant),
            "cosine" => Ok(Self::Cosine),
            other => Err(LrScheduleParseError(other.to_string())),
        }
    }
}

/// Pure function mapping a 0-indexed training step to a learning rate.
///
/// - `Constant` always returns `base_lr` (warmup/min ignored) so a default
///   configuration is bit-for-bit identical to the pre-schedule behaviour.
/// - `Cosine`:
///   - `step < warmup_steps` ⇒ linear warmup `base_lr * step / warmup_steps`
///     (so `lr(0) == 0` and `lr(warmup_steps) == base_lr`);
///   - otherwise cosine decay `min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π·t))`
///     where `t = (step - warmup_steps) / (total_steps - warmup_steps)` clamped
///     to `[0, 1]`, so `lr(total_steps) == min_lr`.
pub fn learning_rate_at_step(
    step: usize,
    base_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    schedule: LrSchedule,
) -> f64 {
    match schedule {
        LrSchedule::Constant => base_lr,
        LrSchedule::Cosine => {
            if warmup_steps > 0 && step < warmup_steps {
                return base_lr * (step as f64) / (warmup_steps as f64);
            }
            let decay_steps = total_steps.saturating_sub(warmup_steps);
            if decay_steps == 0 {
                return base_lr;
            }
            let progress = ((step - warmup_steps) as f64 / decay_steps as f64).clamp(0.0, 1.0);
            min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }
}

#[derive(Clone)]
pub struct TrainingLogContext {
    pub backend: &'static str,
    pub model: &'static str,
    pub logger: EventLogger,
}

impl TrainingLogContext {
    pub fn plain(model: &'static str) -> Self {
        Self {
            backend: "cpu",
            model,
            logger: EventLogger::stdout(LogFormat::Plain),
        }
    }
}

#[derive(Clone)]
pub struct TrainingParams {
    pub learning_rate: f64,
    pub steps: usize,
    pub eval_interval: usize,
    pub prefetch_batches: usize,
    pub grad_clipping: Option<GradientClippingConfig>,
    pub log_context: TrainingLogContext,
    /// Cadence (in steps) at which the optional periodic-save callback fires.
    /// `0` disables periodic saves; the callback is then never invoked
    /// regardless of what it does. Only MiniGPT wires a real callback —
    /// the smaller teaching variants always pass `0` plus a no-op closure.
    pub periodic_checkpoint_interval: usize,
    /// Learning-rate schedule. Defaults to [`LrSchedule::Constant`], which
    /// yields `learning_rate` at every step (behaviour-neutral).
    pub lr_schedule: LrSchedule,
    /// Linear-warmup length in steps. `0` disables warmup. Only consulted by
    /// the `Cosine` schedule; ignored under `Constant`.
    pub warmup_steps: usize,
    /// Cosine-decay floor. Only consulted by the `Cosine` schedule.
    pub min_learning_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingMetrics {
    pub final_value_loss: f64,
    pub final_perplexity: f64,
}

pub struct TrainingOutcome<M> {
    pub model: M,
    pub metrics: TrainingMetrics,
    /// Number of training steps actually completed before the loop returned.
    /// Equal to `params.steps` on a clean finish; less when a signal interrupt
    /// broke the loop early.
    pub steps_completed: usize,
    /// `true` when the loop stopped early because [`crate::runtime_signals`]
    /// observed a SIGINT/SIGTERM. The caller is expected to save a partial
    /// checkpoint and exit with [`crate::runtime_signals::INTERRUPTED_EXIT_CODE`].
    pub interrupted: bool,
}

impl TrainingParams {
    pub fn new(
        learning_rate: f64,
        steps: usize,
        eval_interval: usize,
        log_context: TrainingLogContext,
    ) -> Self {
        Self {
            learning_rate,
            steps,
            eval_interval,
            prefetch_batches: 0,
            grad_clipping: None,
            log_context,
            periodic_checkpoint_interval: 0,
            lr_schedule: LrSchedule::Constant,
            warmup_steps: 0,
            min_learning_rate: 0.0,
        }
    }

    pub fn with_grad_clip_norm(mut self, norm: f32) -> Self {
        self.grad_clipping = Some(GradientClippingConfig::Norm(norm));
        self
    }

    /// Configure the learning-rate schedule. `warmup_steps` and
    /// `min_learning_rate` only take effect under [`LrSchedule::Cosine`];
    /// [`LrSchedule::Constant`] ignores them and keeps `learning_rate` fixed.
    pub fn with_lr_schedule(
        mut self,
        schedule: LrSchedule,
        warmup_steps: usize,
        min_learning_rate: f64,
    ) -> Self {
        self.lr_schedule = schedule;
        self.warmup_steps = warmup_steps;
        self.min_learning_rate = min_learning_rate;
        self
    }

    pub fn with_prefetch_batches(mut self, batches: usize) -> Self {
        self.prefetch_batches = batches;
        self
    }

    pub fn with_periodic_checkpoint_interval(mut self, interval: usize) -> Self {
        self.periodic_checkpoint_interval = interval;
        self
    }
}

pub(super) fn should_log_training_step(step: usize, steps: usize, eval_interval: usize) -> bool {
    if eval_interval == 0 {
        return step + 1 == steps;
    }

    steps <= eval_interval || step.is_multiple_of(eval_interval) || step + 1 == steps
}

fn language_model_loss<B: Backend>(
    loss_fn: &CrossEntropyLoss<B>,
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let [batch_size, seq_len, vocab_size] = logits.shape().dims();
    loss_fn.forward(
        logits.reshape([batch_size * seq_len, vocab_size]),
        targets.reshape([batch_size * seq_len]),
    )
}

fn value_loss<B: Backend>(
    loader: &DataLoader,
    device: &B::Device,
    loss_fn: &CrossEntropyLoss<B>,
    forward: impl FnOnce(Tensor<B, 2, Int>) -> Tensor<B, 3>,
) -> Result<B::FloatElem, String> {
    let (inputs, targets) = loader.next_batch::<B>(device)?;
    Ok(language_model_loss(loss_fn, forward(inputs), targets).into_scalar())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn training_progress_log_line<B: Backend>(
    context: TrainingLogContext,
    step: usize,
    steps: usize,
    training_loss: B::FloatElem,
    value_loss: B::FloatElem,
    learning_rate: f64,
    throughput: TrainingThroughput,
) -> String {
    match context.logger.format() {
        LogFormat::Plain => {
            format!(
                "Step {step}: training loss = {training_loss:.4}, value loss = {value_loss:.4}, learning_rate={learning_rate:.6}, tokens_per_second={:.2}, steps_per_second={:.4}, step_ms_mean={:.2}",
                throughput.tokens_per_second, throughput.steps_per_second, throughput.step_ms_mean
            )
        }
        LogFormat::Json => {
            format!(
                r#"{{"event":"training_progress","backend":"{}","model":"{}","step":{},"total_steps":{},"training_loss":{:.6},"value_loss":{:.6},"learning_rate":{:.6},"elapsed_ms":{},"tokens_per_second":{:.6},"steps_per_second":{:.6},"step_ms_mean":{:.6}}}"#,
                context.backend,
                context.model,
                step,
                steps,
                training_loss,
                value_loss,
                learning_rate,
                throughput.elapsed_ms,
                throughput.tokens_per_second,
                throughput.steps_per_second,
                throughput.step_ms_mean
            )
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TrainingThroughput {
    elapsed_ms: u128,
    tokens_per_second: f64,
    steps_per_second: f64,
    step_ms_mean: f64,
}

fn perplexity(value_loss: f64) -> f64 {
    value_loss.exp()
}

impl TrainingThroughput {
    pub(super) fn from_progress(
        completed_steps: usize,
        batch_size: usize,
        block_size: usize,
        elapsed_ms: u128,
    ) -> Self {
        let elapsed_seconds = elapsed_ms.max(1) as f64 / 1000.0;
        let completed_steps = completed_steps as f64;
        let processed_tokens = completed_steps * batch_size as f64 * block_size as f64;

        Self {
            elapsed_ms,
            tokens_per_second: processed_tokens / elapsed_seconds,
            steps_per_second: completed_steps / elapsed_seconds,
            step_ms_mean: elapsed_ms as f64 / completed_steps.max(1.0),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn log_training_progress(
    context: TrainingLogContext,
    step: usize,
    steps: usize,
    training_loss: f64,
    value_loss: f64,
    learning_rate: f64,
    throughput: TrainingThroughput,
) {
    context.logger.log(RuntimeEvent::TrainingProgress {
        backend: context.backend.to_string(),
        model: context.model.to_string(),
        step,
        total_steps: steps,
        training_loss,
        value_loss,
        value_perplexity: perplexity(value_loss),
        learning_rate,
        elapsed_ms: throughput.elapsed_ms,
        tokens_per_second: throughput.tokens_per_second,
        steps_per_second: throughput.steps_per_second,
        step_ms_mean: throughput.step_ms_mean,
    });
}

enum TrainingBatchSource<'a> {
    Direct(&'a DataLoader),
    Prefetch(BatchPrefetcher),
}

impl<'a> TrainingBatchSource<'a> {
    fn new(loader: &'a DataLoader, prefetch_batches: usize) -> Self {
        if prefetch_batches == 0 {
            Self::Direct(loader)
        } else {
            Self::Prefetch(BatchPrefetcher::new(loader.clone(), prefetch_batches))
        }
    }

    fn next_batch<B: Backend>(&self, device: &B::Device) -> Result<TokenBatch<B>, String> {
        match self {
            Self::Direct(loader) => loader.next_batch(device),
            Self::Prefetch(prefetcher) => prefetcher.next_batch(device),
        }
    }
}

fn train_language_model<B, M, F>(
    mut model: M,
    loader: &DataLoader,
    value_loader: &DataLoader,
    device: &B::Device,
    params: TrainingParams,
    forward: impl Fn(&M, Tensor<B, 2, Int>) -> Tensor<B, 3>,
    mut periodic_save: F,
) -> Result<TrainingOutcome<M>, String>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    B::FloatElem: Into<f64>,
    F: FnMut(&M, usize) -> Result<(), String>,
{
    let training_batches = TrainingBatchSource::new(loader, params.prefetch_batches);
    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(params.grad_clipping)
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let started_at = std::time::Instant::now();
    let mut final_value_loss = None;
    let mut interrupted = false;
    let mut steps_completed = 0usize;

    for step in 0..params.steps {
        // Honour any SIGINT/SIGTERM received before this step starts so the
        // operator gets a clean partial-checkpoint save instead of a hard
        // abort mid-step. See `runtime_signals` for the installer.
        if crate::runtime_signals::interrupt_requested() {
            interrupted = true;
            break;
        }

        let (inputs, targets) = training_batches.next_batch::<B>(device)?;
        let loss = language_model_loss(&loss_fn, forward(&model, inputs), targets);

        let learning_rate = learning_rate_at_step(
            step,
            params.learning_rate,
            params.min_learning_rate,
            params.warmup_steps,
            params.steps,
            params.lr_schedule,
        );
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(learning_rate, model, grads);
        steps_completed = step + 1;

        // Periodic checkpoint cadence is orthogonal to `eval_interval` —
        // they fire on independent step counts. Skip the final step so we
        // don't double-save with the caller's end-of-run write.
        if params.periodic_checkpoint_interval > 0
            && steps_completed != params.steps
            && steps_completed.is_multiple_of(params.periodic_checkpoint_interval)
        {
            periodic_save(&model, steps_completed)?;
        }

        if should_log_training_step(step, params.steps, params.eval_interval) {
            let throughput = TrainingThroughput::from_progress(
                step + 1,
                loader.batch_size,
                loader.block_size,
                started_at.elapsed().as_millis(),
            );
            let training_loss = loss.into_scalar().into();
            let value_loss: f64 = value_loss(value_loader, device, &loss_fn, |inputs| {
                forward(&model, inputs)
            })?
            .into();
            final_value_loss = Some(value_loss);
            log_training_progress(
                params.log_context.clone(),
                step,
                params.steps,
                training_loss,
                value_loss,
                learning_rate,
                throughput,
            );
        }
    }

    let final_value_loss = final_value_loss.unwrap_or_else(|| {
        value_loss(value_loader, device, &loss_fn, |inputs| {
            forward(&model, inputs)
        })
        .map(Into::into)
        .unwrap_or(f64::NAN)
    });
    Ok(TrainingOutcome {
        model,
        metrics: TrainingMetrics {
            final_value_loss,
            final_perplexity: perplexity(final_value_loss),
        },
        steps_completed,
        interrupted,
    })
}

impl<B> TrivialModel<B>
where
    B: AutodiffBackend,
    B::FloatElem: Into<f64>,
{
    pub fn train(
        loader: &DataLoader,
        value_loader: &DataLoader,
        device: &B::Device,
        vocab_size: usize,
        d_model: usize,
        params: TrainingParams,
    ) -> Result<TrainingOutcome<Self>, String> {
        train_language_model(
            TrivialModel::<B>::new(vocab_size, d_model, device),
            loader,
            value_loader,
            device,
            params,
            |model, inputs| model.forward(inputs),
            no_periodic_save,
        )
    }
}

impl<B> SingleAttentionModel<B>
where
    B: AutodiffBackend,
    B::FloatElem: Into<f64>,
{
    pub fn train(
        loader: &DataLoader,
        value_loader: &DataLoader,
        device: &B::Device,
        vocab_size: usize,
        d_model: usize,
        head_dim: usize,
        params: TrainingParams,
    ) -> Result<TrainingOutcome<Self>, String> {
        train_language_model(
            SingleAttentionModel::<B>::new(vocab_size, d_model, head_dim, device),
            loader,
            value_loader,
            device,
            params,
            |model, inputs| model.forward_tokens(inputs),
            no_periodic_save,
        )
    }
}

impl<B> MultiAttentionModel<B>
where
    B: AutodiffBackend,
    B::FloatElem: Into<f64>,
{
    pub fn train(
        loader: &DataLoader,
        value_loader: &DataLoader,
        device: &B::Device,
        vocab_size: usize,
        d_model: usize,
        num_heads: usize,
        params: TrainingParams,
    ) -> Result<TrainingOutcome<Self>, String> {
        train_language_model(
            MultiAttentionModel::<B>::new(vocab_size, d_model, num_heads, device),
            loader,
            value_loader,
            device,
            params,
            |model, inputs| model.forward_tokens(inputs),
            no_periodic_save,
        )
    }
}

impl<B> MiniGpt<B>
where
    B: AutodiffBackend,
    B::FloatElem: Into<f64>,
{
    pub fn train(
        loader: &DataLoader,
        value_loader: &DataLoader,
        device: &B::Device,
        config: MiniGptConfig,
        params: TrainingParams,
    ) -> Result<TrainingOutcome<Self>, String> {
        Self::train_with_periodic_save(
            loader,
            value_loader,
            device,
            config,
            params,
            no_periodic_save,
        )
    }

    /// Same as [`MiniGpt::train`] but lets the caller hook into the periodic
    /// save cadence controlled by [`TrainingParams::periodic_checkpoint_interval`].
    /// The closure is invoked with the current model and the step number
    /// (1-indexed) every `periodic_checkpoint_interval` steps, excluding the
    /// final step. It is **not** invoked when the interval is `0`.
    pub fn train_with_periodic_save<F>(
        loader: &DataLoader,
        value_loader: &DataLoader,
        device: &B::Device,
        config: MiniGptConfig,
        params: TrainingParams,
        periodic_save: F,
    ) -> Result<TrainingOutcome<Self>, String>
    where
        F: FnMut(&Self, usize) -> Result<(), String>,
    {
        train_language_model(
            MiniGpt::<B>::new(
                config.vocab_size,
                config.d_model,
                config.num_blocks,
                config.max_position_embeddings,
                config.num_heads,
                device,
            ),
            loader,
            value_loader,
            device,
            params,
            |model, inputs| model.forward_tokens(inputs),
            periodic_save,
        )
    }
}

/// Default no-op save callback used by every training variant that does
/// not support persistence (the three teaching models) or when the caller
/// has not opted in to periodic saves.
fn no_periodic_save<M>(_model: &M, _step: usize) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod lr_schedule_tests {
    use super::*;

    const BASE_LR: f64 = 1e-3;
    const MIN_LR: f64 = 1e-5;
    const TOTAL_STEPS: usize = 100;
    const WARMUP: usize = 10;

    #[test]
    fn constant_schedule_returns_base_lr_everywhere() {
        for step in [0, 1, WARMUP, TOTAL_STEPS / 2, TOTAL_STEPS] {
            let lr = learning_rate_at_step(
                step,
                BASE_LR,
                MIN_LR,
                WARMUP,
                TOTAL_STEPS,
                LrSchedule::Constant,
            );
            assert_eq!(
                lr, BASE_LR,
                "constant schedule must ignore warmup/min at step {step}"
            );
        }
    }

    #[test]
    fn cosine_warmup_starts_near_zero() {
        let lr = learning_rate_at_step(0, BASE_LR, MIN_LR, WARMUP, TOTAL_STEPS, LrSchedule::Cosine);
        assert!(
            lr.abs() < 1e-12,
            "lr(0) during warmup should be ~0, got {lr}"
        );
    }

    #[test]
    fn cosine_warmup_ramps_linearly_to_base_lr_at_warmup_boundary() {
        // Mid-warmup is a linear fraction of base_lr.
        let mid = learning_rate_at_step(
            WARMUP / 2,
            BASE_LR,
            MIN_LR,
            WARMUP,
            TOTAL_STEPS,
            LrSchedule::Cosine,
        );
        assert!(
            (mid - BASE_LR * 0.5).abs() < 1e-12,
            "mid-warmup lr wrong: {mid}"
        );

        // At the warmup boundary the schedule hands off to cosine at t=0,
        // which evaluates to exactly base_lr.
        let boundary = learning_rate_at_step(
            WARMUP,
            BASE_LR,
            MIN_LR,
            WARMUP,
            TOTAL_STEPS,
            LrSchedule::Cosine,
        );
        assert!(
            (boundary - BASE_LR).abs() < 1e-12,
            "lr(warmup_steps) should equal base_lr, got {boundary}"
        );
    }

    #[test]
    fn cosine_decays_to_min_lr_at_total_steps() {
        let lr = learning_rate_at_step(
            TOTAL_STEPS,
            BASE_LR,
            MIN_LR,
            WARMUP,
            TOTAL_STEPS,
            LrSchedule::Cosine,
        );
        assert!(
            (lr - MIN_LR).abs() < 1e-12,
            "lr(total_steps) should equal min_lr, got {lr}"
        );
    }

    #[test]
    fn cosine_without_warmup_starts_at_base_lr() {
        let lr = learning_rate_at_step(0, BASE_LR, MIN_LR, 0, TOTAL_STEPS, LrSchedule::Cosine);
        assert!(
            (lr - BASE_LR).abs() < 1e-12,
            "cosine without warmup should start at base_lr, got {lr}"
        );
    }

    #[test]
    fn cosine_is_monotonically_non_increasing_after_warmup() {
        let mut prev = f64::INFINITY;
        for step in WARMUP..=TOTAL_STEPS {
            let lr = learning_rate_at_step(
                step,
                BASE_LR,
                MIN_LR,
                WARMUP,
                TOTAL_STEPS,
                LrSchedule::Cosine,
            );
            assert!(lr <= prev + 1e-12, "cosine decay increased at step {step}");
            prev = lr;
        }
    }

    #[test]
    fn cosine_clamps_past_total_steps_to_min_lr() {
        let lr = learning_rate_at_step(
            TOTAL_STEPS * 2,
            BASE_LR,
            MIN_LR,
            WARMUP,
            TOTAL_STEPS,
            LrSchedule::Cosine,
        );
        assert!(
            (lr - MIN_LR).abs() < 1e-12,
            "beyond total_steps should clamp to min_lr"
        );
    }

    #[test]
    fn lr_schedule_round_trips_through_str() {
        assert_eq!(
            "constant".parse::<LrSchedule>().unwrap(),
            LrSchedule::Constant
        );
        assert_eq!("cosine".parse::<LrSchedule>().unwrap(), LrSchedule::Cosine);
        let err = "linear".parse::<LrSchedule>().unwrap_err();
        assert!(err.to_string().contains("unsupported lr schedule 'linear'"));
    }
}
