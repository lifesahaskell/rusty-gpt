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
        }
    }

    pub fn with_grad_clip_norm(mut self, norm: f32) -> Self {
        self.grad_clipping = Some(GradientClippingConfig::Norm(norm));
        self
    }

    pub fn with_prefetch_batches(mut self, batches: usize) -> Self {
        self.prefetch_batches = batches;
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
pub(super) fn training_progress_log_line<B: Backend>(
    context: TrainingLogContext,
    step: usize,
    steps: usize,
    training_loss: B::FloatElem,
    value_loss: B::FloatElem,
    throughput: TrainingThroughput,
) -> String {
    match context.logger.format() {
        LogFormat::Plain => {
            format!(
                "Step {step}: training loss = {training_loss:.4}, value loss = {value_loss:.4}, tokens_per_second={:.2}, steps_per_second={:.4}, step_ms_mean={:.2}",
                throughput.tokens_per_second, throughput.steps_per_second, throughput.step_ms_mean
            )
        }
        LogFormat::Json => {
            format!(
                r#"{{"event":"training_progress","backend":"{}","model":"{}","step":{},"total_steps":{},"training_loss":{:.6},"value_loss":{:.6},"elapsed_ms":{},"tokens_per_second":{:.6},"steps_per_second":{:.6},"step_ms_mean":{:.6}}}"#,
                context.backend,
                context.model,
                step,
                steps,
                training_loss,
                value_loss,
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

fn log_training_progress(
    context: TrainingLogContext,
    step: usize,
    steps: usize,
    training_loss: f64,
    value_loss: f64,
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

fn train_language_model<B, M>(
    mut model: M,
    loader: &DataLoader,
    value_loader: &DataLoader,
    device: &B::Device,
    params: TrainingParams,
    forward: impl Fn(&M, Tensor<B, 2, Int>) -> Tensor<B, 3>,
) -> Result<TrainingOutcome<M>, String>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    B::FloatElem: Into<f64>,
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

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(params.learning_rate, model, grads);
        steps_completed = step + 1;

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
        )
    }
}
