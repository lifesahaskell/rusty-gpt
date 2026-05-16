use anyhow::{Context, Result, bail};
use rusty_gpt::loader::data::DataLoader;
use rusty_gpt::model::persistence::{
    CheckpointMetadata, CheckpointModelShape, CheckpointTokenizer, CheckpointTrainingMetrics,
    CheckpointTrainingRun, save_checkpoint_metadata, save_model, sha256_file_hex,
};
use rusty_gpt::model::{
    MiniGpt, MiniGptConfig, MultiAttentionModel, SingleAttentionModel, TrainingLogContext,
    TrainingOutcome, TrainingParams, TrivialModel,
};
use rusty_gpt::observability::{EventLogger, RuntimeEvent};
use rusty_gpt::runtime_signals::{INTERRUPTED_EXIT_CODE, install_training_signal_handler};
use rusty_gpt::utils::{BenchmarkConfig, benchmark_generation};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::runtime_assets::{minigpt_tokenizer_path, tokenizer_for_model};
use crate::runtime_config::{Hyperparameters, ModelChoice};

pub(crate) fn run_training_demo<B>(
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

    // Install the SIGINT/SIGTERM handler *only* on the training path so that
    // --serve and --interactive-generate keep the default Ctrl-C behaviour.
    install_training_signal_handler()
        .context("failed to install SIGINT/SIGTERM handler for training")?;

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
pub(crate) struct TrainingDemoOptions {
    pub(crate) backend_label: &'static str,
    pub(crate) logger: EventLogger,
    pub(crate) benchmark_generation: bool,
    pub(crate) benchmark_config: BenchmarkConfig,
    pub(crate) input_source: String,
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
            let interrupted = outcome.interrupted;
            let steps_completed = outcome.steps_completed;
            if !interrupted && run.benchmark_generation {
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
            if interrupted {
                eprintln!(
                    "training interrupted at step {steps_completed}; partial checkpoint saved. Exiting with code {INTERRUPTED_EXIT_CODE}."
                );
                std::process::exit(INTERRUPTED_EXIT_CODE);
            }
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

    // When the training loop was interrupted, redirect the save to a
    // sibling path tagged with the step number so the normal end-of-run
    // checkpoint (and any periodic snapshot retention from T3) is not
    // overwritten by the partial run.
    let save_path = if outcome.interrupted {
        interrupted_checkpoint_path(checkpoint_path, outcome.steps_completed)
    } else {
        checkpoint_path.to_path_buf()
    };

    let started_at = Instant::now();
    save_model(outcome.model, &save_path)
        .with_context(|| format!("failed to save minigpt checkpoint to {:?}", save_path))?;
    let mut metadata = checkpoint_metadata(run, outcome.metrics)?;
    if outcome.interrupted {
        metadata.interrupted = true;
        metadata.interrupted_at_step = Some(outcome.steps_completed);
    }
    save_checkpoint_metadata(&save_path, &metadata).with_context(|| {
        format!("failed to save checkpoint metadata for {:?}", save_path)
    })?;
    let saved_mpk = save_path.with_extension("mpk").display().to_string();
    run.logger.log(RuntimeEvent::CheckpointSaved {
        path: saved_mpk.clone(),
        elapsed_ms: started_at.elapsed().as_millis(),
    });
    if outcome.interrupted {
        eprintln!("interrupted checkpoint saved at {saved_mpk}");
    }

    Ok(())
}

/// Suffix the checkpoint path with `.interrupted-step-<N>.mpk`. The trailing
/// `.mpk` is included here on purpose: `std::path::Path::with_extension`
/// treats anything after the last `.` as an extension, so Burn's recorder
/// (which calls `with_extension("mpk")` internally) would silently strip
/// `interrupted-step-N` from a bare path. By baking `.mpk` into the path we
/// pass downstream, `with_extension("mpk")` becomes a no-op and the on-disk
/// file is `<base>.interrupted-step-<N>.mpk` as the spec requires.
///
/// Example: `checkpoints/mini_gpt` + step `42` ⇒
/// `checkpoints/mini_gpt.interrupted-step-42.mpk`.
pub(crate) fn interrupted_checkpoint_path(base: &Path, step: usize) -> PathBuf {
    let file_name = base
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");
    let tagged = format!("{file_name}.interrupted-step-{step}.mpk");
    match base.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(tagged),
        _ => PathBuf::from(tagged),
    }
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
        interrupted: false,
        interrupted_at_step: None,
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

pub(crate) fn split_training_and_value_tokens(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupted_checkpoint_path_keeps_parent_and_tags_step() {
        let result = interrupted_checkpoint_path(Path::new("checkpoints/mini_gpt"), 42);
        assert_eq!(
            PathBuf::from("checkpoints/mini_gpt.interrupted-step-42.mpk"),
            result
        );
    }

    #[test]
    fn interrupted_checkpoint_path_handles_bare_filename() {
        let result = interrupted_checkpoint_path(Path::new("mini_gpt"), 7);
        assert_eq!(PathBuf::from("mini_gpt.interrupted-step-7.mpk"), result);
    }

    #[test]
    fn interrupted_checkpoint_path_is_idempotent_through_with_extension() {
        // Critical regression guard: std::path::Path::with_extension("mpk")
        // must NOT strip the interrupted-step suffix. The bug it prevents:
        // `mini_gpt.interrupted-step-1` (without .mpk) round-trips through
        // Burn's recorder as `mini_gpt.mpk`.
        let result = interrupted_checkpoint_path(Path::new("checkpoints/mini_gpt"), 1);
        assert_eq!(
            Path::new("checkpoints/mini_gpt.interrupted-step-1.mpk"),
            result.with_extension("mpk")
        );
    }
}
