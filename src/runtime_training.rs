use anyhow::{Context, Result, bail};
use rusty_gpt::loader::data::DataLoader;
use rusty_gpt::model::persistence::{
    CheckpointMetadata, CheckpointModelShape, CheckpointTokenizer, CheckpointTrainingMetrics,
    CheckpointTrainingRun, load_checkpoint_metadata, save_checkpoint_metadata, save_model,
    sha256_file_hex,
};
use rusty_gpt::model::{
    MiniGpt, MultiAttentionModel, SingleAttentionModel, TrainingLogContext, TrainingOutcome,
    TrainingParams, TrivialModel,
};
use rusty_gpt::observability::{EventLogger, RuntimeEvent};
use rusty_gpt::runtime_signals::{INTERRUPTED_EXIT_CODE, install_training_signal_handler};
use rusty_gpt::utils::{BenchmarkConfig, benchmark_generation};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::runtime_assets::{load_minigpt_checkpoint, minigpt_tokenizer_path, tokenizer_for_model};
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
            resume_from: options.resume_from.as_deref(),
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
    /// Resume MiniGPT training from this checkpoint (`--resume-from`). `None`
    /// for a fresh run. Only the MiniGPT training arm consumes it.
    pub(crate) resume_from: Option<PathBuf>,
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
    resume_from: Option<&'a Path>,
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
    .with_prefetch_batches(run.hyperparameters.prefetch_batches)
    .with_learning_rate_schedule(
        run.hyperparameters.learning_rate_schedule,
        run.hyperparameters.lr_warmup_steps,
    )
    .with_sampling_policy(run.hyperparameters.sampling_policy)
    .with_periodic_checkpoint_interval(run.hyperparameters.checkpoint_interval);

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
            let interval = run.hyperparameters.checkpoint_interval;
            let keep = run.hyperparameters.checkpoint_keep;
            let checkpoint_path = run.checkpoint_path;
            let logger = run.logger.clone();
            let metadata_template = checkpoint_metadata(
                &run,
                rusty_gpt::model::TrainingMetrics {
                    final_value_loss: 0.0,
                    final_perplexity: 0.0,
                },
            )?;
            // Fresh run builds a template; `--resume-from` loads weights via
            // the strict metadata loader and reports the step count to continue
            // from. `--train-steps` is the absolute target in both cases.
            let (initial_model, start_step) = resolve_minigpt_start_model::<B>(&run)?;
            let train_params = params
                .with_grad_clip_norm(run.hyperparameters.minigpt_grad_clip_norm)
                .with_start_step(start_step);
            let outcome = MiniGpt::<B>::train_prebuilt_with_periodic_save(
                initial_model,
                run.data_loader,
                run.value_loader,
                run.device,
                train_params,
                |model, step| {
                    save_periodic_minigpt_checkpoint(
                        model,
                        checkpoint_path,
                        step,
                        interval,
                        keep,
                        &metadata_template,
                        &logger,
                    )
                    .map_err(|err| err.to_string())
                },
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

/// Build the model the training loop should start from and the absolute step
/// index it should start at.
///
/// - Fresh run (`--resume-from` absent): a newly-initialised [`MiniGpt`] and
///   step `0`.
/// - Resume (`--resume-from <checkpoint>`): weights loaded through the strict
///   metadata loader (shape + tokenizer-hash validated, diff-style error on
///   mismatch) and the checkpoint's recorded `completed_steps` so the loop
///   continues from `completed_steps + 1`. Optimizer moments are *not*
///   restored — Burn only persists module weights.
fn resolve_minigpt_start_model<B>(run: &TrainingRun<'_, B>) -> Result<(MiniGpt<B>, usize)>
where
    B: burn::tensor::backend::AutodiffBackend,
{
    let make_template = || {
        MiniGpt::<B>::new(
            run.vocab_size,
            run.hyperparameters.embed_dim,
            run.hyperparameters.num_layers,
            run.hyperparameters.block_size,
            run.hyperparameters.num_heads,
            run.device,
        )
    };

    let Some(resume_path) = run.resume_from else {
        return Ok((make_template(), 0));
    };

    // Burn's recorder and the metadata loaders call `with_extension("mpk")`,
    // which corrupts paths whose stem contains a dot (e.g. `mini_gpt.step-4`).
    // Baking `.mpk` in makes `with_extension("mpk")` a no-op and keeps the
    // sibling `.metadata.json` sidecar path correct for periodic/interrupted
    // snapshots too. See `step_checkpoint_path` for the same trick.
    let load_path = resume_load_path(resume_path);

    let metadata = load_checkpoint_metadata(&load_path)
        .with_context(|| {
            format!(
                "failed to read resume checkpoint metadata for {:?}",
                load_path
            )
        })?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "cannot resume from {:?}: missing metadata sidecar {:?}. --resume-from needs a checkpoint written with a completed_steps sidecar (train one with this build first).",
                load_path,
                rusty_gpt::model::persistence::metadata_path(&load_path)
            )
        })?;

    let completed_steps = metadata.completed_steps;
    let target = run.hyperparameters.train_steps;
    if completed_steps >= target {
        bail!(
            "nothing to resume: checkpoint {:?} already completed {completed_steps} steps, which is >= --train-steps {target}. Raise --train-steps above {completed_steps} to continue training.",
            load_path
        );
    }

    let model = load_minigpt_checkpoint(make_template(), &load_path, run.device, &run.logger)
        .with_context(|| format!("failed to resume minigpt training from {:?}", load_path))?;

    Ok((model, completed_steps))
}

/// Normalise a `--resume-from` checkpoint path so downstream loaders (which
/// call `with_extension("mpk")`) resolve the real on-disk `.mpk` file even when
/// the stem contains a dot, e.g. `mini_gpt.step-4` → `mini_gpt.step-4.mpk`.
fn resume_load_path(resume_path: &Path) -> PathBuf {
    if resume_path.extension().and_then(|ext| ext.to_str()) == Some("mpk") {
        resume_path.to_path_buf()
    } else {
        let mut raw = resume_path.as_os_str().to_owned();
        raw.push(".mpk");
        PathBuf::from(raw)
    }
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
    // Absolute steps completed — the value `--resume-from` reads to continue.
    metadata.completed_steps = outcome.steps_completed;
    if outcome.interrupted {
        metadata.interrupted = true;
        metadata.interrupted_at_step = Some(outcome.steps_completed);
    }
    save_checkpoint_metadata(&save_path, &metadata)
        .with_context(|| format!("failed to save checkpoint metadata for {:?}", save_path))?;
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

/// Sibling of [`interrupted_checkpoint_path`] for the periodic (T3) cadence.
/// Produces `<base>.step-<N>.mpk` with the same `.mpk`-baked-in trick to
/// survive Burn's internal `with_extension("mpk")`.
pub(crate) fn step_checkpoint_path(base: &Path, step: usize) -> PathBuf {
    let file_name = base
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");
    let tagged = format!("{file_name}.step-{step}.mpk");
    match base.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(tagged),
        _ => PathBuf::from(tagged),
    }
}

/// Save a mid-run MiniGPT snapshot to `<checkpoint>.step-<N>.mpk` and prune
/// older periodic snapshots beyond `keep`. The final end-of-run save and any
/// `interrupted-step-*` save are **never** pruned — only the periodic
/// (`.step-N.`) snapshots are subject to retention.
fn save_periodic_minigpt_checkpoint<B>(
    model: &MiniGpt<B>,
    checkpoint_path: &Path,
    step: usize,
    interval: usize,
    keep: usize,
    metadata_template: &CheckpointMetadata,
    logger: &EventLogger,
) -> Result<()>
where
    B: burn::tensor::backend::AutodiffBackend,
{
    if let Some(parent) = checkpoint_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create checkpoint directory {:?}", parent))?;
    }

    let save_path = step_checkpoint_path(checkpoint_path, step);
    let started_at = Instant::now();
    // Burn's save_file takes ownership; cloning is cheap relative to a
    // full training step and keeps the in-memory weights available for the
    // next iteration. For S1 this is fine; async double-buffering is a
    // Sprint 3 enhancement if profiling shows it matters.
    save_model(model.clone(), &save_path).with_context(|| {
        format!(
            "failed to save periodic minigpt checkpoint to {:?}",
            save_path
        )
    })?;

    let mut metadata = metadata_template.clone();
    metadata.step = Some(step);
    metadata.interval = Some(interval);
    // `step` here is the absolute completed-step count for this snapshot.
    metadata.completed_steps = step;
    save_checkpoint_metadata(&save_path, &metadata).with_context(|| {
        format!(
            "failed to save periodic checkpoint metadata for {:?}",
            save_path
        )
    })?;

    let saved_mpk = save_path.display().to_string();
    logger.log(RuntimeEvent::CheckpointSaved {
        path: saved_mpk,
        elapsed_ms: started_at.elapsed().as_millis(),
    });

    prune_old_step_checkpoints(checkpoint_path, keep);
    Ok(())
}

/// Scan the directory containing `checkpoint_base`, find all periodic
/// `<base>.step-<N>.mpk` snapshots, keep the most recent `keep`, delete the
/// rest along with their `.step-<N>.metadata.json` sidecars. Orphan
/// sidecars (no matching `.mpk`) are also removed. Files that don't match
/// the periodic pattern — including the final `<base>.mpk` and any
/// `interrupted-step-*` save — are left untouched.
///
/// Every deletion is logged to stderr so a curious operator can audit what
/// the retention policy did.
fn prune_old_step_checkpoints(checkpoint_base: &Path, keep: usize) {
    if keep == 0 {
        return;
    }
    let Some(dir) = checkpoint_base.parent() else {
        return;
    };
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    let Some(base_name) = checkpoint_base.file_name().and_then(|name| name.to_str()) else {
        return;
    };
    let mpk_prefix = format!("{base_name}.step-");
    let sidecar_prefix = format!("{base_name}.step-");

    let Ok(read) = fs::read_dir(dir) else {
        return;
    };

    let mut step_mpks: Vec<(usize, PathBuf)> = Vec::new();
    let mut sidecars_by_step: std::collections::HashMap<usize, PathBuf> =
        std::collections::HashMap::new();
    for entry in read.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(rest) = name.strip_prefix(&mpk_prefix)
            && let Some(step_str) = rest.strip_suffix(".mpk")
            && let Ok(step) = step_str.parse::<usize>()
        {
            step_mpks.push((step, path.clone()));
            continue;
        }
        if let Some(rest) = name.strip_prefix(&sidecar_prefix)
            && let Some(step_str) = rest.strip_suffix(".metadata.json")
            && let Ok(step) = step_str.parse::<usize>()
        {
            sidecars_by_step.insert(step, path);
        }
    }

    step_mpks.sort_by(|(a, _), (b, _)| b.cmp(a)); // newest first
    if step_mpks.len() > keep {
        for (step, mpk_path) in step_mpks.iter().skip(keep) {
            match fs::remove_file(mpk_path) {
                Ok(()) => eprintln!(
                    "checkpoint retention: pruned {} (older than the last {keep} periodic snapshots)",
                    mpk_path.display()
                ),
                Err(err) => eprintln!(
                    "checkpoint retention: failed to prune {}: {err}",
                    mpk_path.display()
                ),
            }
            if let Some(sidecar) = sidecars_by_step.remove(step) {
                match fs::remove_file(&sidecar) {
                    Ok(()) => eprintln!(
                        "checkpoint retention: pruned {} (sidecar for step {step})",
                        sidecar.display()
                    ),
                    Err(err) => eprintln!(
                        "checkpoint retention: failed to prune sidecar {}: {err}",
                        sidecar.display()
                    ),
                }
            }
        }
    }

    // Orphan sidecars: any remaining sidecar whose .mpk no longer exists.
    let surviving_steps: std::collections::HashSet<usize> =
        step_mpks.iter().take(keep).map(|(step, _)| *step).collect();
    for (step, sidecar) in sidecars_by_step {
        if !surviving_steps.contains(&step) {
            match fs::remove_file(&sidecar) {
                Ok(()) => eprintln!(
                    "checkpoint retention: pruned orphan sidecar {}",
                    sidecar.display()
                ),
                Err(err) => eprintln!(
                    "checkpoint retention: failed to prune orphan sidecar {}: {err}",
                    sidecar.display()
                ),
            }
        }
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
        step: None,
        interval: None,
        // Overwritten at each save site with the absolute completed-step count.
        completed_steps: 0,
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

    #[test]
    fn step_checkpoint_path_keeps_parent_and_tags_step() {
        let result = step_checkpoint_path(Path::new("checkpoints/mini_gpt"), 100);
        assert_eq!(PathBuf::from("checkpoints/mini_gpt.step-100.mpk"), result);
    }

    #[test]
    fn step_checkpoint_path_is_idempotent_through_with_extension() {
        let result = step_checkpoint_path(Path::new("checkpoints/mini_gpt"), 200);
        assert_eq!(
            Path::new("checkpoints/mini_gpt.step-200.mpk"),
            result.with_extension("mpk")
        );
    }

    #[test]
    fn prune_old_step_checkpoints_keeps_newest_k_and_deletes_orphan_sidecars() {
        let dir = std::env::temp_dir().join(format!("rusty-gpt-prune-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let base = dir.join("mini_gpt");

        // Seed the directory with periodic snapshots at steps 100/200/300,
        // a stale orphan sidecar at step 50 (no .mpk), the final save, and
        // an interrupted save (which must NOT be pruned).
        for step in [100usize, 200, 300] {
            fs::write(dir.join(format!("mini_gpt.step-{step}.mpk")), b"weights").unwrap();
            fs::write(
                dir.join(format!("mini_gpt.step-{step}.metadata.json")),
                b"{}",
            )
            .unwrap();
        }
        fs::write(dir.join("mini_gpt.step-50.metadata.json"), b"{}").unwrap(); // orphan
        fs::write(dir.join("mini_gpt.mpk"), b"final").unwrap();
        fs::write(dir.join("mini_gpt.metadata.json"), b"{}").unwrap();
        fs::write(dir.join("mini_gpt.interrupted-step-7.mpk"), b"int").unwrap();
        fs::write(dir.join("mini_gpt.interrupted-step-7.metadata.json"), b"{}").unwrap();

        prune_old_step_checkpoints(&base, 2);

        let names: std::collections::HashSet<String> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect();

        // Step-300 and step-200 survive; step-100 + its sidecar are pruned;
        // orphan step-50 sidecar is removed; the final and the interrupted
        // saves are untouched.
        assert!(names.contains("mini_gpt.step-300.mpk"));
        assert!(names.contains("mini_gpt.step-300.metadata.json"));
        assert!(names.contains("mini_gpt.step-200.mpk"));
        assert!(names.contains("mini_gpt.step-200.metadata.json"));
        assert!(!names.contains("mini_gpt.step-100.mpk"));
        assert!(!names.contains("mini_gpt.step-100.metadata.json"));
        assert!(!names.contains("mini_gpt.step-50.metadata.json"));
        assert!(names.contains("mini_gpt.mpk"));
        assert!(names.contains("mini_gpt.metadata.json"));
        assert!(names.contains("mini_gpt.interrupted-step-7.mpk"));
        assert!(names.contains("mini_gpt.interrupted-step-7.metadata.json"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn prune_old_step_checkpoints_with_keep_zero_is_noop() {
        let dir = std::env::temp_dir().join(format!("rusty-gpt-prune-zero-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let base = dir.join("mini_gpt");

        fs::write(dir.join("mini_gpt.step-100.mpk"), b"weights").unwrap();

        prune_old_step_checkpoints(&base, 0);

        assert!(dir.join("mini_gpt.step-100.mpk").exists());

        let _ = fs::remove_dir_all(&dir);
    }
}
