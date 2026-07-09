use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, RecorderError};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

type ModelRecorder = NamedMpkFileRecorder<FullPrecisionSettings>;

pub fn save_model<M, B, P>(model: M, path: P) -> Result<(), RecorderError>
where
    M: Module<B>,
    B: Backend,
    P: Into<PathBuf>,
{
    model.save_file(path, &ModelRecorder::default())
}

pub fn load_model<M, B, P>(model: M, path: P, device: &B::Device) -> Result<M, RecorderError>
where
    M: Module<B>,
    B: Backend,
    P: Into<PathBuf>,
{
    model.load_file(path, &ModelRecorder::default(), device)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub version: u32,
    pub created_at_utc: String,
    pub git_commit: Option<String>,
    pub input_source: String,
    pub model_shape: CheckpointModelShape,
    pub tokenizer: CheckpointTokenizer,
    pub training: CheckpointTrainingRun,
    pub final_metrics: CheckpointTrainingMetrics,
    /// `true` when this checkpoint was written by the SIGINT/SIGTERM graceful
    /// shutdown path instead of a normal end-of-run save. Older sidecars
    /// without this field deserialize to `false`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub interrupted: bool,
    /// When `interrupted` is true, the step number at which the training loop
    /// observed the signal and broke out. `None` for normal end-of-run saves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interrupted_at_step: Option<usize>,
    /// Step number at which this periodic-cadence snapshot was written.
    /// `None` for the normal end-of-run save and interrupted saves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<usize>,
    /// `--checkpoint-interval` value in effect when this periodic snapshot
    /// was written. `None` for the normal end-of-run save and interrupted
    /// saves; lets future tooling reason about cadence without inferring
    /// it from filenames.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<usize>,
    /// Absolute number of training steps completed at the moment this
    /// checkpoint was written. Recorded on *every* save — the final
    /// end-of-run save, each periodic `.step-N` snapshot, and any interrupted
    /// save. `--resume-from` reads this field to continue the step counter
    /// (and with it the LR schedule and `--checkpoint-interval` numbering)
    /// exactly where the previous run stopped. Legacy sidecars written before
    /// this field existed deserialize to `0` via `#[serde(default)]`, keeping
    /// old dense checkpoints loadable (they simply resume from scratch).
    #[serde(default)]
    pub completed_steps: usize,
}

fn is_false(value: &bool) -> bool {
    !value
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointModelShape {
    pub vocab_size: usize,
    pub block_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointTokenizer {
    pub path: String,
    pub sha256: Option<String>,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointTrainingRun {
    pub backend: String,
    pub train_tokens: usize,
    pub value_tokens: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub train_steps: usize,
    pub eval_interval: usize,
    pub grad_clip_norm: f32,
    pub prefetch_batches: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CheckpointTrainingMetrics {
    pub final_value_loss: f64,
    pub final_perplexity: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointCompatibilityIssue {
    pub field: String,
    pub expected: String,
    pub found: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointCompatibilityReport {
    pub compatible: bool,
    pub issues: Vec<CheckpointCompatibilityIssue>,
}

impl CheckpointCompatibilityReport {
    fn new(issues: Vec<CheckpointCompatibilityIssue>) -> Self {
        Self {
            compatible: issues.is_empty(),
            issues,
        }
    }
}

impl fmt::Display for CheckpointCompatibilityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.compatible {
            return write!(f, "checkpoint metadata is compatible");
        }

        write!(f, "checkpoint metadata compatibility failed")?;
        for issue in &self.issues {
            write!(
                f,
                "; {} expected {}, found {}",
                issue.field, issue.expected, issue.found
            )?;
        }
        Ok(())
    }
}

pub fn metadata_path<P: AsRef<Path>>(checkpoint_path: P) -> PathBuf {
    checkpoint_path.as_ref().with_extension("metadata.json")
}

pub fn save_checkpoint_metadata<P: AsRef<Path>>(
    checkpoint_path: P,
    metadata: &CheckpointMetadata,
) -> anyhow::Result<()> {
    let path = metadata_path(checkpoint_path);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(metadata)?)?;
    Ok(())
}

pub fn load_checkpoint_metadata<P: AsRef<Path>>(
    checkpoint_path: P,
) -> anyhow::Result<Option<CheckpointMetadata>> {
    let path = metadata_path(checkpoint_path);
    if !path.exists() {
        return Ok(None);
    }

    Ok(Some(serde_json::from_slice(&fs::read(path)?)?))
}

pub fn checkpoint_compatibility_report(
    metadata: &CheckpointMetadata,
    expected_shape: &CheckpointModelShape,
    tokenizer_path: Option<&Path>,
) -> anyhow::Result<CheckpointCompatibilityReport> {
    let mut issues = Vec::new();

    push_usize_issue(
        &mut issues,
        "model_shape.vocab_size",
        expected_shape.vocab_size,
        metadata.model_shape.vocab_size,
    );
    push_usize_issue(
        &mut issues,
        "model_shape.block_size",
        expected_shape.block_size,
        metadata.model_shape.block_size,
    );
    push_usize_issue(
        &mut issues,
        "model_shape.embed_dim",
        expected_shape.embed_dim,
        metadata.model_shape.embed_dim,
    );
    push_usize_issue(
        &mut issues,
        "model_shape.num_heads",
        expected_shape.num_heads,
        metadata.model_shape.num_heads,
    );
    push_usize_issue(
        &mut issues,
        "model_shape.num_layers",
        expected_shape.num_layers,
        metadata.model_shape.num_layers,
    );
    push_usize_issue(
        &mut issues,
        "tokenizer.vocab_size",
        expected_shape.vocab_size,
        metadata.tokenizer.vocab_size,
    );

    if let Some(expected_sha256) = metadata.tokenizer.sha256.as_deref()
        && let Some(path) = tokenizer_path
    {
        let found = sha256_file_hex(path)?.unwrap_or_else(|| "missing file".to_string());
        if found != expected_sha256 {
            issues.push(CheckpointCompatibilityIssue {
                field: "tokenizer.sha256".to_string(),
                expected: expected_sha256.to_string(),
                found,
            });
        }
    }

    Ok(CheckpointCompatibilityReport::new(issues))
}

pub fn validate_checkpoint_compatibility(
    metadata: &CheckpointMetadata,
    expected_shape: &CheckpointModelShape,
    tokenizer_path: Option<&Path>,
) -> anyhow::Result<()> {
    let report = checkpoint_compatibility_report(metadata, expected_shape, tokenizer_path)?;
    if !report.compatible {
        anyhow::bail!(report.to_string());
    }

    Ok(())
}

pub fn validate_checkpoint_compatibility_strict(
    checkpoint_path: &Path,
    metadata: &CheckpointMetadata,
    expected_shape: &CheckpointModelShape,
    tokenizer_path: &Path,
) -> anyhow::Result<()> {
    let mut report =
        checkpoint_compatibility_report(metadata, expected_shape, Some(tokenizer_path))?;

    if metadata.tokenizer.sha256.is_none() {
        report.issues.push(CheckpointCompatibilityIssue {
            field: "tokenizer.sha256".to_string(),
            expected: format!(
                "sha256 recorded in checkpoint metadata for tokenizer {}",
                tokenizer_path.display()
            ),
            found: "missing".to_string(),
        });
        report.compatible = false;
    }

    if !report.compatible {
        anyhow::bail!(
            "strict checkpoint metadata validation failed for checkpoint {} with tokenizer {}: {}",
            checkpoint_path.with_extension("mpk").display(),
            tokenizer_path.display(),
            report
        );
    }

    Ok(())
}

fn push_usize_issue(
    issues: &mut Vec<CheckpointCompatibilityIssue>,
    field: &str,
    expected: usize,
    found: usize,
) {
    if expected != found {
        issues.push(CheckpointCompatibilityIssue {
            field: field.to_string(),
            expected: expected.to_string(),
            found: found.to_string(),
        });
    }
}

pub fn load_model_with_metadata_validation<M, B, P>(
    model: M,
    path: P,
    expected_shape: &CheckpointModelShape,
    tokenizer_path: Option<&Path>,
    device: &B::Device,
) -> anyhow::Result<M>
where
    M: Module<B>,
    B: Backend,
    P: Into<PathBuf> + Clone,
{
    let path_buf: PathBuf = path.clone().into();
    if let Some(metadata) = load_checkpoint_metadata(&path_buf)? {
        validate_checkpoint_compatibility(&metadata, expected_shape, tokenizer_path)?;
    }

    load_model(model, path, device).map_err(anyhow::Error::msg)
}

pub fn load_model_with_strict_metadata_validation<M, B, P>(
    model: M,
    path: P,
    expected_shape: &CheckpointModelShape,
    tokenizer_path: &Path,
    device: &B::Device,
) -> anyhow::Result<M>
where
    M: Module<B>,
    B: Backend,
    P: Into<PathBuf> + Clone,
{
    let path_buf: PathBuf = path.clone().into();
    let metadata_path = metadata_path(&path_buf);
    let metadata = load_checkpoint_metadata(&path_buf)?.ok_or_else(|| {
        anyhow::anyhow!(
            "strict checkpoint metadata validation failed for checkpoint {}: missing metadata sidecar {}; runtime MiniGPT checkpoint loads require metadata before Burn model deserialization",
            path_buf.with_extension("mpk").display(),
            metadata_path.display()
        )
    })?;

    validate_checkpoint_compatibility_strict(&path_buf, &metadata, expected_shape, tokenizer_path)?;

    load_model(model, path, device).map_err(anyhow::Error::msg)
}

pub fn sha256_file_hex(path: &Path) -> anyhow::Result<Option<String>> {
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    let digest_hex: String = Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok(Some(digest_hex))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MiniGpt, TrivialModel};
    use burn::backend::Autodiff;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::{Int, Tensor};

    #[test]
    fn saves_and_loads_module_outputs() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let path =
            std::env::temp_dir().join(format!("rusty-gpt-trivial-model-{}", std::process::id()));
        let saved_path = path.with_extension("mpk");

        let model = TrivialModel::<TestBackend>::new(5, 3, &device);
        let input = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2], [3, 4, 0]], &device);
        let expected = model
            .forward(input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        save_model(model, &path).unwrap();
        let template = TrivialModel::<TestBackend>::new(5, 3, &device);
        let loaded = load_model(template, &path, &device).unwrap();
        let actual = loaded.forward(input).into_data().to_vec::<f32>().unwrap();

        assert_eq!(expected, actual);

        let _ = std::fs::remove_file(saved_path);
    }

    #[test]
    fn autodiff_checkpoint_loads_into_inference_backend() {
        type TrainBackend = Autodiff<NdArray<f32, i64>>;
        type InferenceBackend = NdArray<f32, i64>;

        let device = NdArrayDevice::Cpu;
        let path = std::env::temp_dir().join(format!(
            "rusty-gpt-minigpt-inference-load-{}",
            std::process::id()
        ));
        let saved_path = path.with_extension("mpk");

        let model = MiniGpt::<TrainBackend>::new(7, 8, 1, 4, 2, &device);
        save_model(model, &path).unwrap();

        let template = MiniGpt::<InferenceBackend>::new(7, 8, 1, 4, 2, &device);
        let loaded = load_model(template, &path, &device).unwrap();
        let input = Tensor::<InferenceBackend, 2, Int>::from_data([[0, 1, 2]], &device);

        assert_eq!([1, 3, 7], loaded.forward_tokens(input).shape().dims());

        let _ = std::fs::remove_file(saved_path);
    }

    #[test]
    fn saves_and_loads_checkpoint_metadata_sidecar() {
        let path = std::env::temp_dir().join(format!("rusty-gpt-metadata-{}", std::process::id()));
        let metadata_path = metadata_path(&path);
        let _ = std::fs::remove_file(&metadata_path);
        let metadata = CheckpointMetadata {
            version: 1,
            created_at_utc: "2026-05-14T00:00:00Z".to_string(),
            git_commit: Some("abc123".to_string()),
            input_source: "data/input.txt".to_string(),
            model_shape: CheckpointModelShape {
                vocab_size: 7,
                block_size: 4,
                embed_dim: 8,
                num_heads: 2,
                num_layers: 1,
            },
            tokenizer: CheckpointTokenizer {
                path: "checkpoints/tokenizer.json".to_string(),
                sha256: Some("hash".to_string()),
                vocab_size: 7,
            },
            training: CheckpointTrainingRun {
                backend: "cpu".to_string(),
                train_tokens: 90,
                value_tokens: 10,
                batch_size: 2,
                learning_rate: 1e-4,
                train_steps: 1,
                eval_interval: 0,
                grad_clip_norm: 1.0,
                prefetch_batches: 0,
            },
            final_metrics: CheckpointTrainingMetrics {
                final_value_loss: 1.25,
                final_perplexity: 3.49,
            },
            interrupted: false,
            interrupted_at_step: None,
            step: None,
            interval: None,
            completed_steps: 5,
        };

        save_checkpoint_metadata(&path, &metadata).unwrap();

        let loaded = load_checkpoint_metadata(&path).unwrap();
        // The sidecar round-trips `completed_steps` verbatim — this is the
        // field `--resume-from` reads to continue the step counter.
        assert_eq!(Some(&5), loaded.as_ref().map(|m| &m.completed_steps));
        assert_eq!(Some(metadata), loaded);

        let _ = std::fs::remove_file(metadata_path);
    }

    #[test]
    fn metadata_validation_rejects_shape_mismatch() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let path = std::env::temp_dir().join(format!(
            "rusty-gpt-metadata-mismatch-{}",
            std::process::id()
        ));
        let saved_path = path.with_extension("mpk");
        let metadata_path = metadata_path(&path);
        let _ = std::fs::remove_file(&saved_path);
        let _ = std::fs::remove_file(&metadata_path);

        save_model(MiniGpt::<TestBackend>::new(7, 8, 1, 4, 2, &device), &path).unwrap();
        save_checkpoint_metadata(
            &path,
            &CheckpointMetadata {
                version: 1,
                created_at_utc: "2026-05-14T00:00:00Z".to_string(),
                git_commit: None,
                input_source: "test".to_string(),
                model_shape: CheckpointModelShape {
                    vocab_size: 8,
                    block_size: 4,
                    embed_dim: 8,
                    num_heads: 2,
                    num_layers: 1,
                },
                tokenizer: CheckpointTokenizer {
                    path: "tokenizer.json".to_string(),
                    sha256: None,
                    vocab_size: 8,
                },
                training: CheckpointTrainingRun {
                    backend: "cpu".to_string(),
                    train_tokens: 10,
                    value_tokens: 2,
                    batch_size: 1,
                    learning_rate: 1e-4,
                    train_steps: 1,
                    eval_interval: 0,
                    grad_clip_norm: 1.0,
                    prefetch_batches: 0,
                },
                final_metrics: CheckpointTrainingMetrics {
                    final_value_loss: 1.0,
                    final_perplexity: 2.71,
                },
                interrupted: false,
                interrupted_at_step: None,
                step: None,
                interval: None,
                completed_steps: 0,
            },
        )
        .unwrap();

        let template = MiniGpt::<TestBackend>::new(7, 8, 1, 4, 2, &device);
        let err = load_model_with_metadata_validation(
            template,
            &path,
            &CheckpointModelShape {
                vocab_size: 7,
                block_size: 4,
                embed_dim: 8,
                num_heads: 2,
                num_layers: 1,
            },
            None,
            &device,
        )
        .expect_err("shape mismatch should fail");

        let message = err.to_string();
        assert!(message.contains("checkpoint metadata compatibility failed"));
        assert!(message.contains("model_shape.vocab_size expected 7, found 8"));
        assert!(message.contains("tokenizer.vocab_size expected 7, found 8"));

        let _ = std::fs::remove_file(saved_path);
        let _ = std::fs::remove_file(metadata_path);
    }

    #[test]
    fn strict_metadata_validation_rejects_missing_sidecar_before_raw_load() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let path = std::env::temp_dir().join(format!(
            "rusty-gpt-strict-missing-metadata-{}",
            std::process::id()
        ));
        let saved_path = path.with_extension("mpk");
        let metadata_path = metadata_path(&path);
        let tokenizer_path = path.with_extension("tokenizer.json");
        let _ = std::fs::remove_file(&saved_path);
        let _ = std::fs::remove_file(&metadata_path);
        let _ = std::fs::remove_file(&tokenizer_path);

        let template = MiniGpt::<TestBackend>::new(7, 8, 1, 4, 2, &device);
        let err = load_model_with_strict_metadata_validation(
            template,
            &path,
            &CheckpointModelShape {
                vocab_size: 7,
                block_size: 4,
                embed_dim: 8,
                num_heads: 2,
                num_layers: 1,
            },
            &tokenizer_path,
            &device,
        )
        .expect_err("strict runtime loads should reject checkpoints without metadata");

        let message = err.to_string();
        assert!(message.contains("missing metadata sidecar"));
        assert!(message.contains(&metadata_path.display().to_string()));
        assert!(message.contains("before Burn model deserialization"));

        let _ = std::fs::remove_file(saved_path);
        let _ = std::fs::remove_file(metadata_path);
        let _ = std::fs::remove_file(tokenizer_path);
    }

    #[test]
    fn strict_metadata_validation_rejects_missing_tokenizer_hash() {
        let checkpoint_path = std::env::temp_dir().join(format!(
            "rusty-gpt-strict-missing-tokenizer-sha-{}",
            std::process::id()
        ));
        let tokenizer_path = checkpoint_path.with_extension("tokenizer.json");
        std::fs::write(&tokenizer_path, br#"{"tokenizer":"current"}"#).unwrap();
        let metadata = compatible_test_metadata(7, None);

        let err = validate_checkpoint_compatibility_strict(
            &checkpoint_path,
            &metadata,
            &CheckpointModelShape {
                vocab_size: 7,
                block_size: 4,
                embed_dim: 8,
                num_heads: 2,
                num_layers: 1,
            },
            &tokenizer_path,
        )
        .expect_err("strict runtime loads should require tokenizer sha256 metadata");

        let message = err.to_string();
        assert!(message.contains("strict checkpoint metadata validation failed"));
        assert!(message.contains(&checkpoint_path.with_extension("mpk").display().to_string()));
        assert!(message.contains(&tokenizer_path.display().to_string()));
        assert!(message.contains("tokenizer.sha256 expected sha256 recorded"));
        assert!(message.contains("found missing"));

        let _ = std::fs::remove_file(tokenizer_path);
    }

    #[test]
    fn strict_metadata_validation_reports_tokenizer_hash_mismatch_with_context() {
        let checkpoint_path = std::env::temp_dir().join(format!(
            "rusty-gpt-strict-tokenizer-sha-mismatch-{}",
            std::process::id()
        ));
        let tokenizer_path = checkpoint_path.with_extension("tokenizer.json");
        std::fs::write(&tokenizer_path, br#"{"tokenizer":"saved"}"#).unwrap();
        let metadata = compatible_test_metadata(7, sha256_file_hex(&tokenizer_path).unwrap());
        std::fs::write(&tokenizer_path, br#"{"tokenizer":"runtime"}"#).unwrap();

        let err = validate_checkpoint_compatibility_strict(
            &checkpoint_path,
            &metadata,
            &CheckpointModelShape {
                vocab_size: 7,
                block_size: 4,
                embed_dim: 8,
                num_heads: 2,
                num_layers: 1,
            },
            &tokenizer_path,
        )
        .expect_err("strict runtime loads should reject tokenizer sha256 mismatches");

        let message = err.to_string();
        assert!(message.contains("strict checkpoint metadata validation failed"));
        assert!(message.contains(&checkpoint_path.with_extension("mpk").display().to_string()));
        assert!(message.contains(&tokenizer_path.display().to_string()));
        assert!(message.contains("tokenizer.sha256"));

        let _ = std::fs::remove_file(tokenizer_path);
    }

    #[test]
    fn compatibility_report_validates_tokenizer_hash_when_path_is_provided() {
        let tokenizer_path = std::env::temp_dir().join(format!(
            "rusty-gpt-tokenizer-compatibility-{}.json",
            std::process::id()
        ));
        std::fs::write(&tokenizer_path, br#"{"tokenizer":"old"}"#).unwrap();

        let metadata = CheckpointMetadata {
            version: 1,
            created_at_utc: "2026-05-14T00:00:00Z".to_string(),
            git_commit: None,
            input_source: "test".to_string(),
            model_shape: CheckpointModelShape {
                vocab_size: 7,
                block_size: 4,
                embed_dim: 8,
                num_heads: 2,
                num_layers: 1,
            },
            tokenizer: CheckpointTokenizer {
                path: tokenizer_path.display().to_string(),
                sha256: sha256_file_hex(&tokenizer_path).unwrap(),
                vocab_size: 7,
            },
            training: CheckpointTrainingRun {
                backend: "cpu".to_string(),
                train_tokens: 10,
                value_tokens: 2,
                batch_size: 1,
                learning_rate: 1e-4,
                train_steps: 1,
                eval_interval: 0,
                grad_clip_norm: 1.0,
                prefetch_batches: 0,
            },
            final_metrics: CheckpointTrainingMetrics {
                final_value_loss: 1.0,
                final_perplexity: 2.71,
            },
            interrupted: false,
            interrupted_at_step: None,
            step: None,
            interval: None,
            completed_steps: 0,
        };
        std::fs::write(&tokenizer_path, br#"{"tokenizer":"new"}"#).unwrap();

        let report = checkpoint_compatibility_report(
            &metadata,
            &CheckpointModelShape {
                vocab_size: 7,
                block_size: 4,
                embed_dim: 8,
                num_heads: 2,
                num_layers: 1,
            },
            Some(&tokenizer_path),
        )
        .unwrap();

        assert!(!report.compatible);
        assert_eq!(1, report.issues.len());
        assert_eq!("tokenizer.sha256", report.issues[0].field);
        assert_ne!(report.issues[0].expected, report.issues[0].found);

        let _ = std::fs::remove_file(tokenizer_path);
    }

    #[test]
    fn compatibility_report_accepts_matching_metadata() {
        let metadata = CheckpointMetadata {
            version: 1,
            created_at_utc: "2026-05-14T00:00:00Z".to_string(),
            git_commit: None,
            input_source: "test".to_string(),
            model_shape: CheckpointModelShape {
                vocab_size: 7,
                block_size: 4,
                embed_dim: 8,
                num_heads: 2,
                num_layers: 1,
            },
            tokenizer: CheckpointTokenizer {
                path: "tokenizer.json".to_string(),
                sha256: None,
                vocab_size: 7,
            },
            training: CheckpointTrainingRun {
                backend: "cpu".to_string(),
                train_tokens: 10,
                value_tokens: 2,
                batch_size: 1,
                learning_rate: 1e-4,
                train_steps: 1,
                eval_interval: 0,
                grad_clip_norm: 1.0,
                prefetch_batches: 0,
            },
            final_metrics: CheckpointTrainingMetrics {
                final_value_loss: 1.0,
                final_perplexity: 2.71,
            },
            interrupted: false,
            interrupted_at_step: None,
            step: None,
            interval: None,
            completed_steps: 0,
        };

        let report = checkpoint_compatibility_report(
            &metadata,
            &CheckpointModelShape {
                vocab_size: 7,
                block_size: 4,
                embed_dim: 8,
                num_heads: 2,
                num_layers: 1,
            },
            None,
        )
        .unwrap();

        assert!(report.compatible);
        assert!(report.issues.is_empty());
    }

    fn compatible_test_metadata(
        vocab_size: usize,
        tokenizer_sha256: Option<String>,
    ) -> CheckpointMetadata {
        CheckpointMetadata {
            version: 1,
            created_at_utc: "2026-05-14T00:00:00Z".to_string(),
            git_commit: None,
            input_source: "test".to_string(),
            model_shape: CheckpointModelShape {
                vocab_size,
                block_size: 4,
                embed_dim: 8,
                num_heads: 2,
                num_layers: 1,
            },
            tokenizer: CheckpointTokenizer {
                path: "tokenizer.json".to_string(),
                sha256: tokenizer_sha256,
                vocab_size,
            },
            training: CheckpointTrainingRun {
                backend: "cpu".to_string(),
                train_tokens: 10,
                value_tokens: 2,
                batch_size: 1,
                learning_rate: 1e-4,
                train_steps: 1,
                eval_interval: 0,
                grad_clip_norm: 1.0,
                prefetch_batches: 0,
            },
            final_metrics: CheckpointTrainingMetrics {
                final_value_loss: 1.0,
                final_perplexity: 2.71,
            },
            interrupted: false,
            interrupted_at_step: None,
            step: None,
            interval: None,
            completed_steps: 0,
        }
    }

    /// INVARIANT: legacy dense sidecars written before `completed_steps`
    /// existed must keep loading. This fixture is a verbatim pre-S3-T2
    /// sidecar (no `completed_steps` key). `#[serde(default)]` must fill it
    /// with `0` rather than failing deserialization, so an old checkpoint
    /// resumes from scratch instead of refusing to load.
    #[test]
    fn legacy_sidecar_without_completed_steps_defaults_to_zero() {
        let legacy_json = r#"{
            "version": 1,
            "created_at_utc": "2026-05-14T00:00:00Z",
            "git_commit": "deadbee",
            "input_source": "data/input.txt",
            "model_shape": {
                "vocab_size": 7,
                "block_size": 4,
                "embed_dim": 8,
                "num_heads": 2,
                "num_layers": 1
            },
            "tokenizer": {
                "path": "checkpoints/tokenizer.json",
                "sha256": "hash",
                "vocab_size": 7
            },
            "training": {
                "backend": "cpu",
                "train_tokens": 90,
                "value_tokens": 10,
                "batch_size": 2,
                "learning_rate": 0.0001,
                "train_steps": 10,
                "eval_interval": 0,
                "grad_clip_norm": 1.0,
                "prefetch_batches": 0
            },
            "final_metrics": {
                "final_value_loss": 1.25,
                "final_perplexity": 3.49
            }
        }"#;

        let metadata: CheckpointMetadata =
            serde_json::from_str(legacy_json).expect("legacy sidecar must still deserialize");

        assert_eq!(0, metadata.completed_steps);
        // The other optional fields also default correctly for a legacy file.
        assert!(!metadata.interrupted);
        assert_eq!(None, metadata.step);
        assert_eq!(10, metadata.training.train_steps);
    }
}
