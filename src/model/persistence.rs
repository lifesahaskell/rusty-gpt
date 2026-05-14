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

pub fn load_model_with_metadata_validation<M, B, P>(
    model: M,
    path: P,
    expected_shape: &CheckpointModelShape,
    device: &B::Device,
) -> anyhow::Result<M>
where
    M: Module<B>,
    B: Backend,
    P: Into<PathBuf> + Clone,
{
    let path_buf: PathBuf = path.clone().into();
    if let Some(metadata) = load_checkpoint_metadata(&path_buf)?
        && metadata.model_shape != *expected_shape
    {
        anyhow::bail!(
            "checkpoint metadata shape mismatch: expected {:?}, found {:?}",
            expected_shape,
            metadata.model_shape
        );
    }

    load_model(model, path, device).map_err(anyhow::Error::msg)
}

pub fn sha256_file_hex(path: &Path) -> anyhow::Result<Option<String>> {
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    let digest = Sha256::digest(bytes);
    Ok(Some(format!("{digest:x}")))
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
        };

        save_checkpoint_metadata(&path, &metadata).unwrap();

        assert_eq!(Some(metadata), load_checkpoint_metadata(&path).unwrap());

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
            &device,
        )
        .expect_err("shape mismatch should fail");

        assert!(
            err.to_string()
                .contains("checkpoint metadata shape mismatch")
        );

        let _ = std::fs::remove_file(saved_path);
        let _ = std::fs::remove_file(metadata_path);
    }
}
