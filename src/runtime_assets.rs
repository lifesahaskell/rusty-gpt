use crate::runtime_config::ModelChoice;
use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use rusty_gpt::loader::huggingface;
use rusty_gpt::loader::{DEFAULT_MAX_LOCAL_INPUT_BYTES, InputSource};
use rusty_gpt::model::persistence::{
    CheckpointModelShape, load_model_with_strict_metadata_validation,
};
use rusty_gpt::model::{MiniGpt, MoeGpt};
use rusty_gpt::observability::{EventLogger, RuntimeEvent};
use rusty_gpt::tokenizer::RuntimeTokenizer;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub(crate) const DEFAULT_BPE_TOKENIZER_PATH: &str = "checkpoints/tokenizer.json";
pub(crate) const BPE_TOKENIZER_ENV: &str = "RUSTY_GPT_BPE_TOKENIZER";
pub(crate) const DEFAULT_CHECKPOINT_DIR: &str = "checkpoints";

pub(crate) fn load_minigpt_checkpoint<B: Backend>(
    template: MiniGpt<B>,
    checkpoint_path: &Path,
    device: &B::Device,
    logger: &EventLogger,
) -> Result<MiniGpt<B>> {
    let started_at = Instant::now();
    let expected_shape = CheckpointModelShape {
        kind: Some("minigpt".to_string()),
        vocab_size: template.vocab_size(),
        block_size: template.block_size(),
        embed_dim: template.d_model(),
        num_heads: template.num_heads(),
        num_layers: template.num_layers(),
        num_experts: 0,
        moe_top_k: 0,
    };
    let tokenizer_path = minigpt_tokenizer_path();
    let model = load_model_with_strict_metadata_validation(
        template,
        checkpoint_path,
        &expected_shape,
        Path::new(&tokenizer_path),
        device,
    )
    .with_context(|| {
        format!(
            "failed to load minigpt checkpoint from {:?}",
            checkpoint_path.with_extension("mpk")
        )
    })?;
    logger.log(RuntimeEvent::CheckpointLoaded {
        path: checkpoint_path.with_extension("mpk").display().to_string(),
        elapsed_ms: started_at.elapsed().as_millis(),
    });

    Ok(model)
}

pub(crate) fn load_moegpt_checkpoint<B: Backend>(
    template: MoeGpt<B>,
    checkpoint_path: &Path,
    device: &B::Device,
    logger: &EventLogger,
) -> Result<MoeGpt<B>> {
    let started_at = Instant::now();
    let expected_shape = CheckpointModelShape {
        kind: Some("moe-gpt".to_string()),
        vocab_size: template.vocab_size(),
        block_size: template.block_size(),
        embed_dim: template.d_model(),
        num_heads: template.num_heads(),
        num_layers: template.num_layers(),
        num_experts: template.num_experts(),
        moe_top_k: template.moe_top_k(),
    };
    let tokenizer_path = minigpt_tokenizer_path();
    let model = load_model_with_strict_metadata_validation(
        template,
        checkpoint_path,
        &expected_shape,
        Path::new(&tokenizer_path),
        device,
    )
    .with_context(|| {
        format!(
            "failed to load moe-gpt checkpoint from {:?}",
            checkpoint_path.with_extension("mpk")
        )
    })?;
    logger.log(RuntimeEvent::CheckpointLoaded {
        path: checkpoint_path.with_extension("mpk").display().to_string(),
        elapsed_ms: started_at.elapsed().as_millis(),
    });

    Ok(model)
}

pub(crate) fn tokenizer_for_model(
    text: &str,
    model_choice: ModelChoice,
) -> Result<RuntimeTokenizer> {
    if model_choice.uses_bpe_tokenizer() {
        load_minigpt_tokenizer()
    } else {
        Ok(RuntimeTokenizer::char_from_text(text))
    }
}

pub(crate) fn load_minigpt_tokenizer() -> Result<RuntimeTokenizer> {
    let tokenizer_path = minigpt_tokenizer_path();
    RuntimeTokenizer::load_bpe(Path::new(&tokenizer_path)).with_context(|| {
        format!(
            "failed to load MiniGPT BPE tokenizer from {tokenizer_path}; train one with `cargo run --bin train-tokenizer -- --corpus data/fafolang.txt --vocab-size 2048 --output {DEFAULT_BPE_TOKENIZER_PATH}`"
        )
    })
}

pub(crate) fn minigpt_tokenizer_path() -> String {
    env::var(BPE_TOKENIZER_ENV).unwrap_or_else(|_| DEFAULT_BPE_TOKENIZER_PATH.to_string())
}

pub(crate) fn latest_checkpoint_path(checkpoint_dir: &Path) -> Result<PathBuf> {
    let mut latest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(checkpoint_dir)
        .with_context(|| format!("failed to read checkpoint directory {:?}", checkpoint_dir))?
    {
        let entry = entry.with_context(|| {
            format!(
                "failed to read checkpoint directory entry in {:?}",
                checkpoint_dir
            )
        })?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("mpk") {
            continue;
        }

        let modified = entry
            .metadata()
            .with_context(|| format!("failed to read checkpoint metadata for {:?}", path))?
            .modified()
            .with_context(|| format!("failed to read checkpoint modified time for {:?}", path))?;
        if latest
            .as_ref()
            .map(|(latest_modified, _)| modified > *latest_modified)
            .unwrap_or(true)
        {
            latest = Some((modified, path.with_extension("")));
        }
    }

    latest
        .map(|(_, path)| path)
        .with_context(|| format!("no .mpk checkpoints found in {:?}", checkpoint_dir))
}

pub(crate) fn load_input_text(source: &InputSource) -> Result<String> {
    load_input_text_with_max_bytes(source, DEFAULT_MAX_LOCAL_INPUT_BYTES)
}

pub(crate) fn load_input_text_with_max_bytes(
    source: &InputSource,
    max_local_bytes: u64,
) -> Result<String> {
    match source {
        InputSource::Local(path) => {
            // Enforce the size cap via metadata before we read the file body.
            source
                .validate_local_size(max_local_bytes)
                .map_err(|err| anyhow::anyhow!(err.to_string()))?;
            fs::read_to_string(path)
                .with_context(|| format!("failed to read input text from {:?}", path))
        }
        InputSource::HuggingFace { .. } => {
            // The Hugging Face loader is the existing implementation; we feed
            // it the canonical, validated form rendered by `InputSource::display`.
            let canonical = source.display();
            match huggingface::load_text_from_uri(&canonical)? {
                Some(text) => Ok(text),
                None => Err(anyhow::anyhow!(
                    "Hugging Face loader returned no text for {canonical}"
                )),
            }
        }
    }
}
