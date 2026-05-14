use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use rusty_gpt::loader::huggingface;
use rusty_gpt::tokenizer::Tokenizer;
use rusty_gpt::tokenizer::bpe::BpeTrainer;
use serde::Serialize;

#[derive(Serialize)]
struct TokenizerSavedEvent {
    event: &'static str,
    vocab_size: usize,
    output: String,
}

#[derive(Debug, PartialEq, Eq)]
struct Config {
    corpus: String,
    vocab_size: usize,
    output: PathBuf,
}

fn main() -> Result<()> {
    let config = parse_args(env::args().skip(1))?;
    let corpus = load_corpus_text(&config.corpus)?;
    let tokenizer = BpeTrainer::new(config.vocab_size).train_with_observer(&corpus, |event| {
        log_json(&event);
    });

    if let Some(parent) = config.output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {:?}", parent))?;
    }

    tokenizer
        .save(&config.output)
        .with_context(|| format!("failed to save tokenizer to {:?}", config.output))?;
    log_json(&TokenizerSavedEvent {
        event: "tokenizer_saved",
        vocab_size: tokenizer.vocab_size(),
        output: config.output.display().to_string(),
    });

    Ok(())
}

fn log_json(event: &impl Serialize) {
    match serde_json::to_string(event) {
        Ok(line) => println!("{line}"),
        Err(err) => eprintln!(r#"{{"event":"structured_log_error","error":"{err}"}}"#),
    }
}

fn parse_args<I, S>(args: I) -> Result<Config>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let args: Vec<String> = args
        .into_iter()
        .map(|arg| arg.as_ref().to_string())
        .collect();
    let mut corpus = None;
    let mut vocab_size = None;
    let mut output = None;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--corpus" => {
                let value = args
                    .get(index + 1)
                    .context("--corpus requires a path to a text file")?;
                corpus = Some(value.to_string());
                index += 2;
            }
            "--vocab-size" => {
                let value = args
                    .get(index + 1)
                    .context("--vocab-size requires an integer")?;
                vocab_size = Some(
                    value
                        .parse()
                        .with_context(|| format!("invalid --vocab-size value: {value}"))?,
                );
                index += 2;
            }
            "--output" => {
                let value = args
                    .get(index + 1)
                    .context("--output requires a tokenizer JSON path")?;
                output = Some(PathBuf::from(value));
                index += 2;
            }
            "-h" | "--help" => {
                bail!("Usage: train-tokenizer --corpus <path> --vocab-size <n> --output <path>");
            }
            other => bail!("unsupported argument: {other}"),
        }
    }

    let config = Config {
        corpus: corpus.context("--corpus is required")?,
        vocab_size: vocab_size.context("--vocab-size is required")?,
        output: output.context("--output is required")?,
    };
    if config.vocab_size < 256 {
        bail!("--vocab-size must be at least 256");
    }

    Ok(config)
}

fn load_corpus_text(source: &str) -> Result<String> {
    if let Some(text) = huggingface::load_text_from_uri(source)? {
        return Ok(text);
    }

    fs::read_to_string(source).with_context(|| format!("failed to read corpus from {source}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_required_args() {
        let config = parse_args([
            "--corpus",
            "data/fafolang.txt",
            "--vocab-size",
            "2048",
            "--output",
            "checkpoints/tokenizer.json",
        ])
        .unwrap();

        assert_eq!(
            Config {
                corpus: "data/fafolang.txt".to_string(),
                vocab_size: 2048,
                output: PathBuf::from("checkpoints/tokenizer.json"),
            },
            config
        );
    }

    #[test]
    fn rejects_too_small_vocab() {
        let err = parse_args([
            "--corpus",
            "data/fafolang.txt",
            "--vocab-size",
            "255",
            "--output",
            "checkpoints/tokenizer.json",
        ])
        .expect_err("vocab below byte vocabulary should fail");

        assert!(
            err.to_string()
                .contains("--vocab-size must be at least 256")
        );
    }
}
