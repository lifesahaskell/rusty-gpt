use std::fmt;
use std::sync::Arc;

use anyhow::{Result, bail};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Plain,
    Json,
}

impl LogFormat {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "plain" => Ok(Self::Plain),
            "json" => Ok(Self::Json),
            other => bail!("unsupported log format '{other}'; expected plain or json"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Json => "json",
        }
    }
}

impl fmt::Display for LogFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

type LogSink = Arc<dyn Fn(String) + Send + Sync>;

#[derive(Clone)]
pub struct EventLogger {
    format: LogFormat,
    sink: LogSink,
}

impl EventLogger {
    pub fn stdout(format: LogFormat) -> Self {
        Self {
            format,
            sink: Arc::new(|line| println!("{line}")),
        }
    }

    pub fn with_sink(format: LogFormat, sink: impl Fn(String) + Send + Sync + 'static) -> Self {
        Self {
            format,
            sink: Arc::new(sink),
        }
    }

    pub fn format(&self) -> LogFormat {
        self.format
    }

    pub fn log(&self, event: RuntimeEvent) {
        (self.sink)(event.render(self.format));
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum RuntimeEvent {
    AppConfigured {
        backend: String,
        model: String,
        input_path: String,
        tokenizer_path: String,
        checkpoint_path: String,
        log_format: LogFormat,
        serve: bool,
        benchmark_generation: bool,
    },
    RuntimeBatchPrepared {
        vocab_size: usize,
        input_chars: usize,
        encoded_tokens: usize,
        batch_size: usize,
        block_size: usize,
        dropout: f64,
    },
    ModelForwardCompleted {
        model: String,
        logits_shape: [usize; 3],
        input_shape: [usize; 2],
        target_shape: [usize; 2],
    },
    ServerStarted {
        addr: String,
        backend: String,
        vocab_size: usize,
        block_size: usize,
    },
    TrainingStarted {
        backend: String,
        model: String,
        vocab_size: usize,
        batch_size: usize,
        block_size: usize,
        total_steps: usize,
    },
    TrainingProgress {
        backend: String,
        model: String,
        step: usize,
        total_steps: usize,
        training_loss: f64,
        value_loss: f64,
        value_perplexity: f64,
        learning_rate: f64,
        elapsed_ms: u128,
        tokens_per_second: f64,
        steps_per_second: f64,
        step_ms_mean: f64,
    },
    TrainingCompleted {
        backend: String,
        model: String,
        total_steps: usize,
        elapsed_ms: u128,
        final_value_loss: f64,
        final_perplexity: f64,
    },
    CheckpointLoaded {
        path: String,
        elapsed_ms: u128,
    },
    CheckpointSaved {
        path: String,
        elapsed_ms: u128,
    },
    GenerateRequestAccepted {
        max_tokens: usize,
        temperature: f32,
        prompt_chars: usize,
    },
    GenerateRequestRejected {
        status: u16,
        reason: String,
        elapsed_ms: u128,
    },
    GenerateRequestCompleted {
        status: u16,
        prompt_tokens: usize,
        generated_tokens: usize,
        elapsed_ms: u128,
    },
    BenchmarkSkipped {
        prompt_len: usize,
        gen_len: usize,
        reason: String,
    },
    BenchmarkResult {
        prompt_len: usize,
        gen_len: usize,
        warmups: usize,
        iterations: usize,
        naive: BenchmarkStats,
        cached: BenchmarkStats,
        speedup: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchmarkStats {
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub tokens_per_second: f64,
}

impl RuntimeEvent {
    fn render(&self, format: LogFormat) -> String {
        match format {
            LogFormat::Plain => self.to_plain(),
            LogFormat::Json => serde_json::to_string(self).unwrap_or_else(|err| {
                format!(r#"{{"event":"structured_log_error","error":"{err}"}}"#)
            }),
        }
    }

    fn to_plain(&self) -> String {
        match self {
            Self::AppConfigured {
                backend,
                model,
                input_path,
                tokenizer_path,
                checkpoint_path,
                log_format,
                serve,
                benchmark_generation,
            } => format!(
                "Configured app: backend={backend}, model={model}, input={input_path}, tokenizer={tokenizer_path}, checkpoint={checkpoint_path}, log_format={log_format}, serve={serve}, benchmark_generation={benchmark_generation}"
            ),
            Self::RuntimeBatchPrepared {
                vocab_size,
                input_chars,
                encoded_tokens,
                batch_size,
                block_size,
                dropout,
            } => format!(
                "Prepared runtime batch: vocab_size={vocab_size}, input_chars={input_chars}, encoded_tokens={encoded_tokens}, batch_size={batch_size}, block_size={block_size}, dropout={dropout}"
            ),
            Self::ModelForwardCompleted {
                model,
                logits_shape,
                input_shape,
                target_shape,
            } => format!(
                "{model} forward pass: logits_shape={logits_shape:?}, x_shape={input_shape:?}, y_shape={target_shape:?}"
            ),
            Self::ServerStarted {
                addr,
                backend,
                vocab_size,
                block_size,
            } => format!(
                "Serving GPT API on http://{addr} (backend={backend}, vocab_size={vocab_size}, block_size={block_size})"
            ),
            Self::TrainingStarted {
                backend,
                model,
                vocab_size,
                batch_size,
                block_size,
                total_steps,
            } => format!(
                "Training {model} model on {backend}: vocab_size={vocab_size}, batch_size={batch_size}, block_size={block_size}, total_steps={total_steps}"
            ),
            Self::TrainingProgress {
                step,
                total_steps,
                training_loss,
                value_loss,
                value_perplexity,
                learning_rate,
                elapsed_ms,
                tokens_per_second,
                steps_per_second,
                step_ms_mean,
                ..
            } => format!(
                "Step {step}/{total_steps}: training loss = {training_loss:.4}, value loss = {value_loss:.4}, perplexity={value_perplexity:.4}, learning_rate={learning_rate:.6}, elapsed={elapsed_ms}ms, tokens_per_second={tokens_per_second:.2}, steps_per_second={steps_per_second:.4}, step_ms_mean={step_ms_mean:.2}"
            ),
            Self::TrainingCompleted {
                backend,
                model,
                total_steps,
                elapsed_ms,
                final_value_loss,
                final_perplexity,
            } => format!(
                "Completed {model} training on {backend}: total_steps={total_steps}, final_value_loss={final_value_loss:.4}, final_perplexity={final_perplexity:.4}, elapsed={elapsed_ms}ms"
            ),
            Self::CheckpointLoaded { path, elapsed_ms } => {
                format!("Loaded minigpt checkpoint from {path} in {elapsed_ms}ms")
            }
            Self::CheckpointSaved { path, elapsed_ms } => {
                format!("Saved minigpt checkpoint to {path} in {elapsed_ms}ms")
            }
            Self::GenerateRequestAccepted {
                max_tokens,
                temperature,
                prompt_chars,
            } => format!(
                "Accepted generation request: prompt_chars={prompt_chars}, max_tokens={max_tokens}, temperature={temperature}"
            ),
            Self::GenerateRequestRejected {
                status,
                reason,
                elapsed_ms,
            } => format!(
                "Rejected generation request: status={status}, reason={reason}, elapsed={elapsed_ms}ms"
            ),
            Self::GenerateRequestCompleted {
                status,
                prompt_tokens,
                generated_tokens,
                elapsed_ms,
            } => format!(
                "Completed generation request: status={status}, prompt_tokens={prompt_tokens}, generated_tokens={generated_tokens}, elapsed={elapsed_ms}ms"
            ),
            Self::BenchmarkSkipped {
                prompt_len,
                gen_len,
                reason,
            } => format!("Benchmark skipped: prompt={prompt_len}, gen={gen_len}, reason={reason}"),
            Self::BenchmarkResult {
                prompt_len,
                gen_len,
                warmups,
                iterations,
                naive,
                cached,
                speedup,
            } => format!(
                "Benchmark prompt={prompt_len}, gen={gen_len}, warmups={warmups}, iterations={iterations}: naive mean {:.3}ms ({:.2} tok/s), cached mean {:.3}ms ({:.2} tok/s), speedup {:.2}x",
                naive.mean_ms,
                naive.tokens_per_second,
                cached.mean_ms,
                cached.tokens_per_second,
                speedup
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn parses_supported_log_formats() {
        assert_eq!(LogFormat::Plain, LogFormat::parse("plain").unwrap());
        assert_eq!(LogFormat::Json, LogFormat::parse("json").unwrap());
    }

    #[test]
    fn rejects_unsupported_log_format() {
        let err = LogFormat::parse("xml").unwrap_err();

        assert!(err.to_string().contains("unsupported log format"));
    }

    #[test]
    fn json_logger_emits_structured_event() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&lines);
        let logger = EventLogger::with_sink(LogFormat::Json, move |line| {
            captured.lock().unwrap().push(line);
        });

        logger.log(RuntimeEvent::GenerateRequestCompleted {
            status: 200,
            prompt_tokens: 3,
            generated_tokens: 5,
            elapsed_ms: 7,
        });

        let lines = lines.lock().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!("generate_request_completed", parsed["event"]);
        assert_eq!(200, parsed["status"]);
        assert_eq!(3, parsed["prompt_tokens"]);
        assert_eq!(5, parsed["generated_tokens"]);
        assert_eq!(7, parsed["elapsed_ms"]);
    }

    #[test]
    fn json_training_progress_includes_throughput_fields() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&lines);
        let logger = EventLogger::with_sink(LogFormat::Json, move |line| {
            captured.lock().unwrap().push(line);
        });

        logger.log(RuntimeEvent::TrainingProgress {
            backend: "cuda".to_string(),
            model: "minigpt".to_string(),
            step: 9,
            total_steps: 100,
            training_loss: 1.25,
            value_loss: 1.5,
            value_perplexity: 4.481689,
            learning_rate: 5e-4,
            elapsed_ms: 250,
            tokens_per_second: 8192.0,
            steps_per_second: 40.0,
            step_ms_mean: 25.0,
        });

        let lines = lines.lock().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!("training_progress", parsed["event"]);
        assert_eq!(4.481689, parsed["value_perplexity"]);
        assert_eq!(5e-4, parsed["learning_rate"]);
        assert_eq!(8192.0, parsed["tokens_per_second"]);
        assert_eq!(40.0, parsed["steps_per_second"]);
        assert_eq!(25.0, parsed["step_ms_mean"]);
    }
}
