use anyhow::Result;
#[cfg(feature = "cuda")]
use burn::backend::Autodiff;
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "cuda")]
use burn::backend::cuda::CudaDevice;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
#[cfg(test)]
use rusty_gpt::observability::LogFormat;
use rusty_gpt::observability::{EventLogger, RuntimeEvent};
use rusty_gpt::server::ServerLimits;
#[cfg(test)]
use rusty_gpt::utils::BenchmarkConfig;
use std::env;
#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::net::SocketAddr;
#[cfg(test)]
use std::path::PathBuf;

mod runtime_assets;
mod runtime_config;
mod runtime_orchestration;
mod runtime_training;

#[cfg(test)]
use runtime_assets::{BPE_TOKENIZER_ENV, latest_checkpoint_path};
use runtime_assets::{load_input_text, minigpt_tokenizer_path};
#[cfg(test)]
use runtime_config::{
    BATCH_SIZE, BLOCK_SIZE, DEFAULT_INPUT_PATH, DEFAULT_MINIGPT_CHECKPOINT_PATH,
    DEFAULT_SERVER_ADDR, DROPOUT, EMBED_DIM, EVAL_INTERVAL, GENERATE_TOKENS, HEAD_DIM,
    LEARNING_RATE, MINIGPT_GRAD_CLIP_NORM, NUM_HEADS, NUM_LAYERS, PREFETCH_BATCHES, TRAIN_STEPS,
    parse_runtime_config,
};
use runtime_config::{BackendChoice, RuntimeEnv, parse_runtime_config_with_checkpoint};
#[cfg(feature = "cuda")]
use runtime_orchestration::run_demo;
use runtime_orchestration::{
    CpuDemoOptions, ServerRuntimeOptions, run_cpu_demo, run_http_server_with_runtime,
};
#[cfg(test)]
use runtime_training::split_training_and_value_tokens;
#[cfg(any(test, feature = "cuda"))]
use runtime_training::{TrainingDemoOptions, run_training_demo};

fn main() -> Result<()> {
    let config =
        parse_runtime_config_with_checkpoint(env::args().skip(1), RuntimeEnv::from_process_env())?;
    let text = load_input_text(&config.input_source)?;
    let hyperparameters = config.hyperparameters;
    let logger = EventLogger::stdout(config.log_format);
    let input_display = config.input_source.display();
    logger.log(RuntimeEvent::AppConfigured {
        backend: config.backend.label().to_string(),
        model: config.model.label().to_string(),
        input_path: input_display.clone(),
        tokenizer_path: minigpt_tokenizer_path(),
        checkpoint_path: config.checkpoint_path.display().to_string(),
        log_format: config.log_format,
        serve: config.serve,
        benchmark_generation: config.benchmark_generation,
    });

    if config.serve {
        let limits = ServerLimits {
            max_prompt_bytes: config.max_prompt_bytes,
            max_output_tokens: config.max_output_tokens,
            rate_limit_rps: config.rate_limit_rps,
            rate_limit_burst: config.rate_limit_burst,
            max_train_steps: config.max_train_steps,
            max_train_learning_rate: config.max_train_learning_rate,
        };
        return match config.backend {
            BackendChoice::Cpu => run_http_server_with_runtime::<NdArray<f32, i64>>(
                &text,
                hyperparameters,
                ServerRuntimeOptions {
                    model_choice: config.model,
                    server_addr: config.server_addr,
                    checkpoint_path: &config.checkpoint_path,
                    load_checkpoint_enabled: config.load_checkpoint,
                    load_latest_checkpoint_enabled: config.load_latest_checkpoint,
                    backend_label: "cpu",
                    logger,
                    limits,
                    input_source: &input_display,
                },
                &NdArrayDevice::Cpu,
            ),
            #[cfg(feature = "cuda")]
            BackendChoice::Cuda => run_http_server_with_runtime::<Cuda>(
                &text,
                hyperparameters,
                ServerRuntimeOptions {
                    model_choice: config.model,
                    server_addr: config.server_addr,
                    checkpoint_path: &config.checkpoint_path,
                    load_checkpoint_enabled: config.load_checkpoint,
                    load_latest_checkpoint_enabled: config.load_latest_checkpoint,
                    backend_label: "cuda",
                    logger,
                    limits,
                    input_source: &input_display,
                },
                &CudaDevice::default(),
            ),
        };
    }

    match config.backend {
        BackendChoice::Cpu => run_cpu_demo(
            &text,
            hyperparameters,
            CpuDemoOptions {
                model_choice: config.model,
                interactive: config.interactive,
                benchmark_generation: config.benchmark_generation,
                benchmark_config: &config.benchmark_config,
                logger,
                checkpoint_path: &config.checkpoint_path,
                input_source: &input_display,
                resume_from: config.resume_from.as_deref(),
            },
        ),
        #[cfg(feature = "cuda")]
        BackendChoice::Cuda => {
            if config.interactive {
                anyhow::bail!("interactive generation currently requires --backend cpu");
            }
            let device = CudaDevice::default();
            run_demo::<Cuda>(&text, hyperparameters, config.model, &device, &logger)?;
            run_training_demo::<Autodiff<Cuda>>(
                &text,
                hyperparameters,
                config.model,
                &device,
                &config.checkpoint_path,
                TrainingDemoOptions {
                    backend_label: "cuda",
                    logger,
                    benchmark_generation: config.benchmark_generation,
                    benchmark_config: config.benchmark_config,
                    input_source: input_display.clone(),
                    resume_from: config.resume_from.clone(),
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_config::{Hyperparameters, ModelChoice};
    use burn::backend::Autodiff;
    use std::sync::Mutex;

    // ponytail: cargo test runs in parallel within one process, so tests that
    // mutate real process env vars (RUSTY_GPT_*) race each other. Serialize
    // them on this lock instead of adding a test harness; add more tests here
    // if a future one starts touching process env.
    static ENV_MUTATION_LOCK: Mutex<()> = Mutex::new(());

    fn lock_env() -> std::sync::MutexGuard<'static, ()> {
        ENV_MUTATION_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn backend_defaults_to_cpu() {
        let config = parse_runtime_config(Vec::<String>::new(), None, None, None).unwrap();

        assert_eq!(BackendChoice::Cpu, config.backend);
        assert_eq!(ModelChoice::MiniGpt, config.model);
        assert_eq!(PathBuf::from(DEFAULT_INPUT_PATH), config.input_path);
        assert_eq!(
            PathBuf::from(DEFAULT_MINIGPT_CHECKPOINT_PATH),
            config.checkpoint_path
        );
        assert!(!config.serve);
        assert!(!config.benchmark_generation);
        assert!(!config.load_checkpoint);
        assert!(!config.load_latest_checkpoint);
        assert_eq!(LogFormat::Plain, config.log_format);
        assert_eq!(BenchmarkConfig::default(), config.benchmark_config);
        assert_eq!(
            DEFAULT_SERVER_ADDR.parse::<SocketAddr>().unwrap(),
            config.server_addr
        );
    }

    #[test]
    fn log_format_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--log-format".to_string(), "json".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(LogFormat::Json, config.log_format);
    }

    #[test]
    fn log_format_arg_takes_precedence_over_env() {
        let config = parse_runtime_config_with_checkpoint(
            ["--log-format".to_string(), "plain".to_string()],
            RuntimeEnv {
                log_format: Some("json".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .unwrap();

        assert_eq!(LogFormat::Plain, config.log_format);
    }

    #[test]
    fn invalid_log_format_returns_clear_error() {
        let err = parse_runtime_config(
            ["--log-format".to_string(), "yaml".to_string()],
            None,
            None,
            None,
        )
        .expect_err("invalid log format should fail");

        assert!(err.to_string().contains("unsupported log format"));
    }

    #[test]
    fn benchmark_config_can_be_selected_from_args() {
        let config = parse_runtime_config(
            [
                "--benchmark-prompt-lens".to_string(),
                "2,4".to_string(),
                "--benchmark-gen-lens".to_string(),
                "1,3".to_string(),
                "--benchmark-warmups".to_string(),
                "2".to_string(),
                "--benchmark-iterations".to_string(),
                "7".to_string(),
            ],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(vec![2, 4], config.benchmark_config.prompt_lens);
        assert_eq!(vec![1, 3], config.benchmark_config.gen_lens);
        assert_eq!(2, config.benchmark_config.warmups);
        assert_eq!(7, config.benchmark_config.iterations);
    }

    #[test]
    fn invalid_benchmark_config_returns_clear_error() {
        let err = parse_runtime_config(
            ["--benchmark-iterations".to_string(), "0".to_string()],
            None,
            None,
            None,
        )
        .expect_err("zero iterations should fail");

        assert!(
            err.to_string()
                .contains("benchmark iterations must be greater than zero")
        );
    }

    #[test]
    fn hyperparameters_default_to_main_constants() {
        let hyperparameters = Hyperparameters::default();

        assert_eq!(BLOCK_SIZE, hyperparameters.block_size);
        assert_eq!(BATCH_SIZE, hyperparameters.batch_size);
        assert_eq!(EMBED_DIM, hyperparameters.embed_dim);
        assert_eq!(NUM_HEADS, hyperparameters.num_heads);
        assert_eq!(HEAD_DIM, hyperparameters.head_dim);
        assert_eq!(NUM_LAYERS, hyperparameters.num_layers);
        assert_eq!(DROPOUT, hyperparameters.dropout);
        assert_eq!(LEARNING_RATE, hyperparameters.learning_rate);
        assert_eq!(TRAIN_STEPS, hyperparameters.train_steps);
        assert_eq!(EVAL_INTERVAL, hyperparameters.eval_interval);
        assert_eq!(GENERATE_TOKENS, hyperparameters.generate_tokens);
        assert_eq!(
            MINIGPT_GRAD_CLIP_NORM,
            hyperparameters.minigpt_grad_clip_norm
        );
        assert_eq!(PREFETCH_BATCHES, hyperparameters.prefetch_batches);
    }

    #[test]
    fn hyperparameters_can_override_training_settings_from_env() {
        let _guard = lock_env();
        // SAFETY: This unit test mutates process environment while not relying on
        // other environment-sensitive code in parallel.
        unsafe {
            env::set_var("RUSTY_GPT_TRAIN_STEPS", "7");
            env::set_var("RUSTY_GPT_EVAL_INTERVAL", "3");
            env::set_var("RUSTY_GPT_GENERATE_TOKENS", "11");
            env::set_var("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM", "0.5");
            env::set_var("RUSTY_GPT_PREFETCH_BATCHES", "4");
        }

        let hyperparameters = Hyperparameters::from_env().unwrap();

        assert_eq!(7, hyperparameters.train_steps);
        assert_eq!(3, hyperparameters.eval_interval);
        assert_eq!(11, hyperparameters.generate_tokens);
        assert_eq!(0.5, hyperparameters.minigpt_grad_clip_norm);
        assert_eq!(4, hyperparameters.prefetch_batches);

        // SAFETY: See note above.
        unsafe {
            env::remove_var("RUSTY_GPT_TRAIN_STEPS");
            env::remove_var("RUSTY_GPT_EVAL_INTERVAL");
            env::remove_var("RUSTY_GPT_GENERATE_TOKENS");
            env::remove_var("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM");
            env::remove_var("RUSTY_GPT_PREFETCH_BATCHES");
        }
    }

    #[test]
    fn hyperparameter_args_take_precedence_over_env() {
        let config = parse_runtime_config_with_checkpoint(
            [
                "--block-size".to_string(),
                "64".to_string(),
                "--batch-size".to_string(),
                "4".to_string(),
                "--embed-dim".to_string(),
                "32".to_string(),
                "--num-heads".to_string(),
                "4".to_string(),
                "--num-layers".to_string(),
                "2".to_string(),
                "--learning-rate".to_string(),
                "0.001".to_string(),
            ],
            RuntimeEnv {
                block_size: Some("128".to_string()),
                batch_size: Some("16".to_string()),
                embed_dim: Some("64".to_string()),
                num_heads: Some("8".to_string()),
                num_layers: Some("3".to_string()),
                learning_rate: Some("0.01".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .unwrap();

        assert_eq!(64, config.hyperparameters.block_size);
        assert_eq!(4, config.hyperparameters.batch_size);
        assert_eq!(32, config.hyperparameters.embed_dim);
        assert_eq!(4, config.hyperparameters.num_heads);
        assert_eq!(8, config.hyperparameters.head_dim);
        assert_eq!(2, config.hyperparameters.num_layers);
        assert_eq!(0.001, config.hyperparameters.learning_rate);
    }

    #[test]
    fn hyperparameters_reject_invalid_head_split() {
        let err = parse_runtime_config(
            [
                "--embed-dim".to_string(),
                "30".to_string(),
                "--num-heads".to_string(),
                "8".to_string(),
            ],
            None,
            None,
            None,
        )
        .expect_err("invalid head split should fail");

        assert!(
            err.to_string()
                .contains("embed_dim must be divisible by num_heads")
        );
    }

    #[test]
    fn hyperparameters_reject_non_positive_minigpt_gradient_clip_norm() {
        let _guard = lock_env();
        // SAFETY: This unit test mutates process environment while not relying on
        // other environment-sensitive code in parallel.
        unsafe {
            env::set_var("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM", "0");
        }

        let err = Hyperparameters::from_env()
            .expect_err("non-positive gradient clipping norm should fail");

        assert!(
            err.to_string()
                .contains("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM must be greater than zero")
        );

        // SAFETY: See note above.
        unsafe {
            env::remove_var("RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM");
        }
    }

    #[test]
    fn split_training_and_value_tokens_uses_last_ten_percent_for_value() {
        let tokens: Vec<usize> = (0..100).collect();

        let (training_tokens, value_tokens, value_block_size) =
            split_training_and_value_tokens(&tokens, 8).unwrap();

        assert_eq!((0..90).collect::<Vec<_>>(), training_tokens);
        assert_eq!((90..100).collect::<Vec<_>>(), value_tokens);
        assert_eq!(8, value_block_size);
    }

    #[test]
    fn split_training_and_value_tokens_allows_shorter_value_block() {
        let tokens: Vec<usize> = (0..230).collect();

        let (_training_tokens, value_tokens, value_block_size) =
            split_training_and_value_tokens(&tokens, 128).unwrap();

        assert_eq!((207..230).collect::<Vec<_>>(), value_tokens);
        assert_eq!(22, value_block_size);
    }

    #[test]
    fn minigpt_training_saves_checkpoint_mpk_file() {
        type TestBackend = Autodiff<NdArray<f32, i64>>;
        let device = NdArrayDevice::Cpu;
        let checkpoint_path = std::env::temp_dir()
            .join("rusty-gpt-checkpoint-tests")
            .join(format!("mini-gpt-{}", std::process::id()));
        let saved_path = checkpoint_path.with_extension("mpk");
        let metadata_path = checkpoint_path.with_extension("metadata.json");
        let _ = fs::remove_file(&saved_path);
        let _ = fs::remove_file(&metadata_path);

        let hyperparameters = Hyperparameters {
            block_size: 4,
            batch_size: 2,
            embed_dim: 8,
            num_heads: 2,
            head_dim: 4,
            num_layers: 1,
            dropout: 0.0,
            learning_rate: 1e-4,
            train_steps: 1,
            eval_interval: 0,
            generate_tokens: 4,
            minigpt_grad_clip_norm: 1.0,
            prefetch_batches: 2,
            checkpoint_interval: 0,
            checkpoint_keep: 3,
            ..Hyperparameters::default()
        };
        let text = "abcdefghijklmnopqrstuvwxyz ".repeat(8);

        let _guard = lock_env();
        // SAFETY: This test mutates process environment before running any
        // threaded code and restores it before returning.
        unsafe {
            env::set_var(BPE_TOKENIZER_ENV, "tests/fixtures/tokenizer.json");
        }

        run_training_demo::<TestBackend>(
            &text,
            hyperparameters,
            ModelChoice::MiniGpt,
            &device,
            &checkpoint_path,
            TrainingDemoOptions {
                backend_label: "cpu",
                logger: EventLogger::stdout(LogFormat::Plain),
                benchmark_generation: false,
                benchmark_config: BenchmarkConfig::default(),
                input_source: "test".to_string(),
                resume_from: None,
            },
        )
        .unwrap();

        // SAFETY: See note above.
        unsafe {
            env::remove_var(BPE_TOKENIZER_ENV);
        }

        assert!(
            saved_path.is_file(),
            "expected training to save {:?}",
            saved_path
        );
        assert!(
            metadata_path.is_file(),
            "expected training to save {:?}",
            metadata_path
        );

        let _ = fs::remove_file(saved_path);
        let _ = fs::remove_file(metadata_path);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn backend_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--backend".to_string(), "cuda".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(BackendChoice::Cuda, config.backend);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn backend_cuda_arg_requires_feature() {
        let err = parse_runtime_config(
            ["--backend".to_string(), "cuda".to_string()],
            None,
            None,
            None,
        )
        .expect_err("cuda backend should fail when feature is disabled");

        assert!(err.to_string().contains("cuda backend is not compiled in"));
    }

    #[test]
    fn backend_can_be_selected_from_env() {
        let config = parse_runtime_config(Vec::<String>::new(), Some("cpu"), None, None).unwrap();

        assert_eq!(BackendChoice::Cpu, config.backend);
    }

    #[test]
    fn backend_arg_takes_precedence_over_env() {
        let config = parse_runtime_config(
            ["--backend".to_string(), "cpu".to_string()],
            Some("cuda"),
            None,
            None,
        )
        .unwrap();

        assert_eq!(BackendChoice::Cpu, config.backend);
    }

    #[test]
    fn input_path_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--input".to_string(), "data/custom.txt".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(PathBuf::from("data/custom.txt"), config.input_path);
    }

    #[test]
    fn input_arg_takes_precedence_over_env() {
        let config = parse_runtime_config(
            ["--input".to_string(), "data/from-arg.txt".to_string()],
            None,
            Some("data/from-env.txt"),
            None,
        )
        .unwrap();

        assert_eq!(PathBuf::from("data/from-arg.txt"), config.input_path);
    }

    #[test]
    fn checkpoint_path_can_be_selected_from_args() {
        fs::create_dir_all("checkpoints").unwrap();
        let config = parse_runtime_config_with_checkpoint(
            ["--checkpoint".to_string(), "checkpoints/custom".to_string()],
            RuntimeEnv::default(),
        )
        .unwrap();

        assert!(config.checkpoint_path.ends_with("checkpoints/custom"));
    }

    #[test]
    fn checkpoint_arg_takes_precedence_over_env() {
        fs::create_dir_all("checkpoints").unwrap();
        let config = parse_runtime_config_with_checkpoint(
            [
                "--checkpoint".to_string(),
                "checkpoints/from-arg".to_string(),
            ],
            RuntimeEnv {
                checkpoint: Some("checkpoints/from-env".to_string()),
                ..RuntimeEnv::default()
            },
        )
        .unwrap();

        assert!(config.checkpoint_path.ends_with("checkpoints/from-arg"));
    }

    #[test]
    fn missing_checkpoint_arg_value_returns_clear_error() {
        let err = parse_runtime_config_with_checkpoint(
            ["--checkpoint".to_string()],
            RuntimeEnv::default(),
        )
        .expect_err("missing checkpoint value should fail");

        assert!(
            err.to_string()
                .contains("a value is required for '--checkpoint")
        );
    }

    #[test]
    fn missing_backend_arg_value_returns_clear_error() {
        let err = parse_runtime_config(["--backend".to_string()], None, None, None)
            .expect_err("missing backend value should fail");

        assert!(
            err.to_string()
                .contains("a value is required for '--backend")
        );
    }

    #[test]
    fn missing_input_arg_value_returns_clear_error() {
        let err = parse_runtime_config(["--input".to_string()], None, None, None)
            .expect_err("missing input value should fail");

        assert!(err.to_string().contains("a value is required for '--input"));
    }

    #[test]
    fn invalid_backend_returns_clear_error() {
        let err = parse_runtime_config(
            ["--backend".to_string(), "metal".to_string()],
            None,
            None,
            None,
        )
        .expect_err("invalid backend should fail");

        assert!(err.to_string().contains("unsupported backend"));
    }

    #[test]
    fn model_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--model".to_string(), "single-attention".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::SingleAttention, config.model);
    }

    #[test]
    fn multi_attention_model_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--model".to_string(), "multi-attention".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::MultiAttention, config.model);
    }

    #[test]
    fn minigpt_model_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--model".to_string(), "minigpt".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::MiniGpt, config.model);
    }

    #[test]
    fn mini_gpt_alias_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--model".to_string(), "mini-gpt".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::MiniGpt, config.model);
    }

    #[test]
    fn moe_gpt_model_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--model".to_string(), "moe-gpt".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::MoeGpt, config.model);
    }

    #[test]
    fn moe_hyperparameters_can_be_selected_from_args() {
        let config = parse_runtime_config(
            [
                "--model".to_string(),
                "moe-gpt".to_string(),
                "--moe-experts".to_string(),
                "6".to_string(),
                "--moe-top-k".to_string(),
                "3".to_string(),
                "--moe-aux-loss-weight".to_string(),
                "0.05".to_string(),
            ],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(6, config.hyperparameters.moe_experts);
        assert_eq!(3, config.hyperparameters.moe_top_k);
        assert_eq!(0.05, config.hyperparameters.moe_aux_loss_weight);
    }

    #[test]
    fn moe_hyperparameters_reject_invalid_top_k() {
        let err = parse_runtime_config(
            [
                "--model".to_string(),
                "moe-gpt".to_string(),
                "--moe-experts".to_string(),
                "2".to_string(),
                "--moe-top-k".to_string(),
                "3".to_string(),
            ],
            None,
            None,
            None,
        )
        .expect_err("top-k larger than expert count should fail");

        assert!(err.to_string().contains("moe_top_k must be <= moe_experts"));
    }

    #[test]
    fn interactive_generation_can_be_selected_from_args() {
        let config = parse_runtime_config(
            [
                "--model".to_string(),
                "minigpt".to_string(),
                "--interactive-generate".to_string(),
            ],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::MiniGpt, config.model);
        assert!(config.interactive);
    }

    #[test]
    fn generation_benchmark_can_be_selected_from_args() {
        let config =
            parse_runtime_config(["--benchmark-generation".to_string()], None, None, None).unwrap();

        assert!(config.benchmark_generation);
    }

    #[test]
    fn checkpoint_loading_can_be_selected_from_args() {
        let config =
            parse_runtime_config(["--load-checkpoint".to_string()], None, None, None).unwrap();

        assert!(config.load_checkpoint);
    }

    #[test]
    fn latest_checkpoint_loading_can_be_selected_from_args() {
        let config =
            parse_runtime_config(["--load-latest-checkpoint".to_string()], None, None, None)
                .unwrap();

        assert!(config.load_latest_checkpoint);
    }

    #[test]
    fn checkpoint_loading_modes_are_mutually_exclusive() {
        let err = parse_runtime_config(
            [
                "--load-checkpoint".to_string(),
                "--load-latest-checkpoint".to_string(),
            ],
            None,
            None,
            None,
        )
        .expect_err("checkpoint loading modes should conflict");

        assert!(
            err.to_string()
                .contains("--load-checkpoint and --load-latest-checkpoint are mutually exclusive")
        );
    }

    #[test]
    fn latest_checkpoint_path_uses_newest_mpk_file_without_extension() {
        let dir = std::env::temp_dir().join(format!(
            "rusty-gpt-latest-checkpoint-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let old = dir.join("old.mpk");
        let new = dir.join("new.mpk");
        fs::write(&old, b"old").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        fs::write(&new, b"new").unwrap();

        let latest = latest_checkpoint_path(&dir).unwrap();

        assert_eq!(dir.join("new"), latest);

        let _ = fs::remove_file(old);
        let _ = fs::remove_file(new);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn generation_benchmark_requires_minigpt_training_model() {
        type TestBackend = Autodiff<NdArray<f32, i64>>;
        let device = NdArrayDevice::Cpu;
        let checkpoint_path = std::env::temp_dir().join("rusty-gpt-unused-benchmark-checkpoint");
        let hyperparameters = Hyperparameters {
            block_size: 4,
            batch_size: 2,
            embed_dim: 8,
            num_heads: 2,
            head_dim: 4,
            num_layers: 1,
            dropout: 0.0,
            learning_rate: 1e-4,
            train_steps: 1,
            eval_interval: 0,
            generate_tokens: 4,
            minigpt_grad_clip_norm: 1.0,
            prefetch_batches: 0,
            checkpoint_interval: 0,
            checkpoint_keep: 3,
            ..Hyperparameters::default()
        };

        let err = run_training_demo::<TestBackend>(
            &"abcdefghijklmnopqrstuvwxyz ".repeat(8),
            hyperparameters,
            ModelChoice::Trivial,
            &device,
            &checkpoint_path,
            TrainingDemoOptions {
                backend_label: "cpu",
                logger: EventLogger::stdout(LogFormat::Plain),
                benchmark_generation: true,
                benchmark_config: BenchmarkConfig::default(),
                input_source: "test".to_string(),
                resume_from: None,
            },
        )
        .expect_err("benchmarking should require minigpt or compare");

        assert!(
            err.to_string()
                .contains("generation benchmarks require --model minigpt, moe-gpt, or compare")
        );
    }

    #[test]
    fn compare_model_can_be_selected_from_args() {
        let config = parse_runtime_config(
            ["--model".to_string(), "compare".to_string()],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(ModelChoice::Compare, config.model);
        assert_eq!(
            vec![
                ModelChoice::Trivial,
                ModelChoice::SingleAttention,
                ModelChoice::MultiAttention,
                ModelChoice::MiniGpt,
                ModelChoice::MoeGpt
            ],
            config.model.comparison_models()
        );
    }

    #[test]
    fn model_can_be_selected_from_env() {
        let config =
            parse_runtime_config(Vec::<String>::new(), None, None, Some("single-attention"))
                .unwrap();

        assert_eq!(ModelChoice::SingleAttention, config.model);
    }

    #[test]
    fn model_arg_takes_precedence_over_env() {
        let config = parse_runtime_config(
            ["--model".to_string(), "trivial".to_string()],
            None,
            None,
            Some("single-attention"),
        )
        .unwrap();

        assert_eq!(ModelChoice::Trivial, config.model);
    }

    #[test]
    fn missing_model_arg_value_returns_clear_error() {
        let err = parse_runtime_config(["--model".to_string()], None, None, None)
            .expect_err("missing model value should fail");

        assert!(err.to_string().contains("a value is required for '--model"));
    }

    #[test]
    fn invalid_model_returns_clear_error() {
        let err = parse_runtime_config(
            ["--model".to_string(), "large".to_string()],
            None,
            None,
            None,
        )
        .expect_err("invalid model should fail");

        assert!(err.to_string().contains("unsupported model"));
    }
}
