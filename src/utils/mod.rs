use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::model::MiniGpt;
use crate::observability::{BenchmarkStats, EventLogger, RuntimeEvent};

pub const DEFAULT_PROMPT_LENS: [usize; 3] = [10, 50, 100];
pub const DEFAULT_GEN_LENS: [usize; 3] = [50, 100, 200];
pub const DEFAULT_BENCHMARK_WARMUPS: usize = 1;
pub const DEFAULT_BENCHMARK_ITERATIONS: usize = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkConfig {
    pub prompt_lens: Vec<usize>,
    pub gen_lens: Vec<usize>,
    pub warmups: usize,
    pub iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            prompt_lens: DEFAULT_PROMPT_LENS.to_vec(),
            gen_lens: DEFAULT_GEN_LENS.to_vec(),
            warmups: DEFAULT_BENCHMARK_WARMUPS,
            iterations: DEFAULT_BENCHMARK_ITERATIONS,
        }
    }
}

impl BenchmarkConfig {
    pub fn validate(&self) -> Result<()> {
        if self.prompt_lens.is_empty() {
            bail!("benchmark prompt lengths must not be empty");
        }
        if self.gen_lens.is_empty() {
            bail!("benchmark generation lengths must not be empty");
        }
        if self.iterations == 0 {
            bail!("benchmark iterations must be greater than zero");
        }
        if self.prompt_lens.contains(&0) {
            bail!("benchmark prompt lengths must be greater than zero");
        }
        if self.gen_lens.contains(&0) {
            bail!("benchmark generation lengths must be greater than zero");
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenerationBenchmarkCase {
    Result(GenerationBenchmarkResult),
    Skipped(GenerationBenchmarkSkipped),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationBenchmarkResult {
    pub prompt_len: usize,
    pub gen_len: usize,
    pub warmups: usize,
    pub iterations: usize,
    pub naive: BenchmarkStats,
    pub cached: BenchmarkStats,
    pub speedup: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationBenchmarkSkipped {
    pub prompt_len: usize,
    pub gen_len: usize,
    pub reason: String,
}

pub fn parse_usize_list(value: &str, label: &str) -> Result<Vec<usize>> {
    let values: Result<Vec<usize>, _> = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::parse::<usize>)
        .collect();
    let values = values.map_err(|err| anyhow::anyhow!("invalid {label}: {err}"))?;
    if values.is_empty() {
        bail!("{label} must include at least one integer");
    }
    if values.contains(&0) {
        bail!("{label} values must be greater than zero");
    }

    Ok(values)
}

pub fn benchmark_generation<B: Backend>(
    model: &MiniGpt<B>,
    device: &B::Device,
    config: &BenchmarkConfig,
    logger: &EventLogger,
) -> Result<Vec<GenerationBenchmarkCase>, String> {
    let cases = benchmark_generation_cases(model, device, config)?;
    for case in &cases {
        match case {
            GenerationBenchmarkCase::Result(result) => logger.log(RuntimeEvent::BenchmarkResult {
                prompt_len: result.prompt_len,
                gen_len: result.gen_len,
                warmups: result.warmups,
                iterations: result.iterations,
                naive: result.naive.clone(),
                cached: result.cached.clone(),
                speedup: result.speedup,
            }),
            GenerationBenchmarkCase::Skipped(skipped) => {
                logger.log(RuntimeEvent::BenchmarkSkipped {
                    prompt_len: skipped.prompt_len,
                    gen_len: skipped.gen_len,
                    reason: skipped.reason.clone(),
                })
            }
        }
    }

    Ok(cases)
}

pub fn benchmark_generation_cases<B: Backend>(
    model: &MiniGpt<B>,
    device: &B::Device,
    config: &BenchmarkConfig,
) -> Result<Vec<GenerationBenchmarkCase>, String> {
    config.validate().map_err(|err| err.to_string())?;
    let mut cases = Vec::new();

    for &prompt_len in &config.prompt_lens {
        for &gen_len in &config.gen_lens {
            if prompt_len + gen_len > model.block_size() {
                cases.push(GenerationBenchmarkCase::Skipped(
                    GenerationBenchmarkSkipped {
                        prompt_len,
                        gen_len,
                        reason: format!(
                            "cached generation needs block_size >= {}",
                            prompt_len + gen_len
                        ),
                    },
                ));
                continue;
            }

            let prompt = random_tokens(prompt_len, model.vocab_size());
            let prompt_tensor = prompt_tensor(&prompt, device);

            for _ in 0..config.warmups {
                model.generate(&prompt, gen_len, device)?;
                let _ = model.generate_cached(prompt_tensor.clone(), gen_len);
            }

            let mut naive_times = Vec::with_capacity(config.iterations);
            let mut cached_times = Vec::with_capacity(config.iterations);
            for _ in 0..config.iterations {
                let t0 = Instant::now();
                model.generate(&prompt, gen_len, device)?;
                naive_times.push(t0.elapsed());

                let t0 = Instant::now();
                let _ = model.generate_cached(prompt_tensor.clone(), gen_len);
                cached_times.push(t0.elapsed());
            }

            let naive = summarize(&naive_times, gen_len);
            let cached = summarize(&cached_times, gen_len);
            let speedup = naive.mean_ms / cached.mean_ms;

            cases.push(GenerationBenchmarkCase::Result(GenerationBenchmarkResult {
                prompt_len,
                gen_len,
                warmups: config.warmups,
                iterations: config.iterations,
                naive,
                cached,
                speedup,
            }));
        }
    }

    Ok(cases)
}

fn prompt_tensor<B: Backend>(prompt: &[usize], device: &B::Device) -> Tensor<B, 2, Int> {
    let prompt_data: Vec<i64> = prompt.iter().map(|&token| token as i64).collect();
    Tensor::<B, 2, Int>::from_data(TensorData::new(prompt_data, [1, prompt.len()]), device)
}

fn summarize(durations: &[Duration], gen_len: usize) -> BenchmarkStats {
    let mut millis: Vec<f64> = durations
        .iter()
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .collect();
    millis.sort_by(f64::total_cmp);

    let sum_ms: f64 = millis.iter().sum();
    let mean_ms = sum_ms / millis.len() as f64;
    let total_generated_tokens = gen_len * millis.len();
    let total_seconds = sum_ms / 1000.0;

    BenchmarkStats {
        min_ms: millis[0],
        max_ms: millis[millis.len() - 1],
        mean_ms,
        p50_ms: percentile(&millis, 0.50),
        p95_ms: percentile(&millis, 0.95),
        tokens_per_second: total_generated_tokens as f64 / total_seconds,
    }
}

fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    let rank = ((sorted_values.len() as f64 * percentile).ceil() as usize).saturating_sub(1);
    sorted_values[rank.min(sorted_values.len() - 1)]
}

pub fn random_tokens(len: usize, vocab_size: usize) -> Vec<usize> {
    assert!(vocab_size > 0, "vocab_size must be greater than zero");
    (0..len)
        .map(|_| rand::random_range(0..vocab_size))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MiniGpt;
    use crate::observability::{EventLogger, LogFormat};
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use std::sync::{Arc, Mutex};

    #[test]
    fn random_tokens_returns_requested_len_with_vocab_bounds() {
        let tokens = random_tokens(16, 7);

        assert_eq!(16, tokens.len());
        assert!(tokens.iter().all(|&token| token < 7));
    }

    #[test]
    fn parses_comma_separated_usize_list() {
        let values = parse_usize_list("1, 2,3", "benchmark prompt lengths").unwrap();

        assert_eq!(vec![1, 2, 3], values);
    }

    #[test]
    fn rejects_empty_or_zero_benchmark_values() {
        assert!(parse_usize_list("", "values").is_err());
        assert!(parse_usize_list("1,0", "values").is_err());
    }

    #[test]
    fn benchmark_config_rejects_zero_iterations() {
        let config = BenchmarkConfig {
            iterations: 0,
            ..BenchmarkConfig::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(3.0, percentile(&values, 0.50));
        assert_eq!(5.0, percentile(&values, 0.95));
    }

    #[test]
    fn benchmark_generation_cases_returns_one_result_per_supported_case() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 8, 2, &device);
        let config = BenchmarkConfig {
            prompt_lens: vec![2],
            gen_lens: vec![1],
            warmups: 0,
            iterations: 2,
        };

        let cases = benchmark_generation_cases(&model, &device, &config).unwrap();

        let GenerationBenchmarkCase::Result(result) = &cases[0] else {
            panic!("expected benchmark result");
        };
        assert_eq!(1, cases.len());
        assert_eq!(2, result.prompt_len);
        assert_eq!(1, result.gen_len);
        assert_eq!(2, result.iterations);
        assert!(result.speedup.is_finite());
        assert!(result.naive.p95_ms >= result.naive.min_ms);
        assert!(result.cached.tokens_per_second.is_finite());
    }

    #[test]
    fn benchmark_generation_cases_reports_skipped_cases() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 4, 2, &device);
        let config = BenchmarkConfig {
            prompt_lens: vec![3],
            gen_lens: vec![2],
            warmups: 0,
            iterations: 1,
        };

        let cases = benchmark_generation_cases(&model, &device, &config).unwrap();

        assert!(matches!(
            &cases[0],
            GenerationBenchmarkCase::Skipped(GenerationBenchmarkSkipped {
                prompt_len: 3,
                gen_len: 2,
                ..
            })
        ));
    }

    #[test]
    fn benchmark_generation_logs_results() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 8, 2, &device);
        let config = BenchmarkConfig {
            prompt_lens: vec![2],
            gen_lens: vec![1],
            warmups: 0,
            iterations: 1,
        };
        let lines = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&lines);
        let logger = EventLogger::with_sink(LogFormat::Json, move |line| {
            captured.lock().unwrap().push(line);
        });

        benchmark_generation(&model, &device, &config, &logger).unwrap();

        let lines = lines.lock().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!("benchmark_result", parsed["event"]);
        assert_eq!(2, parsed["prompt_len"]);
        assert_eq!(1, parsed["gen_len"]);
    }
}
