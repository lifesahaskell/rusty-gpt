use std::time::{Duration, Instant};

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use crate::model::MiniGpt;

const DEFAULT_PROMPT_LENS: [usize; 3] = [10, 50, 100];
const DEFAULT_GEN_LENS: [usize; 3] = [50, 100, 200];

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationBenchmarkResult {
    pub prompt_len: usize,
    pub gen_len: usize,
    pub naive_time: Duration,
    pub cached_time: Duration,
    pub speedup: f64,
}

pub fn benchmark_generation<B: Backend>(model: &MiniGpt<B>, device: &B::Device) {
    for result in benchmark_generation_cases(model, device, &DEFAULT_PROMPT_LENS, &DEFAULT_GEN_LENS)
    {
        println!(
            "prompt={}, gen={}: naive {:?}, cached {:?}, speedup {:.2}x",
            result.prompt_len,
            result.gen_len,
            result.naive_time,
            result.cached_time,
            result.speedup
        );
    }
}

pub fn benchmark_generation_cases<B: Backend>(
    model: &MiniGpt<B>,
    device: &B::Device,
    prompt_lens: &[usize],
    gen_lens: &[usize],
) -> Vec<GenerationBenchmarkResult> {
    let mut results = Vec::new();

    for &prompt_len in prompt_lens {
        for &gen_len in gen_lens {
            if prompt_len + gen_len > model.block_size() {
                println!(
                    "prompt={prompt_len}, gen={gen_len}: skipped; cached generation needs block_size >= {}",
                    prompt_len + gen_len
                );
                continue;
            }

            let prompt = random_tokens(prompt_len, model.vocab_size());

            let t0 = Instant::now();
            model
                .generate(&prompt, gen_len, device)
                .expect("random benchmark prompt should be valid");
            let naive_time = t0.elapsed();

            let prompt_data: Vec<i64> = prompt.iter().map(|&token| token as i64).collect();
            let prompt_tensor = Tensor::<B, 2, Int>::from_data(
                TensorData::new(prompt_data, [1, prompt_len]),
                device,
            );

            let t0 = Instant::now();
            let _ = model.generate_cached(prompt_tensor, gen_len);
            let cached_time = t0.elapsed();
            let speedup = naive_time.as_secs_f64() / cached_time.as_secs_f64();

            results.push(GenerationBenchmarkResult {
                prompt_len,
                gen_len,
                naive_time,
                cached_time,
                speedup,
            });
        }
    }

    results
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
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    #[test]
    fn random_tokens_returns_requested_len_with_vocab_bounds() {
        let tokens = random_tokens(16, 7);

        assert_eq!(16, tokens.len());
        assert!(tokens.iter().all(|&token| token < 7));
    }

    #[test]
    fn benchmark_generation_cases_returns_one_result_per_supported_case() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 8, 2, &device);

        let results = benchmark_generation_cases(&model, &device, &[2], &[1]);

        assert_eq!(1, results.len());
        assert_eq!(2, results[0].prompt_len);
        assert_eq!(1, results[0].gen_len);
        assert!(results[0].speedup.is_finite());
    }

    #[test]
    fn benchmark_generation_cases_skips_cases_beyond_block_size() {
        type TestBackend = NdArray<f32, i64>;
        let device = NdArrayDevice::Cpu;
        let model = MiniGpt::<TestBackend>::new(7, 8, 1, 4, 2, &device);

        let results = benchmark_generation_cases(&model, &device, &[3], &[2]);

        assert!(results.is_empty());
    }
}
