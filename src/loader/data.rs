use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use std::sync::mpsc::{Receiver, sync_channel};
use std::thread::JoinHandle;

pub type TokenBatch<B> = (Tensor<B, 2, Int>, Tensor<B, 2, Int>);

#[derive(Debug)]
pub struct RawTokenBatch {
    inputs: Vec<i64>,
    targets: Vec<i64>,
    shape: [usize; 2],
}

impl RawTokenBatch {
    pub fn into_tensors<B: Backend>(self, device: &B::Device) -> TokenBatch<B> {
        (
            Tensor::from_data(TensorData::new(self.inputs, self.shape), device),
            Tensor::from_data(TensorData::new(self.targets, self.shape), device),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingPolicy {
    RandomWindow,
    Sequential,
    ShuffledChunks,
}

#[derive(Clone)]
pub struct DataLoader {
    pub tokens: Vec<usize>,
    pub block_size: usize, // context length, start with 64 or 128
    pub batch_size: usize, // start with 32
}

impl DataLoader {
    pub fn next_batch<B: Backend>(&self, device: &B::Device) -> Result<TokenBatch<B>, String> {
        self.next_raw_batch()
            .map(|batch| batch.into_tensors::<B>(device))
    }

    pub fn next_raw_batch(&self) -> Result<RawTokenBatch, String> {
        self.next_raw_batch_with_policy(SamplingPolicy::RandomWindow, 0)
    }

    pub fn next_raw_batch_with_policy(
        &self,
        policy: SamplingPolicy,
        step: usize,
    ) -> Result<RawTokenBatch, String> {
        if self.tokens.len() <= self.block_size {
            return Err(format!(
                "not enough tokens to build a batch: got {}, need at least {}",
                self.tokens.len(),
                self.block_size + 1
            ));
        }

        let mut x: Vec<i64> = Vec::with_capacity(self.batch_size * self.block_size);
        let mut y: Vec<i64> = Vec::with_capacity(self.batch_size * self.block_size);
        let max_start = self.tokens.len() - self.block_size;
        for item in 0..self.batch_size {
            let start = match policy {
                SamplingPolicy::RandomWindow => rand::random_range(0..max_start),
                SamplingPolicy::Sequential => {
                    ((step * self.batch_size + item) * self.block_size) % max_start
                }
                SamplingPolicy::ShuffledChunks => {
                    // ponytail: cheap deterministic shuffle; replace with seeded epoch shuffle if this becomes production data loading.
                    (step * self.batch_size + item)
                        .wrapping_mul(1_103_515_245)
                        .wrapping_add(12_345)
                        % max_start
                }
            };
            x.extend(
                self.tokens[start..start + self.block_size]
                    .iter()
                    .map(|&t| t as i64),
            );
            y.extend(
                self.tokens[start + 1..start + 1 + self.block_size]
                    .iter()
                    .map(|&t| t as i64),
            );
        }

        Ok(RawTokenBatch {
            inputs: x,
            targets: y,
            shape: [self.batch_size, self.block_size],
        })
    }
}

pub struct BatchPrefetcher {
    receiver: Receiver<Result<RawTokenBatch, String>>,
    _worker: JoinHandle<()>,
}

impl BatchPrefetcher {
    pub fn new(loader: DataLoader, depth: usize) -> Self {
        Self::new_with_policy(loader, depth, SamplingPolicy::RandomWindow)
    }

    pub fn new_with_policy(loader: DataLoader, depth: usize, policy: SamplingPolicy) -> Self {
        let depth = depth.max(1);
        let (sender, receiver) = sync_channel(depth);
        let worker = std::thread::spawn(move || {
            let mut step = 0usize;
            loop {
                if sender
                    .send(loader.next_raw_batch_with_policy(policy, step))
                    .is_err()
                {
                    break;
                }
                step += 1;
            }
        });

        Self {
            receiver,
            _worker: worker,
        }
    }

    pub fn next_batch<B: Backend>(&self, device: &B::Device) -> Result<TokenBatch<B>, String> {
        self.receiver
            .recv()
            .map_err(|err| format!("batch prefetch worker stopped: {err}"))?
            .map(|batch| batch.into_tensors::<B>(device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    #[test]
    fn next_batch_returns_requested_shape_when_enough_tokens_exist() {
        type TestBackend = NdArray<f32, i64>;
        let loader = DataLoader {
            tokens: vec![0, 1, 2, 3, 4, 5],
            block_size: 2,
            batch_size: 3,
        };
        let device = NdArrayDevice::Cpu;

        let (x, y) = loader.next_batch::<TestBackend>(&device).unwrap();

        assert_eq!([3, 2], x.shape().dims());
        assert_eq!([3, 2], y.shape().dims());
    }

    #[test]
    fn next_batch_targets_are_inputs_shifted_by_one_token() {
        type TestBackend = NdArray<f32, i64>;
        let loader = DataLoader {
            tokens: vec![10, 20, 30],
            block_size: 2,
            batch_size: 1,
        };
        let device = NdArrayDevice::Cpu;

        let (x, y) = loader.next_batch::<TestBackend>(&device).unwrap();

        assert_eq!(vec![10, 20], x.into_data().to_vec::<i64>().unwrap());
        assert_eq!(vec![20, 30], y.into_data().to_vec::<i64>().unwrap());
    }

    #[test]
    fn next_batch_returns_error_when_tokens_are_too_short() {
        type TestBackend = NdArray<f32, i64>;
        let loader = DataLoader {
            tokens: vec![0, 1],
            block_size: 2,
            batch_size: 1,
        };
        let device = NdArrayDevice::Cpu;

        let err = loader
            .next_batch::<TestBackend>(&device)
            .expect_err("short token stream should fail");

        assert!(err.contains("not enough tokens"));
    }

    #[test]
    fn sequential_policy_advances_by_step() {
        let loader = DataLoader {
            tokens: (0..20).collect(),
            block_size: 2,
            batch_size: 2,
        };

        let first = loader
            .next_raw_batch_with_policy(SamplingPolicy::Sequential, 0)
            .unwrap();
        let second = loader
            .next_raw_batch_with_policy(SamplingPolicy::Sequential, 1)
            .unwrap();

        assert_eq!(vec![0, 1, 2, 3], first.inputs);
        assert_eq!(vec![4, 5, 6, 7], second.inputs);
    }

    #[test]
    fn shuffled_chunks_policy_is_deterministic_for_step() {
        let loader = DataLoader {
            tokens: (0..20).collect(),
            block_size: 2,
            batch_size: 2,
        };

        let first = loader
            .next_raw_batch_with_policy(SamplingPolicy::ShuffledChunks, 3)
            .unwrap();
        let second = loader
            .next_raw_batch_with_policy(SamplingPolicy::ShuffledChunks, 3)
            .unwrap();

        assert_eq!(first.inputs, second.inputs);
    }

    #[test]
    fn prefetcher_returns_requested_batch_shape() {
        type TestBackend = NdArray<f32, i64>;
        let loader = DataLoader {
            tokens: vec![0, 1, 2, 3, 4, 5],
            block_size: 2,
            batch_size: 3,
        };
        let device = NdArrayDevice::Cpu;
        let prefetcher = BatchPrefetcher::new(loader, 2);

        let (x, y) = prefetcher.next_batch::<TestBackend>(&device).unwrap();

        assert_eq!([3, 2], x.shape().dims());
        assert_eq!([3, 2], y.shape().dims());
    }
}
