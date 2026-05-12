use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

pub struct DataLoader {
    pub tokens: Vec<usize>,
    pub block_size: usize, // context length, start with 64 or 128
    pub batch_size: usize, // start with 32
}

impl DataLoader {
    pub fn next_batch<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<(Tensor<B, 2, Int>, Tensor<B, 2, Int>), String> {
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
        for _ in 0..self.batch_size {
            let start = rand::random_range(0..max_start);
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
        let shape = [self.batch_size, self.block_size];
        let x_tensor = Tensor::from_data(TensorData::new(x, shape), device);
        let y_tensor = Tensor::from_data(TensorData::new(y, shape), device);
        Ok((x_tensor, y_tensor))
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
}
