use std::path::PathBuf;

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, RecorderError};
use burn::tensor::backend::Backend;

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
}
