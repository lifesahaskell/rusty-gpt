#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationOptions {
    pub temperature: f32,
    pub top_k: Option<usize>,
}

impl GenerationOptions {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
        }
    }

    pub fn sampling(temperature: f32, top_k: Option<usize>) -> Result<Self, String> {
        if temperature <= 0.0 {
            return Err("temperature must be greater than zero".to_string());
        }
        if top_k == Some(0) {
            return Err("top_k must be greater than zero".to_string());
        }
        Ok(Self { temperature, top_k })
    }
}

pub(super) fn sample_from_logits(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    random_unit: f32,
) -> usize {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(token, _)| token)
            .expect("vocab size should be greater than zero");
    }

    let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    candidates.sort_by(|(_, left), (_, right)| right.total_cmp(left));
    if let Some(top_k) = top_k {
        candidates.truncate(top_k.min(candidates.len()).max(1));
    }
    let max_logit = candidates
        .iter()
        .map(|(_, logit)| *logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = candidates
        .iter()
        .map(|(_, logit)| ((*logit - max_logit) / temperature).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return candidates[0].0;
    }

    let mut threshold = random_unit.clamp(0.0, 0.999_999) * total;
    for ((token, _), weight) in candidates.iter().zip(weights) {
        if threshold <= weight {
            return *token;
        }
        threshold -= weight;
    }

    candidates.last().map(|(token, _)| *token).unwrap_or(0)
}
