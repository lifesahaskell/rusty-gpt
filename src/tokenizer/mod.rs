pub mod bpe;
pub mod char;

#[allow(dead_code)]
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
    fn vocab_size(&self) -> usize;
}
