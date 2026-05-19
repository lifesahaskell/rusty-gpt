pub mod bpe;
pub mod char;

use std::path::Path;

use anyhow::Result;
use bpe::BpeTokenizer;
use char::CharTokenizer;

#[allow(dead_code)]
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
    fn vocab_size(&self) -> usize;
}

#[derive(Clone)]
pub enum RuntimeTokenizer {
    Char(CharTokenizer),
    Bpe(BpeTokenizer),
}

impl RuntimeTokenizer {
    pub fn char_from_text(text: &str) -> Self {
        Self::Char(CharTokenizer::from_text(text))
    }

    pub fn load_bpe(path: &Path) -> Result<Self> {
        Ok(Self::Bpe(BpeTokenizer::load(path)?))
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Char(tokenizer) => tokenizer.vocab_size(),
            Self::Bpe(tokenizer) => tokenizer.vocab_size(),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Char(_) => "char",
            Self::Bpe(_) => "bpe",
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self {
            Self::Char(tokenizer) => tokenizer.encode(text),
            Self::Bpe(tokenizer) => tokenizer
                .encode(text)
                .into_iter()
                .map(|token| token as usize)
                .collect(),
        }
    }

    pub fn try_encode(&self, text: &str) -> Result<Vec<usize>, String> {
        match self {
            Self::Char(tokenizer) => tokenizer.try_encode(text),
            Self::Bpe(_) => Ok(self.encode(text)),
        }
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        match self {
            Self::Char(tokenizer) => tokenizer.decode(ids),
            Self::Bpe(tokenizer) => {
                let ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
                tokenizer.decode(&ids)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::bpe::BpeTrainer;

    #[test]
    fn runtime_tokenizer_kind_returns_char_or_bpe() {
        let char_tokenizer = RuntimeTokenizer::char_from_text("abc");
        assert_eq!("char", char_tokenizer.kind());

        let bpe = BpeTrainer::new(258).train("banana banana banana");
        let path = std::env::temp_dir().join(format!(
            "rusty-gpt-tokenizer-kind-{}.json",
            std::process::id()
        ));
        bpe.save(&path).unwrap();
        let bpe_tokenizer = RuntimeTokenizer::load_bpe(&path).unwrap();
        assert_eq!("bpe", bpe_tokenizer.kind());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn runtime_tokenizer_loads_bpe_from_json() {
        let tokenizer = BpeTrainer::new(258).train("banana banana banana");
        let path = std::env::temp_dir().join(format!(
            "rusty-gpt-runtime-bpe-tokenizer-{}.json",
            std::process::id()
        ));
        tokenizer.save(&path).unwrap();

        let runtime_tokenizer = RuntimeTokenizer::load_bpe(&path).unwrap();
        let ids = runtime_tokenizer.encode("banana");

        assert_eq!(258, runtime_tokenizer.vocab_size());
        assert_eq!("banana", runtime_tokenizer.decode(&ids));

        let _ = std::fs::remove_file(path);
    }
}
