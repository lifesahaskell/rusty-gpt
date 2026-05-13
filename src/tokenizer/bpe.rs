#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use super::Tokenizer;
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub struct BpeTrainer {
    vocab_size: usize,
}

impl BpeTrainer {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    pub fn train(&self, corpus: &str) -> BpeTokenizer {
        // 1. Initialize: each word becomes Vec<u32> of byte IDs.
        let mut words: Vec<Vec<u32>> = corpus
            .split_whitespace()
            .map(|w| w.bytes().map(|b| b as u32).collect())
            .collect();

        let mut vocab: Vec<Vec<u8>> = (0..256u32).map(|i| vec![i as u8]).collect();
        let mut merges = Vec::new();

        while vocab.len() < self.vocab_size {
            // 2. Count all adjacent pairs across all words.
            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            for word in &words {
                for window in word.windows(2) {
                    *pair_counts.entry((window[0], window[1])).or_insert(0) += 1;
                }
            }

            // 3. Find the most frequent pair. Stop if no pair appears > 1 times.
            let Some((&best_pair, &count)) = pair_counts.iter().max_by_key(|(_, c)| *c) else {
                break;
            };
            if count < 2 {
                break;
            }

            // 4. Assign new token ID. Build its bytes representation.
            let new_id = vocab.len() as u32;
            let mut new_bytes = vocab[best_pair.0 as usize].clone();
            new_bytes.extend_from_slice(&vocab[best_pair.1 as usize]);
            vocab.push(new_bytes);
            merges.push((best_pair, new_id));

            // 5. Apply this merge to all words.
            for word in &mut words {
                Self::merge_in_place(word, best_pair, new_id);
            }
        }

        let merge_ranks = merges
            .iter()
            .enumerate()
            .map(|(i, &(pair, _))| (pair, i as u32))
            .collect();

        BpeTokenizer {
            merges,
            merge_ranks,
            vocab,
        }
    }

    fn merge_in_place(word: &mut Vec<u32>, pair: (u32, u32), new_id: u32) {
        let mut i = 0;
        let mut result = Vec::with_capacity(word.len());
        while i < word.len() {
            if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
                result.push(new_id);
                i += 2;
            } else {
                result.push(word[i]);
                i += 1;
            }
        }
        *word = result;
    }
}

#[derive(Serialize, Deserialize)]
struct BpeTokenizerData {
    merges: Vec<((u32, u32), u32)>,
    vocab: Vec<Vec<u8>>,
}

pub struct BpeTokenizer {
    // Ordered list of merges: (token_a, token_b) -> new_token_id
    merges: Vec<((u32, u32), u32)>,
    // For fast lookup during encoding
    merge_ranks: HashMap<(u32, u32), u32>, // pair -> rank (lower = earlier)
    // Vocab: token_id -> bytes
    vocab: Vec<Vec<u8>>,
}

impl BpeTokenizer {
    fn encode_bytes(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        for word in text.split_whitespace() {
            let mut tokens: Vec<u32> = word.bytes().map(|b| b as u32).collect();

            loop {
                // Find the lowest-rank merge applicable in current tokens.
                let mut best: Option<(usize, u32)> = None; // (position, rank)
                for i in 0..tokens.len().saturating_sub(1) {
                    if let Some(&rank) = self.merge_ranks.get(&(tokens[i], tokens[i + 1]))
                        && best.is_none_or(|(_, r)| rank < r)
                    {
                        best = Some((i, rank));
                    }
                }

                let Some((pos, rank)) = best else { break };
                let merge = self.merges[rank as usize];
                tokens.splice(pos..pos + 2, std::iter::once(merge.1));
            }

            result.extend(tokens);
            // Handle whitespace however you split it — see note below.
        }
        result
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = BpeTokenizerData {
            merges: self.merges.clone(),
            vocab: self.vocab.clone(),
        };
        std::fs::write(path, serde_json::to_vec_pretty(&data)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let data: BpeTokenizerData = serde_json::from_slice(&std::fs::read(path)?)?;
        let merge_ranks = data
            .merges
            .iter()
            .enumerate()
            .map(|(i, &(pair, _))| (pair, i as u32))
            .collect();

        Ok(Self {
            merges: data.merges,
            merge_ranks,
            vocab: data.vocab,
        })
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_bytes(text)
    }

    fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter_map(|&id| self.vocab.get(id as usize))
            .flat_map(|piece| piece.iter().copied())
            .collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;

    #[test]
    fn trained_bpe_implements_tokenizer_contract() {
        let tokenizer = BpeTrainer::new(260).train("banana bandana banana");

        let ids = tokenizer.encode("banana");

        assert!(!ids.is_empty());
        assert_eq!("banana", tokenizer.decode(&ids));
        assert!(tokenizer.vocab_size() >= 256);
    }

    #[test]
    fn training_stops_at_requested_vocab_size() {
        let tokenizer = BpeTrainer::new(258).train("banana banana banana");

        assert_eq!(258, tokenizer.vocab_size());
        assert_eq!(2, tokenizer.merges.len());
    }

    #[test]
    fn training_stops_when_no_repeated_pair_exists() {
        let tokenizer = BpeTrainer::new(300).train("ab cd");

        assert_eq!(256, tokenizer.vocab_size());
        assert!(tokenizer.merges.is_empty());
    }

    #[test]
    fn encode_applies_lowest_rank_merge_first() {
        let tokenizer = BpeTokenizer {
            merges: vec![((b'a' as u32, b'b' as u32), 256), ((256, b'c' as u32), 257)],
            merge_ranks: HashMap::from([((b'a' as u32, b'b' as u32), 0), ((256, b'c' as u32), 1)]),
            vocab: {
                let mut vocab: Vec<Vec<u8>> = (0..256u32).map(|i| vec![i as u8]).collect();
                vocab.push(b"ab".to_vec());
                vocab.push(b"abc".to_vec());
                vocab
            },
        };

        let ids = tokenizer.encode("abc");

        assert_eq!(vec![257], ids);
        assert_eq!("abc", tokenizer.decode(&ids));
    }

    #[test]
    fn save_and_load_preserves_encoding() {
        let tokenizer = BpeTrainer::new(260).train("banana bandana banana");
        let path = std::env::temp_dir().join(format!(
            "rusty-gpt-bpe-tokenizer-{}.json",
            std::process::id()
        ));
        let ids = tokenizer.encode("banana");

        tokenizer.save(&path).unwrap();
        let loaded = BpeTokenizer::load(&path).unwrap();

        assert_eq!(ids, loaded.encode("banana"));
        assert_eq!("banana", loaded.decode(&ids));
        assert_eq!(tokenizer.vocab_size(), loaded.vocab_size());

        let _ = std::fs::remove_file(path);
    }
}
