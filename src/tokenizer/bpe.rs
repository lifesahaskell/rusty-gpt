#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use super::Tokenizer;
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub struct BpeTrainer {
    vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum BpeTrainingEvent {
    #[serde(rename = "bpe_training_started")]
    Started {
        target_vocab_size: usize,
        initial_vocab_size: usize,
        word_count: usize,
    },
    #[serde(rename = "bpe_training_merge")]
    Merge {
        merge_index: usize,
        vocab_size: usize,
        pair: (u32, u32),
        new_token_id: u32,
        count: usize,
    },
    #[serde(rename = "bpe_training_completed")]
    Completed {
        target_vocab_size: usize,
        final_vocab_size: usize,
        merge_count: usize,
        reason: BpeTrainingStopReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BpeTrainingStopReason {
    TargetVocabSizeReached,
    NoPairs,
    NoRepeatedPairs,
}

impl BpeTrainer {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    pub fn train(&self, corpus: &str) -> BpeTokenizer {
        self.train_with_observer(corpus, |_| {})
    }

    pub fn train_with_observer(
        &self,
        corpus: &str,
        mut observer: impl FnMut(BpeTrainingEvent),
    ) -> BpeTokenizer {
        // 1. Initialize: each word becomes Vec<u32> of byte IDs.
        let mut words: Vec<Vec<u32>> = corpus
            .split_whitespace()
            .map(|w| w.bytes().map(|b| b as u32).collect())
            .collect();

        let mut vocab: Vec<Vec<u8>> = (0..256u32).map(|i| vec![i as u8]).collect();
        let mut merges = Vec::new();
        let mut stop_reason = BpeTrainingStopReason::TargetVocabSizeReached;

        observer(BpeTrainingEvent::Started {
            target_vocab_size: self.vocab_size,
            initial_vocab_size: vocab.len(),
            word_count: words.len(),
        });

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
                stop_reason = BpeTrainingStopReason::NoPairs;
                break;
            };
            if count < 2 {
                stop_reason = BpeTrainingStopReason::NoRepeatedPairs;
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

            observer(BpeTrainingEvent::Merge {
                merge_index: merges.len() - 1,
                vocab_size: vocab.len(),
                pair: best_pair,
                new_token_id: new_id,
                count,
            });
        }

        let merge_ranks = merges
            .iter()
            .enumerate()
            .map(|(i, &(pair, _))| (pair, i as u32))
            .collect();

        observer(BpeTrainingEvent::Completed {
            target_vocab_size: self.vocab_size,
            final_vocab_size: vocab.len(),
            merge_count: merges.len(),
            reason: stop_reason,
        });

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

#[derive(Clone)]
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
    fn train_with_observer_emits_structured_events() {
        let mut events = Vec::new();
        let tokenizer = BpeTrainer::new(258)
            .train_with_observer("banana banana banana", |event| events.push(event));

        assert_eq!(258, tokenizer.vocab_size());
        assert_eq!(
            Some(&BpeTrainingEvent::Started {
                target_vocab_size: 258,
                initial_vocab_size: 256,
                word_count: 3,
            }),
            events.first()
        );
        assert!(matches!(
            events.get(1),
            Some(BpeTrainingEvent::Merge {
                merge_index: 0,
                vocab_size: 257,
                new_token_id: 256,
                count,
                ..
            }) if *count >= 2
        ));
        assert_eq!(
            Some(&BpeTrainingEvent::Completed {
                target_vocab_size: 258,
                final_vocab_size: 258,
                merge_count: 2,
                reason: BpeTrainingStopReason::TargetVocabSizeReached,
            }),
            events.last()
        );
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
