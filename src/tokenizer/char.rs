use std::collections::HashMap;

#[derive(Clone)]
pub struct CharTokenizer {
    char_to_id: HashMap<char, usize>,
    id_to_char: Vec<char>,
}

impl CharTokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let char_to_id = chars
            .iter()
            .copied()
            .enumerate()
            .map(|(i, c)| (c, i))
            .collect();
        Self {
            char_to_id,
            id_to_char: chars,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| self.char_to_id[&c]).collect()
    }
    pub fn try_encode(&self, text: &str) -> Result<Vec<usize>, String> {
        text.chars()
            .map(|c| {
                self.char_to_id
                    .get(&c)
                    .copied()
                    .ok_or_else(|| format!("unknown character: {c:?}"))
            })
            .collect()
    }
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| self.id_to_char[id]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_text_counts_unique_characters() {
        let tokenizer = CharTokenizer::from_text("banana");

        assert_eq!(3, tokenizer.vocab_size());
    }

    #[test]
    fn encode_decode_round_trips_known_text() {
        let tokenizer = CharTokenizer::from_text("hello world");
        let encoded = tokenizer.encode("world");

        assert_eq!("world", tokenizer.decode(&encoded));
    }

    #[test]
    fn try_encode_returns_error_for_unknown_characters() {
        let tokenizer = CharTokenizer::from_text("abc");

        let err = tokenizer
            .try_encode("az")
            .expect_err("unknown character should fail");

        assert!(err.contains("unknown character"));
    }

    #[test]
    fn ids_are_assigned_in_sorted_character_order() {
        let tokenizer = CharTokenizer::from_text("cba");

        assert_eq!(vec![0, 1, 2], tokenizer.encode("abc"));
        assert_eq!("abc", tokenizer.decode(&[0, 1, 2]));
    }

    #[test]
    #[should_panic]
    fn encode_panics_for_unknown_characters() {
        let tokenizer = CharTokenizer::from_text("abc");

        tokenizer.encode("z");
    }

    #[test]
    #[should_panic]
    fn decode_panics_for_unknown_ids() {
        let tokenizer = CharTokenizer::from_text("abc");

        tokenizer.decode(&[3]);
    }
}
