use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

const DATASETS_SERVER_ROWS_URL: &str = "https://datasets-server.huggingface.co/rows";
const DEFAULT_CONFIG: &str = "default";
const DEFAULT_SPLIT: &str = "train";
const DEFAULT_COLUMN: &str = "text";
const DEFAULT_ROWS: usize = 1_000;
const MAX_PAGE_ROWS: usize = 100;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuggingFaceDatasetSpec {
    pub dataset: String,
    pub config: String,
    pub split: String,
    pub column: String,
    pub offset: usize,
    pub rows: usize,
}

impl HuggingFaceDatasetSpec {
    pub fn parse(input: &str) -> Result<Option<Self>> {
        let Some(rest) = input.strip_prefix("hf://") else {
            return Ok(None);
        };

        let (dataset, query) = rest.split_once('?').unwrap_or((rest, ""));
        let dataset = dataset.trim_matches('/');
        if dataset.is_empty() {
            bail!("Hugging Face dataset URI must include a dataset id after hf://");
        }

        let mut spec = Self {
            dataset: dataset.to_string(),
            config: DEFAULT_CONFIG.to_string(),
            split: DEFAULT_SPLIT.to_string(),
            column: DEFAULT_COLUMN.to_string(),
            offset: 0,
            rows: DEFAULT_ROWS,
        };

        for pair in query.split('&').filter(|part| !part.is_empty()) {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            let value = percent_decode(value);
            match key {
                "config" => spec.config = required_value(key, value)?,
                "split" => spec.split = required_value(key, value)?,
                "column" => spec.column = required_value(key, value)?,
                "offset" => spec.offset = parse_usize_param(key, &value)?,
                "rows" | "limit" => spec.rows = parse_positive_usize_param(key, &value)?,
                other => bail!("unsupported Hugging Face dataset URI parameter '{other}'"),
            }
        }

        Ok(Some(spec))
    }

    fn rows_url(&self, offset: usize, length: usize) -> String {
        format!(
            "{DATASETS_SERVER_ROWS_URL}?dataset={}&config={}&split={}&offset={offset}&length={length}",
            percent_encode(&self.dataset),
            percent_encode(&self.config),
            percent_encode(&self.split)
        )
    }
}

pub fn load_text_from_uri(input: &str) -> Result<Option<String>> {
    let Some(spec) = HuggingFaceDatasetSpec::parse(input)? else {
        return Ok(None);
    };

    load_text(&spec).map(Some)
}

fn load_text(spec: &HuggingFaceDatasetSpec) -> Result<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60))
        .user_agent("rusty-gpt/0.1 huggingface-dataset-loader")
        .build()
        .context("failed to build Hugging Face dataset HTTP client")?;

    let mut output = String::new();
    let mut fetched_rows = 0usize;
    let mut offset = spec.offset;
    let mut total_rows = None;

    while fetched_rows < spec.rows {
        let length = (spec.rows - fetched_rows).min(MAX_PAGE_ROWS);
        let url = spec.rows_url(offset, length);
        let response = client
            .get(&url)
            .send()
            .with_context(|| format!("failed to fetch Hugging Face dataset rows from {url}"))?
            .error_for_status()
            .with_context(|| format!("Hugging Face dataset rows request failed for {url}"))?
            .json::<RowsResponse>()
            .with_context(|| format!("failed to decode Hugging Face dataset rows from {url}"))?;

        if response.rows.is_empty() {
            break;
        }

        total_rows = response.num_rows_total.or(total_rows);
        append_column_text(&mut output, &response.rows, &spec.column)?;

        fetched_rows += response.rows.len();
        offset += response.rows.len();
        if total_rows.is_some_and(|total| offset >= total) {
            break;
        }
    }

    if output.trim().is_empty() {
        bail!(
            "Hugging Face dataset '{}' returned no text for column '{}'",
            spec.dataset,
            spec.column
        );
    }

    Ok(output)
}

#[derive(Debug, Deserialize)]
struct RowsResponse {
    rows: Vec<RowEnvelope>,
    #[serde(default)]
    num_rows_total: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RowEnvelope {
    row: HashMap<String, Value>,
}

fn append_column_text(output: &mut String, rows: &[RowEnvelope], column: &str) -> Result<()> {
    let mut found = false;
    for row in rows {
        let Some(value) = row.row.get(column) else {
            continue;
        };
        let Some(text) = value_to_text(value)? else {
            continue;
        };

        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(&text);
        found = true;
    }

    if !found {
        bail!("Hugging Face dataset rows did not include column '{column}'");
    }

    Ok(())
}

fn value_to_text(value: &Value) -> Result<Option<String>> {
    match value {
        Value::Null => Ok(None),
        Value::String(text) => Ok(Some(text.clone())),
        Value::Bool(value) => Ok(Some(value.to_string())),
        Value::Number(value) => Ok(Some(value.to_string())),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(value)
            .map(Some)
            .context("failed to serialize dataset cell"),
    }
}

fn required_value(key: &str, value: String) -> Result<String> {
    if value.is_empty() {
        bail!("Hugging Face dataset URI parameter '{key}' must not be empty");
    }

    Ok(value)
}

fn parse_usize_param(key: &str, value: &str) -> Result<usize> {
    value
        .parse()
        .with_context(|| format!("invalid Hugging Face dataset URI parameter '{key}={value}'"))
}

fn parse_positive_usize_param(key: &str, value: &str) -> Result<usize> {
    let parsed = parse_usize_param(key, value)?;
    if parsed == 0 {
        bail!("Hugging Face dataset URI parameter '{key}' must be greater than zero");
    }

    Ok(parsed)
}

fn percent_encode(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }

    encoded
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                decoded.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                if let Ok(hex) = std::str::from_utf8(&bytes[index + 1..index + 3])
                    && let Ok(byte) = u8::from_str_radix(hex, 16)
                {
                    decoded.push(byte);
                    index += 3;
                    continue;
                }
                decoded.push(bytes[index]);
                index += 1;
            }
            byte => {
                decoded.push(byte);
                index += 1;
            }
        }
    }

    String::from_utf8_lossy(&decoded).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ignores_local_paths() {
        assert_eq!(
            None,
            HuggingFaceDatasetSpec::parse("data/input.txt").unwrap()
        );
    }

    #[test]
    fn parse_hf_uri_uses_defaults() {
        let spec = HuggingFaceDatasetSpec::parse("hf://Salesforce/wikitext")
            .unwrap()
            .unwrap();

        assert_eq!("Salesforce/wikitext", spec.dataset);
        assert_eq!("default", spec.config);
        assert_eq!("train", spec.split);
        assert_eq!("text", spec.column);
        assert_eq!(0, spec.offset);
        assert_eq!(1_000, spec.rows);
    }

    #[test]
    fn parse_hf_uri_accepts_options() {
        let spec = HuggingFaceDatasetSpec::parse(
            "hf://allenai/c4?config=en&split=validation&column=content&offset=10&limit=25",
        )
        .unwrap()
        .unwrap();

        assert_eq!("allenai/c4", spec.dataset);
        assert_eq!("en", spec.config);
        assert_eq!("validation", spec.split);
        assert_eq!("content", spec.column);
        assert_eq!(10, spec.offset);
        assert_eq!(25, spec.rows);
    }

    #[test]
    fn rows_url_encodes_dataset_query_values() {
        let spec = HuggingFaceDatasetSpec {
            dataset: "Salesforce/wikitext".to_string(),
            config: "wikitext-2-raw-v1".to_string(),
            split: "train".to_string(),
            column: "text".to_string(),
            offset: 0,
            rows: 10,
        };

        assert_eq!(
            "https://datasets-server.huggingface.co/rows?dataset=Salesforce%2Fwikitext&config=wikitext-2-raw-v1&split=train&offset=5&length=10",
            spec.rows_url(5, 10)
        );
    }

    #[test]
    fn append_column_text_concatenates_string_rows() {
        let rows = vec![
            RowEnvelope {
                row: HashMap::from([("text".to_string(), Value::String("first".to_string()))]),
            },
            RowEnvelope {
                row: HashMap::from([("text".to_string(), Value::String("second".to_string()))]),
            },
        ];
        let mut output = String::new();

        append_column_text(&mut output, &rows, "text").unwrap();

        assert_eq!("first\nsecond", output);
    }

    #[test]
    fn append_column_text_reports_missing_column() {
        let rows = vec![RowEnvelope {
            row: HashMap::from([("body".to_string(), Value::String("text".to_string()))]),
        }];
        let mut output = String::new();

        let err = append_column_text(&mut output, &rows, "text").unwrap_err();

        assert!(err.to_string().contains("did not include column 'text'"));
    }
}
