use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const DATASETS_SERVER_ROWS_URL: &str = "https://datasets-server.huggingface.co/rows";
const DEFAULT_CONFIG: &str = "default";
const DEFAULT_SPLIT: &str = "train";
const DEFAULT_COLUMN: &str = "text";
const DEFAULT_ROWS: usize = 1_000;
const MAX_PAGE_ROWS: usize = 100;
const DEFAULT_CACHE_DIR: &str = "data/huggingface-cache";
const CACHE_DIR_ENV: &str = "RUSTY_GPT_HF_DATASET_CACHE";
const REQUEST_DELAY_ENV: &str = "RUSTY_GPT_HF_REQUEST_DELAY_MS";
const MAX_RETRIES_ENV: &str = "RUSTY_GPT_HF_MAX_RETRIES";
const RETRY_BASE_DELAY_ENV: &str = "RUSTY_GPT_HF_RETRY_BASE_DELAY_MS";
const DEFAULT_REQUEST_DELAY_MS: u64 = 250;
const DEFAULT_MAX_RETRIES: usize = 6;
const DEFAULT_RETRY_BASE_DELAY_MS: u64 = 1_000;

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

    fn cache_key(&self) -> String {
        format!(
            "dataset={}\nconfig={}\nsplit={}\ncolumn={}\noffset={}\nrows={}\n",
            self.dataset, self.config, self.split, self.column, self.offset, self.rows
        )
    }

    fn cache_path(&self, cache_dir: &Path) -> PathBuf {
        let digest_hex: String = Sha256::digest(self.cache_key())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let dataset = safe_cache_component(&self.dataset);
        let config = safe_cache_component(&self.config);
        let split = safe_cache_component(&self.split);
        let column = safe_cache_component(&self.column);
        cache_dir.join(format!(
            "{dataset}__{config}__{split}__{column}__offset-{}__rows-{}__{digest_hex}.txt",
            self.offset, self.rows,
        ))
    }

    fn page_cache_path(&self, cache_dir: &Path, offset: usize, length: usize) -> PathBuf {
        let digest_hex: String = Sha256::digest(self.cache_key())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let dataset = safe_cache_component(&self.dataset);
        let config = safe_cache_component(&self.config);
        let split = safe_cache_component(&self.split);
        cache_dir.join("pages").join(format!(
            "{dataset}__{config}__{split}__offset-{offset}__length-{length}__{digest_hex}.json"
        ))
    }
}

pub fn load_text_from_uri(input: &str) -> Result<Option<String>> {
    let Some(spec) = HuggingFaceDatasetSpec::parse(input)? else {
        return Ok(None);
    };

    load_text_cached(&spec, &cache_dir()).map(Some)
}

pub fn load_text_from_uri_with_cache_dir(
    input: &str,
    cache_dir: impl AsRef<Path>,
) -> Result<Option<String>> {
    let Some(spec) = HuggingFaceDatasetSpec::parse(input)? else {
        return Ok(None);
    };

    load_text_cached(&spec, cache_dir.as_ref()).map(Some)
}

fn cache_dir() -> PathBuf {
    env::var(CACHE_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_CACHE_DIR))
}

fn load_text_cached(spec: &HuggingFaceDatasetSpec, cache_dir: &Path) -> Result<String> {
    let cache_path = spec.cache_path(cache_dir);
    if cache_path.is_file() {
        let cached = fs::read_to_string(&cache_path).with_context(|| {
            format!(
                "failed to read cached Hugging Face dataset from {:?}",
                cache_path
            )
        })?;
        if !cached.trim().is_empty() {
            return Ok(cached);
        }
    }

    let text = load_text(spec, cache_dir)?;
    if let Some(parent) = cache_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create Hugging Face dataset cache directory {:?}",
                parent
            )
        })?;
    }
    fs::write(&cache_path, &text).with_context(|| {
        format!(
            "failed to write Hugging Face dataset cache to {:?}",
            cache_path
        )
    })?;

    Ok(text)
}

fn load_text(spec: &HuggingFaceDatasetSpec, cache_dir: &Path) -> Result<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60))
        .user_agent("rusty-gpt/0.1 huggingface-dataset-loader")
        .build()
        .context("failed to build Hugging Face dataset HTTP client")?;
    let retry_policy = RetryPolicy::from_env();
    let request_delay = env_duration_ms(REQUEST_DELAY_ENV, DEFAULT_REQUEST_DELAY_MS);

    let mut output = String::new();
    let mut fetched_rows = 0usize;
    let mut offset = spec.offset;
    let mut total_rows = None;

    while fetched_rows < spec.rows {
        let length = (spec.rows - fetched_rows).min(MAX_PAGE_ROWS);
        let url = spec.rows_url(offset, length);
        let response = load_rows_page(&client, &url, spec, cache_dir, offset, length, retry_policy)
            .with_context(|| format!("failed to load Hugging Face dataset rows from {url}"))?;

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

        if !request_delay.is_zero() && fetched_rows < spec.rows {
            thread::sleep(request_delay);
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

#[derive(Debug, Clone, Copy)]
struct RetryPolicy {
    max_retries: usize,
    base_delay: Duration,
}

impl RetryPolicy {
    fn from_env() -> Self {
        Self {
            max_retries: env_usize(MAX_RETRIES_ENV, DEFAULT_MAX_RETRIES),
            base_delay: env_duration_ms(RETRY_BASE_DELAY_ENV, DEFAULT_RETRY_BASE_DELAY_MS),
        }
    }
}

fn load_rows_page(
    client: &reqwest::blocking::Client,
    url: &str,
    spec: &HuggingFaceDatasetSpec,
    cache_dir: &Path,
    offset: usize,
    length: usize,
    retry_policy: RetryPolicy,
) -> Result<RowsResponse> {
    let page_cache_path = spec.page_cache_path(cache_dir, offset, length);
    if page_cache_path.is_file() {
        let cached = fs::read_to_string(&page_cache_path).with_context(|| {
            format!(
                "failed to read cached Hugging Face dataset page from {:?}",
                page_cache_path
            )
        })?;
        return serde_json::from_str(&cached).with_context(|| {
            format!(
                "failed to decode cached Hugging Face dataset page {:?}",
                page_cache_path
            )
        });
    }

    let mut attempt = 0usize;
    loop {
        let response = client
            .get(url)
            .send()
            .with_context(|| format!("failed to fetch Hugging Face dataset rows from {url}"))?;
        let status = response.status();
        if status.is_success() {
            let body = response
                .text()
                .with_context(|| format!("failed to read Hugging Face dataset rows from {url}"))?;
            let parsed = serde_json::from_str::<RowsResponse>(&body).with_context(|| {
                format!("failed to decode Hugging Face dataset rows from {url}")
            })?;
            write_page_cache(&page_cache_path, &body)?;
            return Ok(parsed);
        }

        let headers = response.headers().clone();
        let body = response.text().unwrap_or_default();
        if should_retry_status(status) && attempt < retry_policy.max_retries {
            let delay = retry_delay(&headers, retry_policy, attempt);
            thread::sleep(delay);
            attempt += 1;
            continue;
        }

        bail!(
            "Hugging Face dataset rows request failed for {url}: HTTP {status}; {}",
            response_body_summary(&body)
        );
    }
}

fn write_page_cache(page_cache_path: &Path, body: &str) -> Result<()> {
    if let Some(parent) = page_cache_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create Hugging Face dataset page cache directory {:?}",
                parent
            )
        })?;
    }

    fs::write(page_cache_path, body).with_context(|| {
        format!(
            "failed to write Hugging Face dataset page cache to {:?}",
            page_cache_path
        )
    })
}

fn should_retry_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn retry_delay(headers: &HeaderMap, retry_policy: RetryPolicy, attempt: usize) -> Duration {
    retry_after_delay(headers)
        .unwrap_or_else(|| exponential_backoff(retry_policy.base_delay, attempt))
}

fn retry_after_delay(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    let seconds = value.parse::<u64>().ok()?;
    Some(Duration::from_secs(seconds))
}

fn exponential_backoff(base_delay: Duration, attempt: usize) -> Duration {
    let multiplier = 1u32.checked_shl(attempt.min(6) as u32).unwrap_or(64);
    base_delay.saturating_mul(multiplier)
}

fn response_body_summary(body: &str) -> String {
    let summary: String = body
        .chars()
        .filter(|ch| !ch.is_control() || *ch == '\n' || *ch == '\t')
        .take(240)
        .collect();
    if summary.trim().is_empty() {
        "empty response body".to_string()
    } else {
        format!("response body: {}", summary.trim())
    }
}

fn env_duration_ms(key: &str, default_ms: u64) -> Duration {
    Duration::from_millis(env_u64(key, default_ms))
}

fn env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
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

fn safe_cache_component(value: &str) -> String {
    let mut component = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            component.push(ch);
        } else {
            component.push('-');
        }
    }

    if component.is_empty() {
        "default".to_string()
    } else {
        component.chars().take(64).collect()
    }
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
    fn cache_path_changes_when_requested_rows_change() {
        let cache_dir =
            std::env::temp_dir().join(format!("rusty-gpt-hf-cache-path-{}", std::process::id()));
        let mut first = HuggingFaceDatasetSpec::parse(
            "hf://Salesforce/wikitext?config=wikitext-2-raw-v1&rows=100",
        )
        .unwrap()
        .unwrap();
        let first_path = first.cache_path(&cache_dir);

        first.rows = 200;

        assert_ne!(first_path, first.cache_path(&cache_dir));
    }

    #[test]
    fn page_cache_path_changes_by_offset_and_length() {
        let cache_dir = std::env::temp_dir().join(format!(
            "rusty-gpt-hf-page-cache-path-{}",
            std::process::id()
        ));
        let spec = HuggingFaceDatasetSpec::parse(
            "hf://Salesforce/wikitext?config=wikitext-2-raw-v1&rows=1000",
        )
        .unwrap()
        .unwrap();

        let first_path = spec.page_cache_path(&cache_dir, 0, 100);
        let second_path = spec.page_cache_path(&cache_dir, 100, 100);
        let shorter_path = spec.page_cache_path(&cache_dir, 0, 50);

        assert_ne!(first_path, second_path);
        assert_ne!(first_path, shorter_path);
    }

    #[test]
    fn load_text_from_uri_reads_cache_before_fetching() {
        let cache_dir =
            std::env::temp_dir().join(format!("rusty-gpt-hf-cache-read-{}", std::process::id()));
        let uri =
            "hf://Salesforce/wikitext?config=wikitext-2-raw-v1&split=train&column=text&rows=7";
        let spec = HuggingFaceDatasetSpec::parse(uri).unwrap().unwrap();
        let cache_path = spec.cache_path(&cache_dir);
        fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
        fs::write(&cache_path, "cached dataset text").unwrap();

        let loaded = load_text_from_uri_with_cache_dir(uri, &cache_dir)
            .unwrap()
            .unwrap();

        assert_eq!("cached dataset text", loaded);

        let _ = fs::remove_file(cache_path);
        let _ = fs::remove_dir_all(cache_dir);
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

    #[test]
    fn retry_after_delay_reads_seconds_header() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, "3".parse().unwrap());

        assert_eq!(Some(Duration::from_secs(3)), retry_after_delay(&headers));
    }

    #[test]
    fn exponential_backoff_doubles_base_delay() {
        assert_eq!(
            Duration::from_secs(4),
            exponential_backoff(Duration::from_secs(1), 2)
        );
    }
}
