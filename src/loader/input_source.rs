//! Strict validation of the `--input` / `--corpus` URI before any I/O.
//!
//! The CLI binaries accept either a local filesystem path or a Hugging Face
//! dataset reference of the form `hf://<org>/<dataset>[@<revision>]`. Both
//! shapes flow into `load_text_from_uri` and `fs::read_to_string`, so any
//! validation has to happen *before* we touch the network or the filesystem.
//!
//! [`InputSource::parse`] performs purely syntactic validation (scheme,
//! character set, dataset/revision shape) and never touches I/O. Callers that
//! resolve a local file should additionally invoke
//! [`InputSource::validate_local_size`] to enforce the maximum file size cap
//! based on `std::fs::metadata().len()` — without reading the file content.

use std::fmt;
use std::path::{Path, PathBuf};

/// Default cap for local input files (1 GiB). The cap is configurable per-call
/// so that tests can use a small synthetic threshold instead of touching this
/// default.
pub const DEFAULT_MAX_LOCAL_INPUT_BYTES: u64 = 1024 * 1024 * 1024;

/// A parsed and validated input reference.
///
/// Construct via [`InputSource::parse`]; the raw string is **never** trusted by
/// downstream code — only the enum variants are. This forces every consumer of
/// `--input` or `--corpus` to branch on the parsed shape rather than re-parsing
/// the original string at every call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    /// A path on the local filesystem.
    Local(PathBuf),
    /// A Hugging Face dataset reference.
    HuggingFace {
        org: String,
        dataset: String,
        revision: Option<String>,
        /// Canonical, already-validated query string in `k=v&k=v` form
        /// (no leading `?`). `None` if the URI had no query suffix.
        query: Option<String>,
    },
}

/// Query keys the downstream `huggingface` loader knows how to interpret.
/// Anything else is rejected at parse time.
const ALLOWED_QUERY_KEYS: &[&str] = &["config", "split", "column", "offset", "rows", "limit"];

/// Errors returned when an input URI fails strict validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSourceError {
    /// The input was the empty string.
    Empty,
    /// A scheme other than `hf://` was provided (e.g. `http://`, `file://`).
    UnsupportedScheme { scheme: String },
    /// The `hf://` URI did not match `hf://<org>/<dataset>[@<revision>]`.
    MalformedHuggingFaceUri { reason: String, raw: String },
    /// A `hf://` URI segment contained characters outside `[A-Za-z0-9._-]` or
    /// non-ASCII bytes.
    DisallowedHuggingFaceCharacters {
        component: &'static str,
        value: String,
    },
    /// A `hf://` URI segment was `.` or `..` — a path-traversal attempt.
    HuggingFaceTraversalSegment {
        component: &'static str,
        value: String,
    },
    /// A query parameter used a key outside the allowlist (`config`, `split`,
    /// `column`, `offset`, `rows`, `limit`).
    DisallowedHuggingFaceQueryKey { key: String },
    /// A query parameter value was empty, contained disallowed characters, or
    /// was a traversal segment.
    DisallowedHuggingFaceQueryValue {
        key: String,
        value: String,
        reason: &'static str,
    },
    /// The query string itself was structurally malformed (e.g. a parameter
    /// missing `=` or an empty `&&` segment).
    MalformedHuggingFaceQuery { reason: String, raw: String },
    /// A local file exceeded the configured maximum size.
    LocalFileTooLarge {
        path: PathBuf,
        size_bytes: u64,
        max_bytes: u64,
    },
    /// A local path did not point to a regular file.
    LocalPathNotAFile { path: PathBuf },
    /// `std::fs::metadata` failed (file missing, permission denied, etc.).
    LocalMetadataUnreadable { path: PathBuf, source: String },
}

impl fmt::Display for InputSourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(
                formatter,
                "--input must be a local path or hf://<org>/<dataset>[@<revision>]"
            ),
            Self::UnsupportedScheme { scheme } => write!(
                formatter,
                "unsupported input URI scheme '{scheme}'; --input must be a local path or hf://<org>/<dataset>[@<revision>]"
            ),
            Self::MalformedHuggingFaceUri { reason, raw } => write!(
                formatter,
                "invalid Hugging Face dataset URI '{raw}': {reason}; expected hf://<org>/<dataset>[@<revision>]"
            ),
            Self::DisallowedHuggingFaceCharacters { component, value } => write!(
                formatter,
                "Hugging Face dataset {component} '{value}' contains characters outside [A-Za-z0-9._-]; expected hf://<org>/<dataset>[@<revision>]"
            ),
            Self::HuggingFaceTraversalSegment { component, value } => write!(
                formatter,
                "Hugging Face dataset {component} '{value}' is not a valid identifier; expected hf://<org>/<dataset>[@<revision>]"
            ),
            Self::LocalFileTooLarge {
                path,
                size_bytes,
                max_bytes,
            } => write!(
                formatter,
                "local input file {} is {size_bytes} bytes which exceeds the configured maximum of {max_bytes} bytes",
                path.display()
            ),
            Self::LocalPathNotAFile { path } => write!(
                formatter,
                "local input path {} is not a regular file",
                path.display()
            ),
            Self::LocalMetadataUnreadable { path, source } => write!(
                formatter,
                "failed to read metadata for local input {}: {source}",
                path.display()
            ),
            Self::DisallowedHuggingFaceQueryKey { key } => write!(
                formatter,
                "unsupported Hugging Face query parameter '{key}'; allowed keys are {ALLOWED_QUERY_KEYS:?}"
            ),
            Self::DisallowedHuggingFaceQueryValue { key, value, reason } => write!(
                formatter,
                "invalid value for Hugging Face query parameter '{key}={value}': {reason}; values must match [A-Za-z0-9._-]"
            ),
            Self::MalformedHuggingFaceQuery { reason, raw } => write!(
                formatter,
                "invalid Hugging Face query string in '{raw}': {reason}; expected ?key=value[&key=value...]"
            ),
        }
    }
}

impl std::error::Error for InputSourceError {}

impl InputSource {
    /// Parse a raw `--input` / `--corpus` string into a validated
    /// [`InputSource`].
    ///
    /// This call is **pure**: it performs syntactic validation only. No
    /// network or filesystem activity occurs.
    pub fn parse(raw: &str) -> Result<Self, InputSourceError> {
        if raw.is_empty() {
            return Err(InputSourceError::Empty);
        }

        if let Some(rest) = raw.strip_prefix("hf://") {
            return parse_huggingface(rest, raw);
        }

        // Reject anything that *looks* like an unknown URI scheme. This means
        // `<scheme>://...` where the scheme is purely ASCII alpha. We do this
        // rather than naively checking for `://` so that a Windows-style local
        // path with a drive letter or a relative path containing a colon does
        // not get mistakenly flagged.
        if let Some(scheme) = unsupported_scheme(raw) {
            return Err(InputSourceError::UnsupportedScheme { scheme });
        }

        Ok(Self::Local(PathBuf::from(raw)))
    }

    /// For [`InputSource::Local`], verify the path points to a regular file
    /// and that its size does not exceed `max_bytes`. Uses
    /// `std::fs::metadata().len()` — does **not** read the file body.
    ///
    /// Non-local variants are no-ops.
    pub fn validate_local_size(&self, max_bytes: u64) -> Result<(), InputSourceError> {
        let Self::Local(path) = self else {
            return Ok(());
        };

        validate_local_path(path, max_bytes)
    }

    /// Convenience accessor for the metadata sidecar / log lines that want a
    /// stable string representation of the original source.
    pub fn display(&self) -> String {
        match self {
            Self::Local(path) => path.display().to_string(),
            Self::HuggingFace {
                org,
                dataset,
                revision,
                query,
            } => {
                let mut s = format!("hf://{org}/{dataset}");
                if let Some(rev) = revision {
                    s.push('@');
                    s.push_str(rev);
                }
                if let Some(q) = query {
                    s.push('?');
                    s.push_str(q);
                }
                s
            }
        }
    }

    /// For a local source, return the borrowed path.
    pub fn local_path(&self) -> Option<&Path> {
        match self {
            Self::Local(path) => Some(path.as_path()),
            _ => None,
        }
    }
}

fn validate_local_path(path: &Path, max_bytes: u64) -> Result<(), InputSourceError> {
    let metadata =
        std::fs::metadata(path).map_err(|err| InputSourceError::LocalMetadataUnreadable {
            path: path.to_path_buf(),
            source: err.to_string(),
        })?;

    if !metadata.is_file() {
        return Err(InputSourceError::LocalPathNotAFile {
            path: path.to_path_buf(),
        });
    }

    let size_bytes = metadata.len();
    if size_bytes > max_bytes {
        return Err(InputSourceError::LocalFileTooLarge {
            path: path.to_path_buf(),
            size_bytes,
            max_bytes,
        });
    }

    Ok(())
}

fn parse_huggingface(rest: &str, raw: &str) -> Result<InputSource, InputSourceError> {
    // Pull off `?query` first so it cannot bleed into the dataset or revision.
    let (without_query, query_str) = match rest.split_once('?') {
        Some((head, tail)) => (head, Some(tail)),
        None => (rest, None),
    };

    // Pull the revision off the tail next so `@` cannot bleed into the
    // dataset name.
    let (without_revision, revision) = match without_query.rsplit_once('@') {
        Some((head, tail)) => (head, Some(tail)),
        None => (without_query, None),
    };

    let mut parts = without_revision.split('/');
    let org = parts.next().unwrap_or("");
    let dataset = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return Err(InputSourceError::MalformedHuggingFaceUri {
            reason: "expected exactly one '/' between org and dataset".to_string(),
            raw: raw.to_string(),
        });
    }

    if org.is_empty() {
        return Err(InputSourceError::MalformedHuggingFaceUri {
            reason: "missing organisation segment".to_string(),
            raw: raw.to_string(),
        });
    }
    if dataset.is_empty() {
        return Err(InputSourceError::MalformedHuggingFaceUri {
            reason: "missing dataset segment".to_string(),
            raw: raw.to_string(),
        });
    }

    validate_hf_segment("org", org)?;
    validate_hf_segment("dataset", dataset)?;
    if let Some(revision) = revision {
        if revision.is_empty() {
            return Err(InputSourceError::MalformedHuggingFaceUri {
                reason: "empty revision after '@'".to_string(),
                raw: raw.to_string(),
            });
        }
        validate_hf_segment("revision", revision)?;
    }

    let query = match query_str {
        None => None,
        Some(q) => Some(parse_hf_query(q, raw)?),
    };

    Ok(InputSource::HuggingFace {
        org: org.to_string(),
        dataset: dataset.to_string(),
        revision: revision.map(str::to_string),
        query,
    })
}

fn parse_hf_query(query: &str, raw: &str) -> Result<String, InputSourceError> {
    if query.is_empty() {
        return Err(InputSourceError::MalformedHuggingFaceQuery {
            reason: "empty query string after '?'".to_string(),
            raw: raw.to_string(),
        });
    }

    let mut canonical = String::with_capacity(query.len());
    for (i, pair) in query.split('&').enumerate() {
        if pair.is_empty() {
            return Err(InputSourceError::MalformedHuggingFaceQuery {
                reason: "empty parameter (consecutive '&' or trailing '&')".to_string(),
                raw: raw.to_string(),
            });
        }
        let (key, value) =
            pair.split_once('=')
                .ok_or_else(|| InputSourceError::MalformedHuggingFaceQuery {
                    reason: format!("parameter '{pair}' missing '='"),
                    raw: raw.to_string(),
                })?;
        if !ALLOWED_QUERY_KEYS.contains(&key) {
            return Err(InputSourceError::DisallowedHuggingFaceQueryKey {
                key: key.to_string(),
            });
        }
        validate_hf_query_value(key, value)?;
        if i > 0 {
            canonical.push('&');
        }
        canonical.push_str(key);
        canonical.push('=');
        canonical.push_str(value);
    }
    Ok(canonical)
}

fn validate_hf_query_value(key: &str, value: &str) -> Result<(), InputSourceError> {
    if value.is_empty() {
        return Err(InputSourceError::DisallowedHuggingFaceQueryValue {
            key: key.to_string(),
            value: value.to_string(),
            reason: "empty value",
        });
    }
    if value == "." || value == ".." {
        return Err(InputSourceError::DisallowedHuggingFaceQueryValue {
            key: key.to_string(),
            value: value.to_string(),
            reason: "traversal segment ('.' or '..')",
        });
    }
    if !value
        .bytes()
        .all(|byte| matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-'))
    {
        return Err(InputSourceError::DisallowedHuggingFaceQueryValue {
            key: key.to_string(),
            value: value.to_string(),
            reason: "character outside [A-Za-z0-9._-]",
        });
    }
    Ok(())
}

fn validate_hf_segment(component: &'static str, value: &str) -> Result<(), InputSourceError> {
    if value == "." || value == ".." {
        return Err(InputSourceError::HuggingFaceTraversalSegment {
            component,
            value: value.to_string(),
        });
    }

    if !value
        .bytes()
        .all(|byte| matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-'))
    {
        return Err(InputSourceError::DisallowedHuggingFaceCharacters {
            component,
            value: value.to_string(),
        });
    }

    Ok(())
}

/// If `raw` looks like `<scheme>://...` where `<scheme>` is ASCII alpha
/// (followed optionally by digits / `+` / `-` / `.`), return the offending
/// scheme. This deliberately matches URI grammar (RFC 3986 §3.1) so plain
/// local paths with colons are not flagged.
fn unsupported_scheme(raw: &str) -> Option<String> {
    let bytes = raw.as_bytes();
    if bytes.len() < 4 {
        return None;
    }

    // Scheme must start with an ASCII alpha.
    if !bytes[0].is_ascii_alphabetic() {
        return None;
    }

    let mut end = 1;
    while end < bytes.len()
        && matches!(bytes[end], b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'+' | b'-' | b'.')
    {
        end += 1;
    }

    if end + 2 < bytes.len() && &bytes[end..end + 3] == b"://" {
        let scheme = &raw[..end];
        // `hf` is handled above; anything else with a `://` triple is rejected.
        if !scheme.eq_ignore_ascii_case("hf") {
            return Some(scheme.to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parses_local_relative_path() {
        let parsed = InputSource::parse("data/input.txt").unwrap();
        assert_eq!(InputSource::Local(PathBuf::from("data/input.txt")), parsed);
    }

    #[test]
    fn parses_local_absolute_path() {
        let parsed = InputSource::parse("/tmp/corpus.txt").unwrap();
        assert_eq!(InputSource::Local(PathBuf::from("/tmp/corpus.txt")), parsed);
    }

    #[test]
    fn parses_hf_uri_without_revision() {
        let parsed = InputSource::parse("hf://Salesforce/wikitext").unwrap();
        assert_eq!(
            InputSource::HuggingFace {
                org: "Salesforce".to_string(),
                dataset: "wikitext".to_string(),
                revision: None,
                query: None,
            },
            parsed
        );
    }

    #[test]
    fn parses_hf_uri_with_revision() {
        let parsed = InputSource::parse("hf://allenai/c4@v1.0.0").unwrap();
        assert_eq!(
            InputSource::HuggingFace {
                org: "allenai".to_string(),
                dataset: "c4".to_string(),
                revision: Some("v1.0.0".to_string()),
                query: None,
            },
            parsed
        );
    }

    #[test]
    fn rejects_empty_input() {
        assert_eq!(Err(InputSourceError::Empty), InputSource::parse(""));
    }

    #[test]
    fn rejects_http_scheme() {
        let err = InputSource::parse("http://example.com/foo").unwrap_err();
        assert_eq!(
            InputSourceError::UnsupportedScheme {
                scheme: "http".to_string()
            },
            err
        );
    }

    #[test]
    fn rejects_https_scheme() {
        let err = InputSource::parse("https://example.com/foo").unwrap_err();
        assert_eq!(
            InputSourceError::UnsupportedScheme {
                scheme: "https".to_string()
            },
            err
        );
    }

    #[test]
    fn rejects_file_scheme() {
        let err = InputSource::parse("file:///etc/passwd").unwrap_err();
        assert_eq!(
            InputSourceError::UnsupportedScheme {
                scheme: "file".to_string()
            },
            err
        );
    }

    #[test]
    fn rejects_ssh_scheme() {
        let err = InputSource::parse("ssh://git@example.com/repo").unwrap_err();
        assert_eq!(
            InputSourceError::UnsupportedScheme {
                scheme: "ssh".to_string()
            },
            err
        );
    }

    #[test]
    fn rejects_ftp_scheme() {
        let err = InputSource::parse("ftp://example.com/foo").unwrap_err();
        assert_eq!(
            InputSourceError::UnsupportedScheme {
                scheme: "ftp".to_string()
            },
            err
        );
    }

    #[test]
    fn rejects_hf_uri_with_traversal_segment() {
        let err = InputSource::parse("hf://org/../dataset").unwrap_err();
        assert!(
            matches!(err, InputSourceError::MalformedHuggingFaceUri { .. }),
            "expected MalformedHuggingFaceUri (too many '/'), got {err:?}"
        );

        let err = InputSource::parse("hf://../dataset").unwrap_err();
        assert_eq!(
            InputSourceError::HuggingFaceTraversalSegment {
                component: "org",
                value: "..".to_string(),
            },
            err
        );

        let err = InputSource::parse("hf://org/..").unwrap_err();
        assert_eq!(
            InputSourceError::HuggingFaceTraversalSegment {
                component: "dataset",
                value: "..".to_string(),
            },
            err
        );
    }

    #[test]
    fn rejects_hf_uri_with_non_ascii_org() {
        // Cyrillic "тест".
        let err = InputSource::parse("hf://тест/dataset").unwrap_err();
        assert!(
            matches!(
                err,
                InputSourceError::DisallowedHuggingFaceCharacters {
                    component: "org",
                    ..
                }
            ),
            "expected DisallowedHuggingFaceCharacters(org), got {err:?}"
        );
    }

    #[test]
    fn rejects_hf_uri_with_non_ascii_dataset() {
        let err = InputSource::parse("hf://org/тест").unwrap_err();
        assert!(
            matches!(
                err,
                InputSourceError::DisallowedHuggingFaceCharacters {
                    component: "dataset",
                    ..
                }
            ),
            "expected DisallowedHuggingFaceCharacters(dataset), got {err:?}"
        );
    }

    #[test]
    fn rejects_hf_uri_with_missing_dataset() {
        let err = InputSource::parse("hf://org").unwrap_err();
        assert!(matches!(
            err,
            InputSourceError::MalformedHuggingFaceUri { .. }
        ));

        let err = InputSource::parse("hf://org/").unwrap_err();
        assert!(matches!(
            err,
            InputSourceError::MalformedHuggingFaceUri { .. }
        ));
    }

    #[test]
    fn rejects_hf_uri_with_empty_revision() {
        let err = InputSource::parse("hf://org/dataset@").unwrap_err();
        assert!(matches!(
            err,
            InputSourceError::MalformedHuggingFaceUri { .. }
        ));
    }

    #[test]
    fn rejects_hf_uri_with_extra_path_segment() {
        // Reject things like `hf://org/dataset/extra` that the legacy loader
        // would have happily passed through as a dataset id of
        // `org/dataset/extra`.
        let err = InputSource::parse("hf://org/dataset/extra").unwrap_err();
        assert!(matches!(
            err,
            InputSourceError::MalformedHuggingFaceUri { .. }
        ));
    }

    #[test]
    fn accepts_hf_uri_with_allowlisted_query_parameters() {
        let parsed = InputSource::parse(
            "hf://Shuu12121/rust-treesitter-dedupe-filtered-datasetsV2?split=train&column=code&rows=50000",
        )
        .unwrap();
        assert_eq!(
            InputSource::HuggingFace {
                org: "Shuu12121".to_string(),
                dataset: "rust-treesitter-dedupe-filtered-datasetsV2".to_string(),
                revision: None,
                query: Some("split=train&column=code&rows=50000".to_string()),
            },
            parsed
        );

        // Revision + query together.
        let parsed = InputSource::parse("hf://allenai/c4@v1.0.0?config=en&split=train").unwrap();
        assert_eq!(
            InputSource::HuggingFace {
                org: "allenai".to_string(),
                dataset: "c4".to_string(),
                revision: Some("v1.0.0".to_string()),
                query: Some("config=en&split=train".to_string()),
            },
            parsed
        );
    }

    #[test]
    fn rejects_hf_uri_with_unknown_query_key() {
        let err = InputSource::parse("hf://org/dataset?config=en&shell_exec=cat%20/etc/passwd")
            .unwrap_err();
        assert_eq!(
            InputSourceError::DisallowedHuggingFaceQueryKey {
                key: "shell_exec".to_string()
            },
            err
        );
    }

    #[test]
    fn rejects_hf_uri_with_disallowed_query_value() {
        // Spaces, %-encoding, slashes are not in [A-Za-z0-9._-].
        let err = InputSource::parse("hf://org/dataset?config=en glish").unwrap_err();
        assert!(
            matches!(
                &err,
                InputSourceError::DisallowedHuggingFaceQueryValue { key, .. } if key == "config"
            ),
            "expected DisallowedHuggingFaceQueryValue(config), got {err:?}"
        );

        let err = InputSource::parse("hf://org/dataset?split=..").unwrap_err();
        assert!(
            matches!(
                &err,
                InputSourceError::DisallowedHuggingFaceQueryValue { key, reason, .. }
                    if key == "split" && reason.contains("traversal")
            ),
            "expected traversal rejection, got {err:?}"
        );
    }

    #[test]
    fn rejects_malformed_query_string() {
        let err = InputSource::parse("hf://org/dataset?").unwrap_err();
        assert!(
            matches!(err, InputSourceError::MalformedHuggingFaceQuery { .. }),
            "expected MalformedHuggingFaceQuery, got {err:?}"
        );

        let err = InputSource::parse("hf://org/dataset?config").unwrap_err();
        assert!(
            matches!(&err, InputSourceError::MalformedHuggingFaceQuery { reason, .. }
                if reason.contains("missing '='")),
            "expected missing-'=' rejection, got {err:?}"
        );

        let err = InputSource::parse("hf://org/dataset?config=en&&split=train").unwrap_err();
        assert!(
            matches!(&err, InputSourceError::MalformedHuggingFaceQuery { reason, .. }
                if reason.contains("empty parameter")),
            "expected empty-parameter rejection, got {err:?}"
        );
    }

    #[test]
    fn rejects_local_file_exceeding_max_size() {
        let dir = std::env::temp_dir().join(format!(
            "rusty-gpt-input-source-too-large-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corpus.txt");
        fs::write(&path, b"the quick brown fox").unwrap();

        let parsed = InputSource::parse(path.to_str().unwrap()).unwrap();
        let err = parsed.validate_local_size(8).unwrap_err();

        assert!(
            matches!(
                err,
                InputSourceError::LocalFileTooLarge {
                    size_bytes: 19,
                    max_bytes: 8,
                    ..
                }
            ),
            "expected LocalFileTooLarge(19, 8), got {err:?}"
        );

        // Boundary check: equal to the max is allowed.
        parsed.validate_local_size(19).unwrap();
        parsed.validate_local_size(1024).unwrap();

        let _ = fs::remove_file(path);
        let _ = fs::remove_dir(dir);
    }

    #[test]
    fn validate_local_size_is_noop_for_hf_uri() {
        let parsed = InputSource::parse("hf://org/dataset").unwrap();
        parsed.validate_local_size(1).unwrap();
    }

    #[test]
    fn validate_local_size_reports_missing_file() {
        let parsed = InputSource::parse("/nonexistent/rusty-gpt-input-source-missing.txt").unwrap();
        let err = parsed
            .validate_local_size(DEFAULT_MAX_LOCAL_INPUT_BYTES)
            .unwrap_err();
        assert!(
            matches!(err, InputSourceError::LocalMetadataUnreadable { .. }),
            "expected LocalMetadataUnreadable, got {err:?}"
        );
    }

    #[test]
    fn display_round_trips_each_variant() {
        assert_eq!(
            "data/input.txt",
            InputSource::parse("data/input.txt").unwrap().display()
        );
        assert_eq!(
            "hf://org/dataset",
            InputSource::parse("hf://org/dataset").unwrap().display()
        );
        assert_eq!(
            "hf://org/dataset@v1",
            InputSource::parse("hf://org/dataset@v1").unwrap().display()
        );
        assert_eq!(
            "hf://org/dataset?config=en&split=train&rows=100",
            InputSource::parse("hf://org/dataset?config=en&split=train&rows=100")
                .unwrap()
                .display()
        );
        assert_eq!(
            "hf://org/dataset@v1?config=en",
            InputSource::parse("hf://org/dataset@v1?config=en")
                .unwrap()
                .display()
        );
    }

    #[test]
    fn schemeless_path_with_colon_is_accepted() {
        // Windows-style drive letter / paths with colons should not be
        // mistaken for an unknown URI scheme — only `scheme://` triggers
        // rejection.
        let parsed = InputSource::parse("C:/data/input.txt").unwrap();
        assert!(matches!(parsed, InputSource::Local(_)));
    }
}
