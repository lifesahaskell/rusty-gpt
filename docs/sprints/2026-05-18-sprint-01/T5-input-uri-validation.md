# S1-T5 — Validate and constrain `--input` / `hf://` URIs before fetching

- **Value:** security
- **Size:** S (half day)
- **Suggested agent:** principal-security-engineer
- **Depends on:** —
- **Blocks:** —

## Context

`src/bin/train-tokenizer.rs` and the main `--input` flag both accept either a local path or an `hf://` dataset URI. The fetcher pulls arbitrary content from Hugging Face into `data/huggingface-cache/`. Without validation, this is an SSRF-shaped attack surface if anything in the CI or compose stack ever takes `--input` from an external source (PR descriptions, Slack-triggered runs, CI matrix configs).

The risk is low **today** (the binary is run from a developer shell) but the dev-container and compose stack work in recent commits indicates the surface is expanding. Closing this in S1 — while the rate-limit and path-traversal work happens in S2 — gives a clean, narrow security PR rather than bundling everything later.

## Goal

Enforce a strict, well-documented validation pass on every `--input` value before any I/O happens, with clear errors and a focused regression test.

## Acceptance criteria

- Local path inputs: must resolve to a file (not a symlink to outside the project root unless explicitly allowed), must be readable, and must not exceed a configurable max size (default 1 GiB) — fail fast otherwise.
- `hf://` URIs: must parse to `hf://<org>/<dataset>` (or `hf://<org>/<dataset>@<revision>`), reject anything with a `..` segment, reject `file://`, `ssh://`, raw IPs, or non-ASCII components. The allowed character set is `[A-Za-z0-9._-]` for org/dataset/revision.
- Any other scheme (`http://`, `https://`, `ftp://`, `file://`) is rejected at parse time with a clear error that names the scheme.
- The error message points the user at the supported forms (e.g. `"--input must be a local path or hf://<org>/<dataset>[@<revision>]"`).
- New unit tests in `src/runtime_assets.rs` (or wherever `--input` parsing lives post-T1) cover: valid local path, valid `hf://` URI with and without revision, rejected schemes, rejected `..` segments, rejected non-ASCII, rejected oversized files (use a small synthetic limit).
- `cargo test --test default_runtime` still passes.
- The `data/input.txt` default still works without explicit `--input`.

## Implementation notes

- Do not introduce a new HTTP client just to validate — the existing `hf://` loader can stay; this task adds a validation layer in front of it.
- Resist the urge to add a URL allowlist for `hf://` (e.g. only specific orgs). That's a policy decision that belongs in compose config, not in the binary.
- For path inputs, use `std::fs::metadata().len()` for the size check rather than reading the file first.
- The validation function should return a typed `InputSource` enum (`InputSource::Local(PathBuf)` / `InputSource::HuggingFace { org, dataset, revision }`) so downstream code branches on a parsed value, not a raw string.

## Definition of done

- PR merged to `main`.
- The `--input` documentation in CLAUDE.md gains a one-line note on what's accepted.
- One regression test per rejected case (table-driven is fine).
