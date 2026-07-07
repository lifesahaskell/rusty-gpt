# S2-T3 — Confine `--checkpoint` paths to `checkpoints/` (prevent traversal)

- **Value:** security
- **Size:** S (half day)
- **Suggested agent:** principal-security-engineer
- **Depends on:** —
- **Blocks:** —

## Context

`--checkpoint` and `RUSTY_GPT_MINIGPT_CHECKPOINT` accept an arbitrary path (without extension) that `NamedMpkFileRecorder` then suffixes with `.mpk`. In the binary-from-shell case this is fine, but:

- The compose stack and dev-container mount work expose paths from CI / container env into the binary.
- A `--checkpoint /etc/passwd` save would happily write `/etc/passwd.mpk`.
- A `--checkpoint ../../some-other-project/secret` load would happily read out-of-tree.

This is low-likelihood-of-exploit, low-effort-to-fix, and is the kind of thing security review tools flag immediately.

## Goal

Enforce that every `--checkpoint` value resolves to a path **inside** the `DEFAULT_CHECKPOINT_DIR` (currently `checkpoints/`, see `src/main.rs:31`) and reject anything else at parse time with a clear error.

## Acceptance criteria

- New helper `validate_checkpoint_path(input: &str, root: &Path) -> Result<PathBuf>` canonicalizes both `input` and `root`, then enforces `input.starts_with(root)`. The function is small, pure, and tested in isolation.
- The check is applied to both `--checkpoint` and `RUSTY_GPT_MINIGPT_CHECKPOINT`, for both save and load paths.
- Symlinks that point outside `checkpoints/` are rejected (canonicalize before the `starts_with` check).
- Relative paths that resolve inside `checkpoints/` (e.g. `checkpoints/mini_gpt`, `mini_gpt` if `checkpoints/` is the working dir convention) continue to work.
- Error message names the violation clearly: `"checkpoint path must be inside checkpoints/ (got: <input>, resolved: <canonical>)"`.
- New unit tests cover: valid relative path, valid absolute path inside dir, `..` traversal rejected, symlink outside rejected, non-existent path inside the dir accepted for save (canonicalize the parent, not the file).
- `--load-latest-checkpoint` continues to work — it already scans `DEFAULT_CHECKPOINT_DIR` and is unaffected.
- `cargo test --test default_runtime` still passes.

## Implementation notes

- `std::fs::canonicalize` requires the path to exist. For **save** paths (file doesn't exist yet), canonicalize the parent directory and append the filename — never canonicalize the full path before write.
- Be careful on Windows: path comparison is case-insensitive; use `same-file` crate or normalize both sides identically.
- Do **not** introduce a runtime allowlist of permitted checkpoint subdirectories — the policy is "inside the configured root, full stop." Anything more configurable belongs in a future task with a real use case.
- Consider a `--checkpoint-root <DIR>` escape hatch with a loud warning, so power-users with multi-project checkpoint stores aren't blocked. Mark it `#[arg(hide = true)]` if Clap supports it, so casual `--help` doesn't show it.

## Definition of done

- PR merged.
- CLAUDE.md "Gotchas" gets a new entry: `--checkpoint` is confined to `checkpoints/` by default, with the escape-hatch flag named.
- A regression test for the rejected-traversal case lives next to the validator.
