# S4-T3 — Derive model-shape flags at `--serve` time from the checkpoint's metadata sidecar

- **Value:** product
- **Size:** M (1–2 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** —
- **Blocks:** —

## Context

`--serve` with `--load-checkpoint` or `--load-latest-checkpoint` requires the user to also pass `--embed-dim`, `--num-heads`, `--num-layers`, and `--block-size` matching the trained model. Without them, `RuntimeConfig` builds a default-shape `MiniGptConfig` (128d/4h/4L/128) and the strict metadata loader bails with a diff-style mismatch error naming each divergent field.

The metadata sidecar (`<checkpoint>.metadata.json`) already contains all four of these values — written by `src/model/persistence.rs` at checkpoint save time. The information needed to avoid the trap is already on disk; the server startup path just does not use it.

This caused a wasted debug cycle during the v2 serve attempt. For v3 the trained checkpoint will be a non-default shape (6L/256d/8h/block-512), making this failure mode even more likely to recur.

## Goal

When `--serve` is combined with `--load-checkpoint` or `--load-latest-checkpoint`, read the checkpoint's `.metadata.json` sidecar before constructing `MiniGptConfig` and apply the stored shape values as defaults for any shape flags the user did not explicitly provide. Explicitly-provided flags win; the sidecar fills in the gaps.

## Acceptance criteria

- `cargo run -- --serve --load-latest-checkpoint` (no shape flags) starts the server successfully if a valid sidecar exists, using the shape from the sidecar.
- `cargo run -- --serve --load-latest-checkpoint --embed-dim 512` (partial override) uses `embed_dim=512` from the flag and the remaining shape fields from the sidecar.
- If the sidecar is missing (legacy checkpoint without `.metadata.json`), the server falls back to the current behavior (default-shape config + lenient loader) and logs a warning to stderr: `"No metadata sidecar found for checkpoint <path>; using default hyperparameters. If the checkpoint was trained with non-default shape, the server may fail to load it."`
- The strict metadata loader is **not** used in this path — use `load_model_with_metadata_validation` (lenient, tolerates a missing sidecar). Strict validation is reserved for production paths where a missing sidecar is an error.
- The sidecar-derived shape is logged at INFO level on server startup: `"Loaded checkpoint shape from sidecar: embed_dim=256, num_heads=8, num_layers=6, block_size=512"`.
- `--serve` without any `--load-*` flag is unaffected (fresh template, default shape, no sidecar read).
- `--load-checkpoint` and `--load-latest-checkpoint` without `--serve` are unaffected (training-path behavior unchanged).
- Unit test: given a mock sidecar JSON with non-default shape, assert that the resolved `Hyperparameters` struct contains the sidecar values when no CLI flags override them.
- `cargo test` passes, including `tests/default_runtime.rs`.

## Implementation notes

- The sidecar is parsed in `src/model/persistence.rs`. The relevant fields are `embed_dim`, `num_heads`, `num_layers`, and `block_size` (the `hyperparameters` sub-object in the JSON).
- The resolution order should be: (1) explicit CLI flag → (2) sidecar value → (3) compiled-in default. The `RuntimeConfig` / `Hyperparameters::from_env_and_overrides` path in `src/runtime_config.rs` is where this merging should happen, or it can be applied as a post-parse fixup in the server startup branch of `main.rs` / `runtime_orchestration.rs`. The agent should choose the approach that minimizes coupling; a post-parse fixup in the server branch is likely cleaner than threading sidecar awareness into the config parser.
- The checkpoint path for sidecar resolution: `--load-latest-checkpoint` resolves to a path via `runtime_assets::latest_checkpoint_path`; `--load-checkpoint` uses the user-supplied path. Both conventions omit the `.mpk` extension; the sidecar path is `<path>.metadata.json`.
- `DEFAULT_CHECKPOINT_DIR` in `src/runtime_assets.rs` is the scan root for `--load-latest-checkpoint`.
- Do not change the behavior of `--interactive-generate` or any non-serve path.

## Definition of done

- PR merged.
- The development runbook (`docs/development-runbook.md`) gains one paragraph: "Serving a checkpoint: `--serve --load-latest-checkpoint` now reads the model shape from the checkpoint's `.metadata.json` sidecar automatically. You no longer need to pass `--embed-dim`, `--num-heads`, `--num-layers`, or `--block-size` when serving a checkpoint trained with non-default hyperparameters."
- CLAUDE.md "Gotchas" section for `--serve` is updated to note that shape flags are optional when a sidecar is present.
