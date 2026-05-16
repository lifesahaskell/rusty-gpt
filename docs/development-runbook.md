# Development Runbook

Operational recipes for working on `rusty-gpt`. The narrative overview lives in
[../README.md](../README.md); this file is the command reference.

## Contents

- [Prerequisites](#prerequisites)
- [Container-based development (recommended)](#container-based-development-recommended)
- [Native development](#native-development)
- [Training](#training)
- [Serving the API](#serving-the-api)
- [Tooling binaries](#tooling-binaries)
- [Tests](#tests)
- [Release packaging](#release-packaging)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Use case                  | Requirements                                                       |
| ------------------------- | ------------------------------------------------------------------ |
| Native dev (any platform) | Rust 1.93+, ~10 GB free disk for cargo target                      |
| Container dev (CPU)       | Docker 24+, Docker Compose v2.24+ (for `!override` / `!reset`)     |
| Container dev (CUDA)      | All of the above + NVIDIA driver + `nvidia-container-toolkit`      |
| VS Code Dev Containers    | VS Code + the "Dev Containers" extension                           |

Verify the CUDA prerequisite:

```bash
bash scripts/test_cuda_passthrough.sh
```

If it fails at step 2 or 3, install the toolkit (one-time, root):

```bash
sudo bash scripts/install_nvidia_container_toolkit.sh
```

On WSL2 the script installs the toolkit **inside** the WSL distro — the NVIDIA
driver itself stays on Windows.

---

## Container-based development (recommended)

### First-time setup

```bash
# Brings up server (cargo watch, port 8787) + ui (vite HMR, port 5173).
# First build is ~5-10 min (rust toolchain + cargo-watch + apt extras).
docker compose up -d

# Tail logs while cargo compiles for the first time.
docker compose logs -f server
```

Visit:

- UI: http://localhost:5173
- API: http://localhost:8787/api/info

On every restart, `scripts/start_dev_server.sh` reads the newest
`checkpoints/*.metadata.json` and matches `--block-size`, `--embed-dim`,
`--num-heads`, `--num-layers`, and `RUSTY_GPT_BPE_TOKENIZER` to that
checkpoint — so `--load-latest-checkpoint` never crashes on shape mismatch.

### VS Code Dev Containers

`Cmd/Ctrl+Shift+P` → **Dev Containers: Reopen in Container** → pick:

- `rusty-gpt` — CPU dev environment (~1.5 GB image), attaches to the `server` service
- `rusty-gpt (cuda)` — CUDA-equipped, attaches to the `trainer` service with GPU passthrough

Open a terminal inside the container (or `docker compose exec server bash`)
and run cargo commands directly — see [Daily iteration](#daily-iteration).

### Daily iteration

Inside the dev container (or via `docker compose exec server bash`):

```bash
cargo test
cargo clippy --all-targets
cargo fmt --all
cargo run --bin rusty-gpt -- --serve --load-latest-checkpoint
```

Edits to `src/` trigger `cargo watch` to rebuild and relaunch the server.
The UI hot-reloads automatically via Vite.

### Cleanup

```bash
# Stop containers; keep named volumes (preserves cargo cache).
docker compose down

# Nuclear: also wipe named volumes (cargo target, registry, ui node_modules).
docker compose down --volumes
```

### Compose profile cheat sheet

| Profile     | Services                | When to use                                     |
| ----------- | ----------------------- | ----------------------------------------------- |
| (default)   | `server`, `ui`          | Day-to-day dev                                  |
| `bootstrap` | `train-tokenizer`       | One-shot: train a BPE tokenizer                 |
| `train`     | `trainer`               | One-shot: train MiniGPT on CUDA                 |
| `cuda-dev`  | (cuda devcontainer)     | Auto-activated by `.devcontainer/cuda/`         |

### Skipping compose for prod-image testing

```bash
# Run with only the production images (no dev override).
docker compose -f compose.yaml up
```

---

## Native development

```bash
# Build / lint / format
cargo build
cargo build --release
cargo clippy --all-targets
cargo fmt --all -- --check
cargo check --features cuda

# Tests (107 unit + 1 integration)
cargo test
cargo test --test default_runtime
cargo test multi_head_attention_returns_model_dim_for_each_token_position

# Quick demo (CPU, trivial model)
cargo run

# Quick smoke run
RUSTY_GPT_TRAIN_STEPS=1 cargo run -- --input tests/fixtures/input.txt
```

---

## Training

### Train a BPE tokenizer

```bash
cargo run --bin train-tokenizer -- \
    --corpus data/input.txt \
    --vocab-size 2048 \
    --output checkpoints/tokenizer.json

# From a Hugging Face dataset
cargo run --bin train-tokenizer -- \
    --corpus 'hf://Salesforce/wikitext?config=wikitext-2-raw-v1&split=train&column=text&rows=1000' \
    --vocab-size 2048 \
    --output checkpoints/wikitext-tokenizer.json
```

Required flags: `--corpus` (path or `hf://` URI), `--vocab-size` (≥ 256),
`--output`.

### Train MiniGPT

```bash
# CPU
cargo run --release --bin rusty-gpt -- --model minigpt

# CUDA (requires --features cuda)
cargo run --release --features cuda --bin rusty-gpt -- --backend cuda --model minigpt

# Via helper script (defaults to release + JSON logs for CUDA)
./scripts/run_training.sh --backend cuda \
    --checkpoint checkpoints/mini_gpt \
    data/input.txt

# From Hugging Face
./scripts/run_training.sh --backend cuda \
    'hf://Salesforce/wikitext?config=wikitext-2-raw-v1&split=train&column=text&rows=1000'

# Train tokenizer + model in one go
./scripts/run_training.sh --train-tokenizer \
    --tokenizer checkpoints/wikitext-tokenizer.json \
    --vocab-size 2048 \
    'hf://Salesforce/wikitext?config=wikitext-2-raw-v1&split=train&column=text&rows=1000'
```

### Train inside the container

```bash
# CUDA training as a one-shot compose run (does not start ui or server)
docker compose --profile train run --rm trainer

# CPU training (no profile needed; explicit override of the compose command)
docker compose run --rm --entrypoint bash server -c \
    'cargo run --release --bin rusty-gpt -- --model minigpt'
```

### Compare all four model variants

```bash
cargo run --bin rusty-gpt -- --model compare
```

`compare` is a pseudo-variant; it expands to the four real models and trains
each in sequence.

---

## Serving the API

```bash
# Start the HTTP API on 127.0.0.1:8787/api (fresh untrained MiniGPT)
cargo run --bin rusty-gpt -- --serve --input data/input.txt

# Auto-load the newest checkpoint in checkpoints/
cargo run --bin rusty-gpt -- --serve --input data/input.txt --load-latest-checkpoint

# Load a specific checkpoint
cargo run --bin rusty-gpt -- --serve --input data/input.txt \
    --checkpoint checkpoints/mini_gpt --load-checkpoint

# CUDA backend
cargo run --features cuda --bin rusty-gpt -- --serve --backend cuda --input data/input.txt

# Interactive chat against a saved checkpoint (CPU + MiniGPT only)
cargo run --bin rusty-gpt -- --model minigpt --interactive-generate \
    --checkpoint checkpoints/mini_gpt
```

### React UI (Vite)

```bash
# Start the API + Vite dev server locally
./scripts/run_local.sh

# Same, with CUDA API backend
RUSTY_GPT_BACKEND=cuda ./scripts/run_local.sh
```

The container path runs both automatically (`docker compose up`).

### API endpoints

| Method | Path             | Notes                                                       |
| ------ | ---------------- | ----------------------------------------------------------- |
| GET    | `/api/info`      | Model shape and tokenizer info                              |
| POST   | `/api/generate`  | `{prompt, max_tokens, temperature, top_k?}`; `temperature > 0` required |

Curl example:

```bash
curl -X POST http://127.0.0.1:8787/api/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Once upon","max_tokens":80,"temperature":0.8,"top_k":40}'
```

---

## Tooling binaries

### Source corpus collection

```bash
# Concatenate source files from a repo into data/<repo-name>.txt
cargo run --bin collect-source -- --repo /path/to/repo

# Or specify the output filename (parent dir is always data/)
cargo run --bin collect-source -- --repo /path/to/repo --output repo-source.txt
```

Includes common source extensions (`rs`, `toml`, `ts`, `tsx`, `js`, `py`,
`go`, `java`, `c`, `cpp`, `sh`, `html`, `css`, `json`, `yaml`, `sql`, `md`)
and skips `.git`, `target`, `node_modules`, `dist`, `build`, `.next`,
`coverage`.

---

## Tests

### Code-level

```bash
cargo test                                   # 107 unit + 1 integration
cargo test --test default_runtime            # CPU-default smoke test
cargo test multi_head_attention_returns_model_dim_for_each_token_position
```

### Dev container regression tests

These were added to catch real bugs we hit while building the container
stack. They use isolated compose project names so they don't conflict with
your running dev stack.

| Script                                       | Catches                                                           |
| -------------------------------------------- | ----------------------------------------------------------------- |
| `scripts/test_cuda_passthrough.sh`           | Missing `nvidia-container-toolkit` or broken compose GPU reservation |
| `scripts/test_devcontainer_ui.sh`            | UI restart-loop from empty named-volume mount                     |
| `scripts/test_devcontainer_server.sh`        | Server restart-loop from PATH stripping or cargo bin disambiguation |
| `scripts/test_devcontainer_generate.sh`     | 502 on `/api/generate` from checkpoint-shape metadata mismatch    |

Each is safe to run in any order; they tear down their own state on exit.
The generate test uses a stable project name (`rusty-gpt-gentest`) so the
cargo cache persists across runs (~30 s warm, ~3 min cold).

### End-to-end test

```bash
# Starts the Rust API and Vite UI, sends a generation request through the UI server
./scripts/run_e2e_tests.sh
```

---

## Release packaging

```bash
# Build a release-candidate tarball (CPU)
./scripts/build_release_candidate.sh

# Stable identifier for repeatable package names
RC_ID=rc1 ./scripts/build_release_candidate.sh

# CUDA-capable artifact
RUSTY_GPT_BACKEND=cuda ./scripts/build_release_candidate.sh
```

Artifacts land in `target/release-candidates/`. See
[release-and-evaluation.md](release-and-evaluation.md) for CPU/CUDA artifact
expectations, packaged API startup, smoke checks, and repeatable
training/benchmark capture.

---

## Troubleshooting

### `/api/generate` returns 502 (Bad Gateway)

The Vite proxy returns 502 when the upstream rusty-gpt server isn't
reachable — almost always because the server crashed at startup.

```bash
docker compose logs --tail=80 server | grep -E '(Error|panicked|Caused)'
```

Most common cause: a `.mpk` checkpoint exists in `checkpoints/` whose
metadata sidecar declares a different model shape or tokenizer than the
running server. `scripts/start_dev_server.sh` is supposed to auto-match
these, but it falls back to defaults if the `*.metadata.json` is missing or
unreadable. Either:

- Delete the stale `.mpk` and let the server start with a fresh model, or
- Regenerate the metadata sidecar by re-training (or saving) the checkpoint, or
- Pass explicit `--block-size`, `--embed-dim`, `--num-heads`, `--num-layers`
  to the server invocation that match the checkpoint.

### `cudarc` symbol error in `/api/generate`

```
undefined symbol: cuCoredumpDeregisterCompleteCallback
```

The loaded `libcuda.so.1` is older than the CUDA driver API expected by
Burn/CubeCL. On WSL this means the Windows NVIDIA driver needs to be
updated. Verify with `nvidia-smi` from the same shell before retrying
`--backend cuda`.

### CUDA training fails with `could not select device driver`

```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

`nvidia-container-toolkit` is not installed or the docker daemon hasn't
been restarted since installing it. Run:

```bash
bash scripts/test_cuda_passthrough.sh                    # diagnose
sudo bash scripts/install_nvidia_container_toolkit.sh    # install
```

### UI container restart-loops with `vite: not found`

The `ui-node-modules` named volume is empty but the install guard isn't
running `npm ci`. Should be fixed by the `[ -x node_modules/.bin/vite ]`
check in `compose.override.yaml`; if it regresses:

```bash
docker compose down --volumes        # wipe the empty volume
docker compose up -d ui              # recreate; npm ci should run
bash scripts/test_devcontainer_ui.sh # verify
```

### Server restart-loops with `cargo: command not found`

A login shell (`bash -lc`) reset PATH via `/etc/profile` and dropped
`/usr/local/cargo/bin`. Compose commands must use `bash -c`, not `bash
-lc`. `scripts/test_devcontainer_server.sh` catches this regression.

### `cargo run` errors with `could not determine which binary to run`

The crate has three binaries (`rusty-gpt`, `train-tokenizer`,
`collect-source`) and no `default-run`. Always specify `--bin rusty-gpt`
when invoking the main binary, or set `default-run = "rusty-gpt"` in
`Cargo.toml`.

### Stale cargo cache after switching CPU ↔ CUDA dev container

The two dev containers use separate target volumes (`cargo-target-cpu`,
`cargo-target-cuda`) because feature flags produce incompatible artifacts.
A switch always recompiles the affected half of the dep graph — this is
expected, not a bug. The registry cache is shared so the fetch is fast.
