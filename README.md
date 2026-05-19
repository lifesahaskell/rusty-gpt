# rusty-gpt

A GPT trained from scratch in Rust on top of [Burn](https://burn.dev).

The crate stacks four progressively richer models — embedding → single-head
attention → multi-head attention → full transformer (MiniGPT) — so the
codebase reads as a teaching arc. It ships an Axum HTTP API and a React UI
for poking at attention matrices interactively.

## Quick start

```bash
# Container dev stack: API at :8787, UI at :5173, hot reload on both.
docker compose up

# Or natively (Rust 1.93+):
cargo run --bin rusty-gpt -- --serve --load-latest-checkpoint
cargo test

# Train a MiniGPT from scratch on the bundled Shakespeare corpus.
# One-time: build the BPE tokenizer (writes checkpoints/tokenizer.json).
cargo run --bin train-tokenizer -- --corpus data/input.txt \
    --vocab-size 2048 --output checkpoints/tokenizer.json

# Then train MiniGPT (defaults to 1000 steps on CPU; add --release for speed).
cargo run --release
```

Open VS Code with the *Dev Containers* extension to pick between a CPU dev
environment and a CUDA one ([details](docs/development-runbook.md#container-based-development-recommended)).

## Highlights

- **Four models, one file** — [src/model/mod.rs](src/model/mod.rs) builds up
  from a trivial embedding head to a full pre-norm transformer block, each
  variant runnable on the same training loop.
- **Two backends from one binary** — CPU (`ndarray`) by default, CUDA opt-in
  via the `cuda` Cargo feature; no recompile to swap.
- **Self-describing checkpoints** — every `.mpk` ships with a
  `.metadata.json` sidecar (shape, tokenizer hash, training config, git
  commit), validated on load.
- **Graceful interrupt** — Ctrl-C (or `SIGTERM`) during a MiniGPT
  training run completes the current step, writes a partial checkpoint to
  `<checkpoint>.interrupted-step-<N>.mpk` (with `interrupted: true` in the
  sidecar), and exits with code 130. A second interrupt within 2s aborts
  immediately without saving.
- **Periodic mid-run checkpoints** — `--checkpoint-interval <N>` saves
  `<checkpoint>.step-<N>.mpk` every N steps; `--checkpoint-keep <K>`
  (default 3) prunes older periodic snapshots. The final end-of-run save
  and any interrupted save are never pruned. Example:
  `cargo run --release --features cuda -- --backend cuda --model minigpt --train-steps 100000 --checkpoint-interval 1000`.
- **Attention visualization** — `POST /api/generate` returns per-layer /
  per-head attention weights alongside the generated tokens for the React
  UI to render.
- **Hugging Face datasets** — `--input hf://owner/dataset?…` pulls text
  directly via the datasets-server rows API with caching + retry/backoff.
- **Dev stack in one command** — `docker compose up` brings up the API +
  UI with `cargo watch` and Vite HMR, and auto-matches server config to the
  newest checkpoint's metadata.

## API

```bash
curl -X POST http://127.0.0.1:8787/api/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Once upon","max_tokens":80,"temperature":0.8,"top_k":40}'
```

Response includes the generated text, per-token splits, and `attention[]`
matrices keyed by `layer` and `head`. `GET /api/info` returns the loaded
model's shape and tokenizer info. `GET /api/health` returns server uptime,
model shape, and the checkpoint + tokenizer sha256 — designed for liveness
probes and never exposes absolute filesystem paths.

## Architecture

```
src/
  tokenizer/    char + BPE tokenizers (Tokenizer trait, runtime dispatch)
  loader/       random-window batch sampler producing (x, y) shifted by 1
  model/        the four models + persistence (.mpk + .metadata.json)
  server/       Axum router for /api/generate, /api/info, /api/health
  bin/          train-tokenizer, collect-source helper binaries

mini-gpt-ui/    React + Vite UI (proxies /api/* to the server)
docker/         CPU + CUDA Dockerfiles (multi-stage; dev + runtime targets)
scripts/        compose-stack regression tests + native helper scripts
```

## Documentation

- **[Development runbook](docs/development-runbook.md)** — every command for
  setup, training, serving, testing, troubleshooting.
- **[Configuration reference](docs/configuration.md)** — full CLI flag and
  env var table.
- **[Release & evaluation](docs/release-and-evaluation.md)** — packaging
  release candidates and capturing benchmark artifacts.
- **[Project refinement](docs/project-refinement-phase.md)** — engineering
  backlog for runtime hardening.

## License

Apache-2.0 — see [LICENSE](LICENSE).
