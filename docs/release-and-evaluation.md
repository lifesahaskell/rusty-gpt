# Release and Evaluation Workflow

This runbook covers the small repeatable workflows used to validate `rusty-gpt`
release candidates and capture lightweight training/evaluation artifacts.

## Release Candidate Artifacts

Build release candidates from the repo root:

```bash
./scripts/build_release_candidate.sh
```

The script writes a tarball under `target/release-candidates/`. Each package
contains:

- `bin/rusty-gpt` - the release API/training binary.
- `ui/` - the built React static bundle.
- `scripts/run_api.sh` - a package-local API launcher.
- `manifest.txt` - version, git commit, backend, and smoke-check status.
- `smoke-check.log` - output from a one-step CPU smoke check when enabled.

Use `RC_ID` when you need a stable package name:

```bash
RC_ID=rc1 ./scripts/build_release_candidate.sh
```

### CPU and CUDA Expectations

CPU is the default artifact type:

```bash
RUSTY_GPT_BACKEND=cpu ./scripts/build_release_candidate.sh
```

CPU artifacts are built without the `cuda` Cargo feature and should run with
`--backend cpu`. They are the expected default for local development and CPU-only
CI.

CUDA artifacts opt in to the Cargo feature at build time:

```bash
RUSTY_GPT_BACKEND=cuda ./scripts/build_release_candidate.sh
```

CUDA artifacts can still run CPU commands, including the package smoke check.
Actual `--backend cuda` validation must happen on a host with a compatible
NVIDIA driver and CUDA runtime visible to the process.

Disable the build-time smoke check only when packaging in an environment where
running the binary is not possible:

```bash
RELEASE_SMOKE_CHECK=0 ./scripts/build_release_candidate.sh
```

## Packaged API Startup

After unpacking an artifact:

```bash
./scripts/run_api.sh --backend cpu --input /path/to/input.txt --server-addr 127.0.0.1:8787
curl http://127.0.0.1:8787/api/info
```

For CUDA artifacts on a GPU host:

```bash
./scripts/run_api.sh --backend cuda --input /path/to/input.txt --server-addr 127.0.0.1:8787
curl http://127.0.0.1:8787/api/info
```

Serve `ui/` with a static file server and proxy `/api` to the Rust API server.

## Lightweight Evaluation Artifacts

`scripts/run_training.sh` can run MiniGPT training plus generation benchmarks and
save a manifest/log bundle for comparison across runs:

```bash
RUSTY_GPT_TRAIN_STEPS=100 \
RUSTY_GPT_EVAL_INTERVAL=20 \
./scripts/run_training.sh \
  --backend cpu \
  --log-format json \
  --benchmark \
  --benchmark-prompt-lens 10,50 \
  --benchmark-gen-lens 25,50 \
  --benchmark-warmups 1 \
  --benchmark-iterations 3 \
  --artifacts-dir target/evaluations/cpu-smoke \
  data/input.txt
```

The artifact directory contains:

- `manifest.txt` - input, model, tokenizer, checkpoint, backend, profile, and benchmark arguments.
- `training.log` - combined tokenizer/training/evaluation output, including JSON events when `--log-format json` is used.

The training log includes value loss and value perplexity from the existing
training loop. With `--benchmark`, it also includes `benchmark_result` events
for naive versus cached generation timings.

Use the same command shape for a Hugging Face dataset slice:

```bash
RUSTY_GPT_TRAIN_STEPS=100 \
./scripts/run_training.sh \
  --backend cpu \
  --log-format json \
  --benchmark \
  --artifacts-dir target/evaluations/wikitext-cpu \
  'hf://Salesforce/wikitext?config=wikitext-2-raw-v1&split=train&column=text&rows=1000'
```

For CUDA comparisons, keep the benchmark dimensions and training step count the
same, changing only the backend and artifact directory:

```bash
RUSTY_GPT_BACKEND=cuda \
RUSTY_GPT_TRAIN_STEPS=100 \
./scripts/run_training.sh \
  --benchmark \
  --artifacts-dir target/evaluations/wikitext-cuda \
  'hf://Salesforce/wikitext?config=wikitext-2-raw-v1&split=train&column=text&rows=1000'
```

MiniGPT still requires an explicit BPE tokenizer. Use `--train-tokenizer` only
when intentionally creating or replacing the tokenizer for that run:

```bash
./scripts/run_training.sh \
  --train-tokenizer \
  --tokenizer checkpoints/wikitext-tokenizer.json \
  --vocab-size 2048 \
  --artifacts-dir target/evaluations/tokenizer-refresh \
  'hf://Salesforce/wikitext?config=wikitext-2-raw-v1&split=train&column=text&rows=1000'
```
