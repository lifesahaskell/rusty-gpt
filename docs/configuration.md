# Configuration Reference

Every CLI flag and `RUSTY_GPT_*` environment variable accepted by the
`rusty-gpt` binary. Both forms work; the CLI wins when both are set.

## Runtime flags

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--backend cpu\|cuda` | `RUSTY_GPT_BACKEND` | `cpu` | `cuda` is only available when the crate is built with `--features cuda` (requires the CUDA toolkit). |
| `--input <path>` | `RUSTY_GPT_INPUT` | `data/input.txt` | Plain UTF-8 text. Also accepts `hf://` URIs (see [Hugging Face datasets](#hugging-face-datasets)). |
| `--model <name>` | `RUSTY_GPT_MODEL` | `minigpt` | `trivial`, `single-attention`, `multi-attention`, `minigpt` (alias `mini-gpt`), `compare`. |
| `--checkpoint <path>` | `RUSTY_GPT_MINIGPT_CHECKPOINT` | `checkpoints/mini_gpt` | Path without `.mpk` — Burn appends it. |
| `--log-format plain\|json` | `RUSTY_GPT_LOG_FORMAT` | backend default | CPU defaults to plain text; CUDA defaults to JSON Lines. |
| — | `RUSTY_GPT_BPE_TOKENIZER` | `checkpoints/tokenizer.json` | BPE tokenizer JSON used by MiniGPT and `compare` runs. |
| `--interactive-generate` | — | off | Requires `--backend cpu` and `--model minigpt`. |
| `--serve` | — | off | Starts the HTTP API under `/api`; supports `cpu` and compiled-in `cuda` backends. |
| `--load-checkpoint` | — | off | With `--serve`, loads MiniGPT API weights from `--checkpoint`. The checkpoint must match the model shape and tokenizer vocabulary from `--input`. |
| `--load-latest-checkpoint` | — | off | With `--serve`, loads the newest `.mpk` file in `checkpoints/`. |
| `--server-addr <host:port>` | `RUSTY_GPT_SERVER_ADDR` | `127.0.0.1:8787` | Address used by `--serve`. |
| `--max-prompt-bytes <n>` | `RUSTY_GPT_MAX_PROMPT_BYTES` | `8192` | `POST /api/generate` prompt byte cap. Must be > 0. |
| `--max-output-tokens <n>` | `RUSTY_GPT_MAX_OUTPUT_TOKENS` | `512` | `POST /api/generate` `max_tokens` cap. Must be > 0. |
| `--rate-limit-rps <n>` | `RUSTY_GPT_RATE_LIMIT_RPS` | `5` | In-process generate request refill rate. `0` disables rate limiting. |
| `--rate-limit-burst <n>` | `RUSTY_GPT_RATE_LIMIT_BURST` | `10` | In-process generate request burst. Must be > 0 unless rate limiting is disabled. |

## Model shape

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--block-size <n>` | `RUSTY_GPT_BLOCK_SIZE` | `128` | Context length. Must be > 0. |
| `--batch-size <n>` | `RUSTY_GPT_BATCH_SIZE` | `32` | Training batch size. Must be > 0. |
| `--embed-dim <n>` | `RUSTY_GPT_EMBED_DIM` | `128` | Model width. Must be divisible by `num_heads`. |
| `--num-heads <n>` | `RUSTY_GPT_NUM_HEADS` | `4` | Attention heads. Must be > 0. |
| `--num-layers <n>` | `RUSTY_GPT_NUM_LAYERS` | `4` | Transformer block count. Must be > 0. |
| `--dropout <p>` | `RUSTY_GPT_DROPOUT` | `0.0` | Reserved model dropout setting. Must be `>= 0` and `< 1`. |

## Training

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--learning-rate <n>` | `RUSTY_GPT_LEARNING_RATE` | `1e-4` | Optimizer learning rate. |
| `--train-steps <n>` | `RUSTY_GPT_TRAIN_STEPS` | `1000` | Must be > 0. |
| `--eval-interval <n>` | `RUSTY_GPT_EVAL_INTERVAL` | `100` | `0` ⇒ log only the final step. |
| `--prefetch-batches <n>` | `RUSTY_GPT_PREFETCH_BATCHES` | `2` | Number of prepared CPU batches queued ahead of training. |
| `--generate-tokens <n>` | `RUSTY_GPT_GENERATE_TOKENS` | `80` | Interactive generation token count. |
| `--grad-clip-norm <n>` | `RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM` | `1.0` | Must be > 0. |

## Hugging Face dataset loader

| Env var | Default | Notes |
|---|---|---|
| `RUSTY_GPT_HF_DATASET_CACHE` | `data/huggingface-cache` | Directory for cached `hf://` dataset text slices. |
| `RUSTY_GPT_HF_REQUEST_DELAY_MS` | `250` | Delay between Hugging Face rows API page requests. Increase if you get 429s. |
| `RUSTY_GPT_HF_MAX_RETRIES` | `6` | Retry count for Hugging Face rows API 429/5xx responses. |
| `RUSTY_GPT_HF_RETRY_BASE_DELAY_MS` | `1000` | Base exponential-backoff delay when a retryable response has no `Retry-After` header. |

## Benchmark flags

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--benchmark-prompt-lens <list>` | `RUSTY_GPT_BENCHMARK_PROMPT_LENS` | `10,50,100` | Prompt lengths for `--benchmark-generation`. |
| `--benchmark-gen-lens <list>` | `RUSTY_GPT_BENCHMARK_GEN_LENS` | `50,100,200` | Generated-token lengths for `--benchmark-generation`. |
| `--benchmark-warmups <n>` | `RUSTY_GPT_BENCHMARK_WARMUPS` | `1` | Warmup iterations per case. |
| `--benchmark-iterations <n>` | `RUSTY_GPT_BENCHMARK_ITERATIONS` | `5` | Measured iterations per case. |

## Notes

### Hugging Face datasets

`--input` accepts URIs in the form
`hf://<dataset-id>?config=<config>&split=<split>&column=<column>&rows=<n>&offset=<n>`.
The loader uses the Hugging Face datasets-server rows API, concatenates the
selected column with newlines, and defaults to `config=default`, `split=train`,
`column=text`, `rows=1000`, `offset=0`.

Resolved dataset text is cached under `RUSTY_GPT_HF_DATASET_CACHE` before
training; subsequent runs with the same dataset/config/split/column/offset/rows
read the local copy before attempting a download. Individual rows-API page
responses are cached too, so a rate-limited run resumes from already-fetched
pages. For large pulls, prefer increasing `RUSTY_GPT_HF_REQUEST_DELAY_MS`
instead of retrying failed downloads.

### `POST /api/generate`

Accepts optional `top_k` in addition to `prompt`, `max_tokens`, and
`temperature`. API requests require `temperature > 0`; internal greedy
generation remains available for benchmarks and tests via zero-temperature
generation options.

The generate route validates `prompt` and `max_tokens` before tokenizer/model
work. A prompt whose UTF-8 byte length exceeds `--max-prompt-bytes` returns
HTTP 400 with `{"error":"prompt_too_large","max_bytes":N,"actual_bytes":M}`.
`max_tokens == 0` or a value above `--max-output-tokens` returns HTTP 400 with
`{"error":"max_tokens_out_of_range","max_allowed":N,"requested":M}`. The
route also has a body-size cap of `max_prompt_bytes + 4096` bytes.

Validated generate requests use an in-process token bucket configured by
`--rate-limit-rps` and `--rate-limit-burst`. Exceeding the bucket returns HTTP
429 with `Retry-After` and `{"error":"rate_limited","retry_after_seconds":N}`.
Invalid cap requests do not consume rate-limit tokens. The limiter is
per-process; if the API is scaled to N replicas, the effective limit is
approximately N times the configured values. `GET /api/info` and
`GET /api/health` are exempt.

To reproduce a 429 with defaults overridden:

```bash
cargo run --bin rusty-gpt -- --serve --rate-limit-rps 1 --rate-limit-burst 1

for i in 1 2 3; do
  curl -i -sS -X POST http://127.0.0.1:8787/api/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Once","max_tokens":1,"temperature":1.0}'
done
```

### Checkpoint metadata sidecar

MiniGPT saves write `<checkpoint>.metadata.json` next to the Burn `.mpk`
weights. The sidecar records model shape, tokenizer path/hash, input source,
training hyperparameters, final value loss, final perplexity, and git commit
when available.

- Legacy `.mpk` files without a sidecar still load unchecked.
- Checkpoints with a sidecar fail fast if the requested model shape is
  incompatible with the running config.
- The compose dev stack reads the newest sidecar at server startup and
  auto-matches its shape flags + tokenizer — see
  [scripts/start_dev_server.sh](../scripts/start_dev_server.sh).

### Compile-time vs runtime

The constants `BLOCK_SIZE`, `BATCH_SIZE`, `EMBED_DIM`, `NUM_HEADS`,
`NUM_LAYERS`, `DROPOUT`, `LEARNING_RATE` at the top of `src/main.rs` are the
hard-coded *defaults* for the `Hyperparameters` struct. Every entry above
overrides them at runtime via a CLI flag or `RUSTY_GPT_*` env var (CLI wins).

### Tokenizer compatibility

MiniGPT and `compare` runs load the BPE tokenizer from
`checkpoints/tokenizer.json` unless `RUSTY_GPT_BPE_TOKENIZER` points
elsewhere. Train or replace that file before training or loading a MiniGPT
checkpoint if the corpus vocabulary changes — checkpoint tensor shapes must
match the tokenizer vocabulary size.
