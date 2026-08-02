# Configuration Reference

Every CLI flag and `RUSTY_GPT_*` environment variable accepted by the
`rusty-gpt` binary. Both forms work; the CLI wins when both are set.

## Runtime flags

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--backend cpu\|cuda` | `RUSTY_GPT_BACKEND` | `cpu` | `cuda` is only available when the crate is built with `--features cuda` (requires the CUDA toolkit). |
| `--input <path>` | `RUSTY_GPT_INPUT` | `data/input.txt` | Plain UTF-8 text. Also accepts `hf://` URIs (see [Hugging Face datasets](#hugging-face-datasets)). |
| `--model <name>` | `RUSTY_GPT_MODEL` | `minigpt` | `trivial`, `single-attention`, `multi-attention`, `minigpt` (alias `mini-gpt`), `moe-gpt` (alias `moegpt`), `compare`. |
| `--checkpoint <path>` | `RUSTY_GPT_MINIGPT_CHECKPOINT` | `checkpoints/mini_gpt` | Path without `.mpk` — Burn appends it. Explicit paths must resolve inside `checkpoints/`; bare names resolve there. |
| `--resume-from <path>` | `RUSTY_GPT_RESUME_FROM` | unset | Resume training from a saved checkpoint. Path without `.mpk`, confined to `checkpoints/` on the same rules as `--checkpoint`. Requires `--model minigpt` or `--model moe-gpt`; any other model is rejected at parse time. |
| `--log-format plain\|json` | `RUSTY_GPT_LOG_FORMAT` | backend default | CPU defaults to plain text; CUDA defaults to JSON Lines. |
| — | `RUSTY_GPT_BPE_TOKENIZER` | `checkpoints/tokenizer.json` | BPE tokenizer JSON used by MiniGPT, MoeGPT, and `compare` runs. |
| `--interactive-generate` | — | off | Requires `--backend cpu` and `--model minigpt` or `--model moe-gpt`. |
| `--serve` | — | off | Starts the HTTP API under `/api`; supports `cpu` and compiled-in `cuda` backends. |
| `--load-checkpoint` | — | off | With `--serve`, loads MiniGPT or MoeGPT API weights from `--checkpoint`. The checkpoint must match the model kind, shape, and tokenizer vocabulary. |
| `--load-latest-checkpoint` | — | off | With `--serve`, loads the newest `.mpk` file in `checkpoints/`. |
| `--server-addr <host:port>` | `RUSTY_GPT_SERVER_ADDR` | `127.0.0.1:8787` | Address used by `--serve`. |
| `--max-prompt-bytes <n>` | `RUSTY_GPT_MAX_PROMPT_BYTES` | `8192` | `POST /api/generate` prompt byte cap. Must be > 0. |
| `--max-output-tokens <n>` | `RUSTY_GPT_MAX_OUTPUT_TOKENS` | `512` | `POST /api/generate` `max_tokens` cap. Must be > 0. |
| `--rate-limit-rps <n>` | `RUSTY_GPT_RATE_LIMIT_RPS` | `5` | In-process generate request refill rate. `0` disables rate limiting. |
| `--rate-limit-burst <n>` | `RUSTY_GPT_RATE_LIMIT_BURST` | `10` | In-process generate request burst. Must be > 0 unless rate limiting is disabled. |
| `--max-train-steps <n>` | `RUSTY_GPT_MAX_TRAIN_STEPS` | `100000` | `POST /api/train` `train_steps` cap. Must be > 0. |
| `--max-train-learning-rate <n>` | `RUSTY_GPT_MAX_TRAIN_LEARNING_RATE` | `1.0` | `POST /api/train` `learning_rate` cap. Must be a finite number > 0. |

## Model shape

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--block-size <n>` | `RUSTY_GPT_BLOCK_SIZE` | `128` | Context length. Must be > 0. |
| `--batch-size <n>` | `RUSTY_GPT_BATCH_SIZE` | `32` | Training batch size. Must be > 0. |
| `--embed-dim <n>` | `RUSTY_GPT_EMBED_DIM` | `128` | Model width. Must be divisible by `num_heads`. |
| `--num-heads <n>` | `RUSTY_GPT_NUM_HEADS` | `4` | Attention heads. Must be > 0. |
| `--num-layers <n>` | `RUSTY_GPT_NUM_LAYERS` | `4` | Transformer block count. Must be > 0. |
| `--dropout <p>` | `RUSTY_GPT_DROPOUT` | `0.0` | Reserved model dropout setting. Must be `>= 0` and `< 1`. |
| `--moe-experts <n>` | `RUSTY_GPT_MOE_EXPERTS` | `4` | Number of MoE experts per block for `--model moe-gpt`. Must be > 0. |
| `--moe-top-k <n>` | `RUSTY_GPT_MOE_TOP_K` | `2` | Experts selected per token for `--model moe-gpt`. Must be >= 1 and <= `moe_experts`. |
| `--moe-aux-loss-weight <n>` | `RUSTY_GPT_MOE_AUX_LOSS_WEIGHT` | `0.01` | Weight applied to the load-balancing auxiliary loss during MoeGPT training. Must be >= 0. |

## Training

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--learning-rate <n>` | `RUSTY_GPT_LEARNING_RATE` | `1e-4` | Base optimizer learning rate. |
| `--lr-schedule constant\|warmup-cosine\|warmup-linear` | `RUSTY_GPT_LR_SCHEDULE` | `constant` | Learning-rate schedule used by training experiments. |
| `--lr-warmup-steps <n>` | `RUSTY_GPT_LR_WARMUP_STEPS` | `0` | Linear warmup length; must be `<= train_steps`. |
| `--sampling-policy random-window\|sequential\|shuffled-chunks` | `RUSTY_GPT_SAMPLING_POLICY` | `random-window` | Batch-window sampling strategy. |
| `--train-steps <n>` | `RUSTY_GPT_TRAIN_STEPS` | `1000` | Must be > 0. |
| `--eval-interval <n>` | `RUSTY_GPT_EVAL_INTERVAL` | `100` | `0` ⇒ log only the final step. |
| `--prefetch-batches <n>` | `RUSTY_GPT_PREFETCH_BATCHES` | `2` | Number of prepared CPU batches queued ahead of training. |
| `--generate-tokens <n>` | `RUSTY_GPT_GENERATE_TOKENS` | `80` | Interactive generation token count. |
| `--grad-clip-norm <n>` | `RUSTY_GPT_MINIGPT_GRAD_CLIP_NORM` | `1.0` | Must be > 0. |
| `--checkpoint-interval <n>` | `RUSTY_GPT_CHECKPOINT_INTERVAL` | `0` | Save `<checkpoint>.step-<N>.mpk` every N steps of MiniGPT or MoeGPT training. `0` disables periodic saves. |
| `--checkpoint-keep <k>` | `RUSTY_GPT_CHECKPOINT_KEEP` | `3` | Retention window for periodic snapshots — older `.step-N.` files are pruned. The final end-of-run save and any `.interrupted-step-*` save are never pruned. Must be > 0 when `checkpoint_interval` is non-zero. |

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

### `POST /api/train`

Starts a background MiniGPT training run and returns immediately with a run
ID. Only enabled when the server is serving MiniGPT (`--model minigpt`);
serving MoeGPT wires the route but every request answers `503`
`training_unavailable`, since only MiniGPT has a training path today.

SECURITY: this route has no authentication. Anyone who can reach it can spend
the box's whole GPU/CPU budget and overwrite the serving checkpoint. Bind to
localhost (the `--server-addr` default) until auth lands — see the Sprint 05
parking lot.

Request body:

```json
{
  "train_steps": 1000,
  "learning_rate": 1e-4,
  "checkpoint_interval": 100,
  "eval_interval": 100,
  "resume_from": "mini_gpt.step-5000"
}
```

`resume_from` is optional and named to match `--resume-from` /
`RUSTY_GPT_RESUME_FROM`: a checkpoint path without `.mpk`, confined to
`checkpoints/` under the same rules as `--checkpoint`. All other fields are
required and override the server's base hyperparameters for this run only —
model shape (`--embed-dim`, `--num-heads`, etc.) is fixed at server startup.

On success, returns `202 Accepted` with `{"run_id": "<uuid>"}` in well under
100ms; training itself runs on a blocking task, not on the response path.

Only one run is active at a time, process-wide:

- A second `POST /api/train` while a run is active returns `409`
  `{"error":"run_in_progress","run_id":"<active run>"}` with `Retry-After: 30`.
- `POST /api/generate` returns `503`
  `{"error":"training_in_progress","retry_after_seconds":30}` with
  `Retry-After: 30` for the same reason — the model is mid-training and
  serving a stale snapshot instead was judged not worth the extra resident
  copy.

Request validation happens before a run is admitted:

- `train_steps` must be `> 0` and `<= --max-train-steps`, else `400`
  `{"error":"train_steps_out_of_range","max_allowed":N,"requested":M}`.
- `learning_rate` must be finite, `> 0`, and `<= --max-train-learning-rate`,
  else `400`
  `{"error":"learning_rate_out_of_range","max_allowed":N,"requested":M}`.

The request body is capped at 4096 bytes (route-local, independent of
`--max-prompt-bytes`).

Each run writes `checkpoints/runs/run-<uuid>.json` immediately on admission
(so a crashed server's runs are auditable) and again on every progress
update:

```json
{
  "run_id": "...",
  "status": "running",
  "request": { "train_steps": 1000, "...": "..." },
  "started_at_unix": 1750000000,
  "ended_at_unix": null,
  "steps_completed": 340,
  "total_steps": 1000,
  "training_loss": 2.14,
  "value_loss": 2.31,
  "checkpoints": ["mini_gpt.step-100.mpk"],
  "error": null
}
```

`status` is one of `running | completed | interrupted | failed`.
`checkpoints` lists basenames only, in the order they were written — this API
never discloses absolute paths, same boundary `/api/health` enforces.
`error` is present only when `status` is `failed`. There is no status endpoint
yet (tracked as S5-T2); read the manifest file directly until it lands.

Starting the first training run installs the same SIGINT/SIGTERM handler the
CLI training path uses — a deliberate, one-way exception to "never install
signal handlers on the serve path." From that point on, Ctrl-C during serving
stops training gracefully (partial checkpoint saved, manifest marked
`interrupted`) instead of killing the process; a second Ctrl-C within 2s still
force-exits. A completed run's weights replace the served model in place —
`/api/generate`, `/api/info`, and `/api/health` see the new weights on their
next request, and the model is never swapped mid-training.

### `DELETE /api/train/{run_id}`

Stops the active training run. Takes no request body.

```bash
curl -i -sS -X DELETE http://127.0.0.1:8787/api/train/<run_id>
```

The stop is graceful, not a kill: it sets the same interrupt flag a SIGINT
does, so the run finishes the step it is on, saves
`<checkpoint>.interrupted-step-<N>.mpk` plus its metadata sidecar
(`interrupted: true`), and lands on `status: "interrupted"` in the manifest.
There is no separate `stopped` status — a programmatic stop and a signal are
indistinguishable to the training loop by design. That partial checkpoint is
never pruned by `--checkpoint-keep`, and a stopped run's weights are **not**
swapped into the served model; resume from the partial checkpoint with
`resume_from` on the next `POST /api/train`.

- `202 Accepted`, empty body, when the run was accepted for stopping. The
  response means "the stop was requested", not "the run has stopped" — the run
  reaches `interrupted` at its next step boundary. Poll the manifest to
  observe the transition.
- `404 Not Found` when `run_id` is not the currently running run: unknown ID,
  an earlier run's ID, or the active run's ID after it already finished,
  failed, or stopped.
- Repeating the `DELETE` while the run is still stopping returns `202` again,
  so a client never has to track whether its first request landed.

SECURITY: unauthenticated, like `POST /api/train`. Matching the `run_id` is
not authorization — it only stops a stale client from killing a run it never
started.

### Checkpoint metadata sidecar

MiniGPT and MoeGPT saves write `<checkpoint>.metadata.json` next to the Burn
`.mpk` weights. The sidecar records model kind/shape, MoE expert shape when
present, tokenizer path/hash, input source, training hyperparameters, final
value loss, final perplexity, optional MoE aux loss, and git commit when
available.

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

MiniGPT, MoeGPT, and `compare` runs load the BPE tokenizer from
`checkpoints/tokenizer.json` unless `RUSTY_GPT_BPE_TOKENIZER` points
elsewhere. Train or replace that file before training or loading a MiniGPT or
MoeGPT checkpoint if the corpus vocabulary changes — checkpoint tensor shapes
must match the tokenizer vocabulary size.
