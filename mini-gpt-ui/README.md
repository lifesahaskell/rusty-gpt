# MiniGPT UI

React/Vite client for the `rusty-gpt` HTTP API. The app calls the Rust server under `/api`, shows model metadata from `GET /api/info`, sends generation requests to `POST /api/generate`, and visualizes returned attention weights.

## Development

From the repository root, start the Rust API:

```bash
cargo run --bin rusty-gpt -- --serve --input data/input.txt
```

Then start the UI dev server:

```bash
cd mini-gpt-ui
npm install
npm run dev
```

The Vite dev server proxies API calls through the same `/api` path used by the packaged app.

## API Contract

`GET /api/info` returns the loaded MiniGPT shape:

```json
{
  "vocab_size": 2048,
  "num_layers": 4,
  "num_heads": 4,
  "block_size": 128
}
```

`POST /api/generate` accepts:

```json
{
  "prompt": "ROMEO:",
  "max_tokens": 80,
  "temperature": 1.0,
  "top_k": 40
}
```

`top_k` is optional. The server rejects empty prompts, zero `max_tokens`, non-positive temperatures, and `top_k: 0`; the UI surfaces those validation errors.

`POST /api/train` starts a MiniGPT run and accepts:

```json
{
  "train_steps": 1000,
  "learning_rate": 0.0001,
  "checkpoint_interval": 100,
  "eval_interval": 100,
  "resume_from": "mini_gpt.step-5000"
}
```

`resume_from` is optional. A `202` carries `{"run_id": "..."}`. Only one run is active at a time: a second start answers `409` with the running `run_id`, which the Training panel adopts instead of reporting an error. A server started without a training runner answers `503`, and the panel disables the form.

`GET /api/train/{run_id}/status` reports the run. `status` is one of `running`, `completed`, `interrupted`, or `failed` — a stop lands on `interrupted`, there is no separate stopped state. `training_loss`, `value_loss`, `steps_per_second`, and `eta_seconds` are `null` until the first progress event, and `error` is present only on `failed`. The route is exempt from the generate rate limiter, so the UI polls it once a second while a run is active and stops as soon as it leaves `running`. The server reports only the latest loss, so the UI keeps the sample history itself to draw the loss curve.

`DELETE /api/train/{run_id}` requests a stop and answers `202`; the run reaches `interrupted` at its next step boundary, so the UI keeps polling afterwards. Repeat deletes while stopping answer `202` again. A `404` means the id is not the running run — usually because it just finished — and the UI refreshes the status rather than showing an error.

## Tests

Run the UI test suite from this directory:

```bash
npm run test:all
```

Run the full API/UI smoke from the repository root:

```bash
./scripts/run_e2e_tests.sh
```
