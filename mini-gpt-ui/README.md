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

## Tests

Run the UI test suite from this directory:

```bash
npm run test:all
```

Run the full API/UI smoke from the repository root:

```bash
./scripts/run_e2e_tests.sh
```
