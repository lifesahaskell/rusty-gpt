# S2-T4 — `GET /api/health` reporting checkpoint, model shape, uptime

- **Value:** maintainability
- **Size:** S (half day)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** —
- **Blocks:** —

## Context

The compose stack and dev-container setup imply this server will eventually be run as a long-lived process with a watchdog. There is no health endpoint today, so liveness/readiness probes have to fall back to a TCP connect or a `/api/info` call (which is heavier and not designed as a probe). A minimal `/api/health` is one of the lowest-effort, highest-leverage maintainability adds.

## Goal

Add `GET /api/health` that returns 200 with a small JSON body describing what the server is hosting and how long it has been up.

## Acceptance criteria

- New route: `GET /api/health` returns 200 with body:

  ```json
  {
    "status": "ok",
    "uptime_seconds": 1234,
    "model": {
      "kind": "minigpt",
      "embed_dim": 128,
      "num_heads": 4,
      "num_layers": 4,
      "block_size": 128,
      "vocab_size": 2048
    },
    "checkpoint": {
      "loaded": true,
      "source": "latest",
      "basename": "mini_gpt.step-5000.mpk",
      "sha256": "ab12...ef"
    },
    "tokenizer": {
      "kind": "bpe",
      "sha256": "cd34...01"
    }
  }
  ```

- `checkpoint.source` is one of `"none"` (fresh template), `"explicit"` (loaded via `--load-checkpoint`), or `"latest"` (loaded via `--load-latest-checkpoint`).
- The endpoint **never** includes the absolute checkpoint path, only the basename. Information disclosure: a remote attacker should not learn the host's filesystem layout.
- The endpoint is **not** rate-limited (or limited far more permissively than `/api/generate`) so monitoring probes don't get 429'd. Document this exemption explicitly in S2-T1's writeup.
- An integration test boots the server with a fresh-template model and asserts `checkpoint.loaded == false`, `checkpoint.source == "none"`, and the model shape matches the default `Hyperparameters`.
- A second test boots with a known checkpoint and asserts the sha256 matches the on-disk file.
- README "API" section and CLAUDE.md "HTTP server module" gain a sentence about `/api/health`.

## Implementation notes

- Uptime: capture `Instant::now()` at server start in `ServerState` and compute the delta per request — no global state needed.
- The sha256 fields can be cached in `ServerState` at load time; recomputing on every probe is wasteful.
- For the React UI consumer (`mini-gpt-ui/`), this endpoint is **not** part of the user-facing flow — flag it as ops-only in the docstring.
- Resist adding `db_status`, `gpu_status`, etc. that don't exist in this stack. Keep the payload tight; expand only when a real consumer needs more.

## Definition of done

- PR merged.
- A `curl localhost:8787/api/health | jq .` example in the development runbook.
- Sprint 03's UI work (S3-T3) inherits a known reliable health probe.
