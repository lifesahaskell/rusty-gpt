# S3-T3 — Expose live training progress in `mini-gpt-ui/`

- **Value:** product
- **Size:** L (2–3 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** T1, T2
- **Blocks:** —

## Context

`mini-gpt-ui/` is described in CLAUDE.md as a separate React frontend that consumes the `/api` routes — a black-box from the backend's perspective. With T1 and T2 in place, the UI now has the data it needs to surface a real training experience.

This is the user-visible payoff of the whole sprint — and arguably of S1 and S2 too. Treat it as the keystone task.

## Goal

Add a "Training" view to the React UI that lets a user start, monitor, and stop a MiniGPT training run, with a live-updating loss curve and step counter.

## Acceptance criteria

- New page / route in `mini-gpt-ui/` (the exact path follows the project's existing routing convention — coordinate with whoever owns the UI).
- A form lets the user configure: `train_steps`, `learning_rate`, `checkpoint_interval`, `eval_interval`, and optionally `from_checkpoint` (a dropdown of available checkpoints fetched via a small new `GET /api/checkpoints` endpoint, or hardcoded text input if that endpoint is descoped).
- A "Start Training" button POSTs to `/api/train`, captures the returned `run_id`, and switches into the live-progress view.
- The live-progress view polls `/api/train/{run_id}/status` every 1–2 seconds and renders:
  - A line chart of training loss vs. step (and value loss vs. step) — use whatever charting library the UI already uses (Recharts, Chart.js, etc.); do **not** add a new dependency for this if one exists.
  - A step counter (`step / total_steps`).
  - Current loss, value loss, perplexity numeric readouts.
  - ETA (if non-null), formatted as `"~5 min remaining"` not a raw timestamp.
  - A "Stop Training" button that calls `DELETE /api/train/{run_id}` and confirms cancellation.
- Terminal states (`completed`, `failed`, `interrupted`, `cancelled`) are clearly displayed with the final loss and the list of saved checkpoints; the user can navigate to the generation view and load one of them.
- All API error responses (429 from the rate limiter, 409 if a training run is already active, 503 from `/api/generate` during training) are surfaced as user-readable messages — no raw JSON dumps in the UI.
- Existing UI flows (generation, attention visualization, `/api/info`) continue to work.
- `npm run test:all` (or whatever the UI's full test command is) passes.
- The UI README gains a paragraph on the new training surface.

## Implementation notes

- Polling is fine for S3 — don't add WebSockets or SSE unless the polling-induced jitter is visibly bad. That's a S4 enhancement.
- The chart should be capped at the last N points (e.g. 500) for very long runs, with a "fit all" toggle. Don't render 100k points; the browser will hate it.
- Cancellation UX: show a confirmation dialog. Users will hit the wrong button.
- Coordinate with whoever owns `mini-gpt-ui/` early in the sprint — they may have constraints (a redesign in flight, a styling guide, etc.) that materially change the implementation.

## Definition of done

- PR merged in `mini-gpt-ui/`.
- A short demo (screenshot or GIF) in the sprint review showing a training run kicked off from the UI.
- The backend `/api/train` and `/api/train/{run_id}/status` endpoints have a real consumer driving them — confirms the contracts work end-to-end.
