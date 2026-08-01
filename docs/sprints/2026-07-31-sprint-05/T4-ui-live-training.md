# S5-T4 — Replace the dead `TrainingDashboard.tsx` stub with a live training panel

- **Value:** product
- **Size:** L (2–3 days)
- **Suggested agent:** fullstack-react-rust-engineer
- **Depends on:** T1, T2, T3
- **Blocks:** none

## Context

`mini-gpt-ui/src/components/TrainingDashboard.tsx` currently: accepts drag-and-dropped files into local `useState`, shows a status string like "3 files ready for training," and calls no API whatsoever. There is no corpus-upload endpoint and adding one is out of scope (see SPRINT.md) — the file-drop UI doesn't correspond to anything the backend can do. Gut it; don't extend it.

## Goal

A Training tab that starts a run against T1, polls T2, renders a live loss curve + step counter, and stops the run via T3.

## Acceptance criteria

- Start form: `train_steps`, `learning_rate`, `checkpoint_interval`, and (if a checkpoint exists) a `from_checkpoint` selector — matches T1's payload, not a redesign of it.
- On submit: `POST /api/train`, store the returned `run_id`, immediately start polling `GET /api/train/{run_id}/status` (a sensible interval — 1s is fine, this endpoint is rate-limiter-exempt per T2).
- Live view while `status == "running"`: step counter (`step / total_steps`), a loss curve (train loss and value loss both plotted — reuse whatever charting approach `AttentionHeatmap.tsx` / `ExpertRoutingHeatmap.tsx` already established in this codebase rather than pulling in a new charting dependency), and an ETA readout from `eta_seconds`.
- Stop button calls `DELETE /api/train/{run_id}`; UI reflects `status == "stopped"` once the poll confirms it (don't optimistically flip state before the server confirms — the run finishes its current step first).
- Terminal states (`completed`, `stopped`, `interrupted`, `failed`) stop polling and show a clear final summary instead of a stale "running" view.
- `409 Conflict` from T1 (a run is already active) surfaces as a clear inline error, not a silent failure — same validation-error pattern already used for `/api/generate` per `docs/project-refinement-phase.md`'s "user-visible validation errors" goal.
- `api.ts` gets typed client functions for the three new routes, following the existing pattern for `generate`/`info`/`health`.
- `npm run test:all` in `mini-gpt-ui/` covers: start → poll → completed, and start → stop → stopped.

## Implementation notes

- Delete the file-drop state, `toTrainingFile`, `formatBytes`, and the drag handlers entirely — none of it has a backend to serve it. Keep the component name and its tab slot in `App.tsx` (`activeWorkspace === 'training'`).
- Match the visual language of `AttentionHeatmap.tsx` and `ExpertRoutingHeatmap.tsx` rather than inventing a new one — this UI already has an established look for the other two tabs.

## Definition of done

- PR merged. Sprint exit criteria's "no dead file-drop UI left behind" is satisfied.
