# Research roadmap automations

Start with boring shell + existing logs. Add heavier automation only after two experiments hurt.

## Implemented now

- `scripts/run_training.sh -- ...` passes extra args through to `rusty-gpt`, so experiment cards can use new runtime flags without changing the wrapper each time.
- `--artifacts-dir` already writes `manifest.txt` and `training.log` per run.
- Experiment cards under `docs/research/experiments/` give copy-pasteable 3-run commands and result tables.

## Recommended next automations

1. **JSONL summary extractor**
   - Read each `artifacts/**/training.log`.
   - Emit one CSV row per run: final train loss, validation loss/perplexity, tokens/sec, wall-clock.
   - Add only after the first cadence experiment produces real logs.

2. **VRAM sampler wrapper**
   - While a run is active, sample `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` once per second.
   - Write `vram.csv` in the run artifact dir.
   - Skip on CPU or when `nvidia-smi` is missing.

3. **Experiment matrix runner**
   - Tiny script that runs a checked-in TSV: `run_name<TAB>artifacts_dir<TAB>extra_args`.
   - Stop there; no YAML framework until shell quoting becomes the actual bottleneck.

4. **Notebook/table seed file**
   - Generate `summary.md` beside each experiment with the required columns prefilled from manifests.
   - Keep plots manual until repeated plotting becomes annoying.

5. **Config files for long commands**
   - Only add config-file support after a command exceeds safe copy/paste length or needs to be shared between scripts and UI.

## Not recommended yet

- Workflow engines, databases, dashboards, or experiment-tracking services.
- Automatic statistical reruns.
- Any automation that hides the exact command used for a run.
