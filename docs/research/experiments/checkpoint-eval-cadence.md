# Experiment card: checkpoint/eval cadence

Roadmap milestone: 1

## Question

How much do eval/checkpoint choices perturb throughput and observability?

## Fixed inputs

- Dataset: `data/input.txt`
- Tokenizer: `checkpoints/tokenizer.json`
- Model: `minigpt`
- Backend: CUDA when available
- Budget: baseline + 2 variants within 6 GPU-hours total

## Runs

Use the same `--train-steps` for all runs.

```bash
scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 500 \
  --checkpoint checkpoints/roadmap/cadence_sparse \
  --artifacts-dir artifacts/roadmap/cadence/sparse \
  data/input.txt -- --checkpoint-interval 0

scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 250 \
  --checkpoint checkpoints/roadmap/cadence_medium \
  --artifacts-dir artifacts/roadmap/cadence/medium \
  data/input.txt -- --checkpoint-interval 500 --checkpoint-keep 3

scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 100 \
  --checkpoint checkpoints/roadmap/cadence_frequent \
  --artifacts-dir artifacts/roadmap/cadence/frequent \
  data/input.txt -- --checkpoint-interval 100 --checkpoint-keep 3
```

## Record

| run | git commit | config/command | tokenizer | wall-clock | train loss | validation loss/perplexity | tokens/sec | VRAM peak | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse | | | | | | | | | |
| medium | | | | | | | | | |
| frequent | | | | | | | | | |

## Plots

- Step time or tokens/sec over time.
- Validation-loss visibility by cadence.
- Checkpoint count and total size.

## Stop when

A rerunnable command exists for each run and the table explains whether cadence overhead is acceptable.
