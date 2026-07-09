# Experiment card: batch sampling strategy

Roadmap milestone: 6

## Question

How do random windows compare to sequential/epoch-style windows for stability, locality, and leakage risk?

## Runs

Use `--sampling-policy`; the implementation stays centralized in `src/loader/data.rs`.


```bash
scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 100 \
  --checkpoint checkpoints/roadmap/sampling_random \
  --artifacts-dir artifacts/roadmap/sampling/random \
  data/input.txt -- --sampling-policy random-window

scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 100 \
  --checkpoint checkpoints/roadmap/sampling_sequential \
  --artifacts-dir artifacts/roadmap/sampling/sequential \
  data/input.txt -- --sampling-policy sequential

scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 100 \
  --checkpoint checkpoints/roadmap/sampling_shuffled_chunks \
  --artifacts-dir artifacts/roadmap/sampling/shuffled-chunks \
  data/input.txt -- --sampling-policy shuffled-chunks
```

## Record

| run | git commit | sampling policy | wall-clock | train loss | validation loss/perplexity | tokens/sec | VRAM peak | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random-window | | | | | | | | |
| sequential | | | | | | | | |
| shuffled-chunks | | | | | | | | |

## Plots

- Loss smoothness.
- Tokens/sec.
- Coverage or duplicate-window summary if cheap.

## Stop when

The notebook identifies one sampling policy to keep for future experiments and explains leakage risks.
