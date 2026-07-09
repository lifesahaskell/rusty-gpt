# Experiment card: learning-rate schedule

Roadmap milestone: 3

## Question

Does warmup/decay improve stability or final validation loss under a fixed token budget?

## Runs

Use `--lr-schedule` and `--lr-warmup-steps`; JSON progress logs include `learning_rate` for plotting.


```bash
scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 100 \
  --checkpoint checkpoints/roadmap/lr_constant \
  --artifacts-dir artifacts/roadmap/lr/constant \
  data/input.txt -- --lr-schedule constant

scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 100 \
  --checkpoint checkpoints/roadmap/lr_cosine \
  --artifacts-dir artifacts/roadmap/lr/cosine \
  data/input.txt -- --lr-schedule warmup-cosine --lr-warmup-steps 100

scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 100 \
  --checkpoint checkpoints/roadmap/lr_linear \
  --artifacts-dir artifacts/roadmap/lr/linear \
  data/input.txt -- --lr-schedule warmup-linear --lr-warmup-steps 100
```

## Record

| run | git commit | schedule | warmup | wall-clock | train loss | validation loss/perplexity | tokens/sec | VRAM peak | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant | | | | | | | | | |
| warmup-cosine | | | | | | | | | |
| warmup-linear | | | | | | | | | |

## Plots

- Learning rate over step.
- Train/value loss.
- Tokens/sec.

## Stop when

Schedule behavior is visible in logs and failures can be diagnosed from the plot.
