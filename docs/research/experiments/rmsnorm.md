# Experiment card: RMSNorm

Roadmap milestone: 5

## Question

Can RMS-only normalization match LayerNorm behavior with different cost/stability tradeoffs?

## Runs

Not runnable until RMSNorm support exists. Keep the first implementation minimal: a runtime flag selecting norm kind for MiniGPT blocks. Only add final-norm variants if block-level RMSNorm is stable.

Target shape:

```bash
scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 100 \
  --checkpoint checkpoints/roadmap/norm_layernorm \
  --artifacts-dir artifacts/roadmap/norm/layernorm \
  data/input.txt -- --norm layernorm

scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 100 \
  --checkpoint checkpoints/roadmap/norm_rmsnorm \
  --artifacts-dir artifacts/roadmap/norm/rmsnorm \
  data/input.txt -- --norm rmsnorm

scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 100 \
  --checkpoint checkpoints/roadmap/norm_rmsnorm_final \
  --artifacts-dir artifacts/roadmap/norm/rmsnorm-final \
  data/input.txt -- --norm rmsnorm --final-norm rmsnorm
```

## Record

| run | git commit | norm | wall-clock | train loss | validation loss/perplexity | tokens/sec | VRAM peak | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| layernorm | | | | | | | | |
| rmsnorm | | | | | | | | |
| rmsnorm-final | | | | | | | | |

## Plots

- Loss.
- Tokens/sec.
- Gradient norm if already exposed.

## Stop when

The writeup explains residual scale control and failure modes.
