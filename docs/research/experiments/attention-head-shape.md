# Experiment card: attention-head shape

Roadmap milestone: 2

## Question

What changes when the same `embed_dim` is split into fewer/wider vs more/narrower heads?

## Fixed inputs

- Dataset: `data/input.txt`
- Tokenizer: `checkpoints/tokenizer.json`
- Model: `minigpt`
- Backend: CUDA when available
- Keep `embed_dim` fixed so only `num_heads/head_dim` changes.

## Runs

```bash
scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 250 \
  --checkpoint checkpoints/roadmap/heads_baseline \
  --artifacts-dir artifacts/roadmap/heads/baseline \
  data/input.txt -- --embed-dim 128 --num-heads 4

scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 250 \
  --checkpoint checkpoints/roadmap/heads_fewer \
  --artifacts-dir artifacts/roadmap/heads/fewer \
  data/input.txt -- --embed-dim 128 --num-heads 2

scripts/run_training.sh --backend cuda --cargo-profile release \
  --train-steps 1000 --eval-interval 250 \
  --checkpoint checkpoints/roadmap/heads_more \
  --artifacts-dir artifacts/roadmap/heads/more \
  data/input.txt -- --embed-dim 128 --num-heads 8
```

## Record

| run | git commit | heads | head dim | wall-clock | train loss | validation loss/perplexity | tokens/sec | VRAM peak | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | | 4 | 32 | | | | | | |
| fewer | | 2 | 64 | | | | | | |
| more | | 8 | 16 | | | | | | |

## Plots

- Train/value loss.
- Tokens/sec.
- Optional attention heatmaps from `/api/generate` after loading each checkpoint.

## Stop when

The writeup explains `head_dim`, shape constraints, and any throughput/loss movement.
