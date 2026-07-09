# Experiment card: tokenizer vocabulary size

Roadmap milestone: 4

## Question

How does BPE vocab size trade sequence length, output-head size, speed, and loss?

## Runs

Each run retrains its tokenizer from the same corpus. Do not compare raw loss as a pure model-quality result; tokenizer shape is the variable.

```bash
scripts/run_training.sh --backend cuda --cargo-profile release --train-tokenizer \
  --vocab-size 1024 --train-steps 1000 --eval-interval 250 \
  --tokenizer checkpoints/roadmap/tokenizers/bpe_1024.json \
  --checkpoint checkpoints/roadmap/vocab_1024 \
  --artifacts-dir artifacts/roadmap/vocab/1024 \
  data/input.txt

scripts/run_training.sh --backend cuda --cargo-profile release --train-tokenizer \
  --vocab-size 2048 --train-steps 1000 --eval-interval 250 \
  --tokenizer checkpoints/roadmap/tokenizers/bpe_2048.json \
  --checkpoint checkpoints/roadmap/vocab_2048 \
  --artifacts-dir artifacts/roadmap/vocab/2048 \
  data/input.txt

scripts/run_training.sh --backend cuda --cargo-profile release --train-tokenizer \
  --vocab-size 4096 --train-steps 1000 --eval-interval 250 \
  --tokenizer checkpoints/roadmap/tokenizers/bpe_4096.json \
  --checkpoint checkpoints/roadmap/vocab_4096 \
  --artifacts-dir artifacts/roadmap/vocab/4096 \
  data/input.txt
```

## Record

| run | git commit | vocab size | tokenizer sha256 | wall-clock | train loss | validation loss/perplexity | tokens/sec | VRAM peak | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | | 1024 | | | | | | | |
| baseline | | 2048 | | | | | | | |
| large | | 4096 | | | | | | | |

## Plots

- Tokens per byte/word.
- Loss/perplexity.
- Tokens/sec.
- Note output-head size or parameter count if cheap to collect.

## Stop when

The writeup separates tokenizer mechanics from model-quality claims.
