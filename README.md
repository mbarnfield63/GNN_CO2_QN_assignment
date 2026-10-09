# Machine learning quantum number assignment for water

Student project branch. Start with the brief: [`docs/handoff.pdf`](docs/handoff.pdf).

This branch is a clean slate. The CO2 work lives on the other branches of this
repository (`main`: GNN pipeline; `CDSD-Analysis`: Laya prototype in
`experiments/laya_prototype/`). Only the two pieces worth reusing are here.

## Setup

```bash
uv sync
uv run src/assignment.py          # self-check, prints "assignment self-check passed"
```

Training needs an NVIDIA GPU with at least 8 GB. Otherwise use a free T4 on
[Google Colab](https://colab.research.google.com) (Runtime > Change runtime type > T4 GPU).

## What is here

| File | Purpose |
|---|---|
| `src/assignment.py` | Hungarian assignment within blocks of mutually exclusive labels, with locked (known) labels and the post-solver `assigned_margin` confidence |
| `src/train_classifier.py` | LoRA fine-tune of ModernBERT-large (Laya's backbone) on `data/{train,val}.jsonl` rows of `{"text": ..., "label": ...}`; `predict` writes logits for `assignment.py` |
| `docs/handoff.tex` | Project brief |

## Typical loop

```bash
# 1. your code: build data/train.jsonl, data/val.jsonl, data/test.jsonl
uv run src/train_classifier.py train
uv run src/train_classifier.py predict data/test.jsonl test_logits.npy
# 2. your code: load logits, call assignment.assign(blocks, logits, locked), threshold on margin
```

`data/`, `checkpoints/` and `*.npy` are gitignored.
