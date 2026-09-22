# Laya / ModernBERT-large Classifier Prototype — Report

**Date:** 2026-09-21
**Status:** Investigation complete (all 7 planned steps). No further work planned unless prioritized.

## Motivation

The current CO2 quantum-number assignment pipeline (`GNN_CO2_QN_assignment/`) is a GraphSAGE GNN
operating on a hand-engineered graph (inter-isotope polyad chains + intra-isotope rotational
ladders), feeding a Hungarian-solver assignment stage. Porting this pipeline to other molecules
(H2O first) means re-engineering that graph for each new molecule's physics. This investigation
tested whether a row-based text classifier — `convaiinnovations/laya`, a ModernBERT-large-backed
model, or its base architecture directly — could match the GNN's accuracy while treating each
state as an independent (i.i.d.) row, with no graph construction at all. If viable, this would
decouple future molecule ports from graph-engineering effort.

Full build plan, scoping decisions, and risk analysis: `PLAN.md`.

## What was built

1. `prepare_data.py` — serializes the 48,234 MARVEL CO2 states to plain text rows (energy, J,
   parity, polyad, dominant coefficient, t1/t2/t3, symmetry, isotope masses), reusing the pipeline's
   existing 80/10/10 train/val/test split so results are directly comparable to the GNN baseline.
2. `train_laya.py` — LoRA fine-tune (r=16, targeting `Wqkv`/`Wo`) of `answerdotai/ModernBERT-large`
   with a 220-way classification head, focal loss (gamma=2.0, matching `src/model.py`), fp16
   mixed precision on the RTX 2070 (Turing GPU — no bf16 tensor cores, `sdpa` attention required).
3. `evaluate.py` — scores the fine-tuned model on the held-out test set using the exact same metric
   definitions as `src/metrics.py`, so the numbers are apples-to-apples with `data/run_metrics.json`.

One deviation from plan: `convaiinnovations/laya` itself isn't loadable via a standard
`AutoModelForSequenceClassification` call — it ships through a custom pip package wrapping a
proprietary decision head, with no top-level `config.json`. Per the plan's own documented fallback,
`answerdotai/ModernBERT-large` was used directly instead (confirmed architecturally identical to
Laya's backbone). This doesn't compromise the test, since the objective was always to evaluate the
ModernBERT-class row-classifier approach, not Laya's own proprietary RLCD training recipe.

## Results

| Metric | Row classifier (this prototype) | GNN baseline |
|---|---|---|
| Perfect 4-QN match accuracy | **93.72%** | 94.5–96.9% (across bootstrap generations) |
| Weakest individual QN | `r` (F1-macro 0.81) | `r` is also typically the hardest QN |
| GPU memory (train / inference) | 5.36 GB / 1.96 GB | — |
| Training wall-clock | ~74 min | — |
| Inference throughput | ~150–160 rows/sec | — |

The row classifier lands within **~1–3 points** of the GNN, with zero graph structure. A specific
technical risk flagged during scoping — that ModernBERT's tokenizer would mishandle the near-
degenerate energies that differentiate some states (e.g. two real rows differing only at the 5th–6th
decimal, `0.713887` vs `0.713966` cm⁻¹) — was directly tested and did **not** materialize: rows with
the tightest energy gaps classified as well as or better than the rest of the test set.

Full numbers and the failure-mode breakdown: `results.md`.

## H2O generalization desk-check

A code-free check (one grounding literature search) into whether H2O's earlier stalled port
(`qn_predictions_h2o/`, no longer on disk) was blocked by graph-engineering effort or by
molecule-specific physics found a more nuanced answer than assumed going in: H2O turns out to have
its own polyad-like vibrational resonance structure (so a row classifier would genuinely save some
graph-engineering effort, similar to what this prototype demonstrates for CO2). But H2O's
asymmetric-top rotational structure (needing `Ka`, `Kc` alongside `J`, not CO2's single rotational
number) means the combinatorial class representation itself would need a from-scratch redesign
regardless of which classifier architecture is used underneath.

**Net effect on the original motivation:** it survives, but is weaker than initially framed. Full
reasoning and sources: `h2o_feasibility_notes.md`.

## Recommendation

Keep this prototype as a parallel research thread, not a reason to reprioritize current effort. It
is empirically viable (within striking distance of the GNN, tractable compute cost on the RTX 2070)
and the main technical risk didn't materialize, but it is not a shortcut past the hardest part of an
eventual H2O port — the QN representation redesign — so it shouldn't pull focus from the JQSRT paper
work currently in progress.

**Explicitly not attempted this round:** bootstrap-loop integration, multi-generation retraining,
LoRA hyperparameter tuning, or closing the remaining ~1–3 point gap to the GNN.
