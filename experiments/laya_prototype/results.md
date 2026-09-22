# Laya/ModernBERT-large Prototype — Results

## Setup deviation from PLAN.md

`convaiinnovations/laya` is not loadable as a plain `AutoModelForSequenceClassification` — it ships
via a custom `laya` pip package wrapping a proprietary option-marker/RLCD decision head, with no
top-level `config.json`. Per the plan's own fallback ("or its declared ModernBERT-large base"), the
experiment uses `answerdotai/ModernBERT-large` directly: architecturally identical to Laya's backbone
(confirmed `hidden_size=1024`, 28 layers), standard AutoModel-loadable, and consistent with the
explicit scope decision not to replicate Laya's RLCD training recipe.

## Hardware feasibility (Step 1)

RTX 2070 (Turing, cc 7.5), `attn_implementation="sdpa"`, fp16 autocast with fp32 master weights.
Model loads and trains without issue. Peak GPU memory during training: **5.36 GB**; during batched
inference: **1.96 GB**. Both comfortably under the 8 GB budget.

One real bug hit and fixed: loading the base model weights themselves in fp16
(`dtype=torch.float16`) alongside `GradScaler` crashes (`Attempting to unscale FP16 gradients`).
Fix: load weights in fp32, let `torch.autocast` cast to fp16 for compute only.

## Accuracy vs. GNN baseline (Steps 3-4)

LoRA (r=16, target modules `Wqkv`/`Wo`, classifier head fully tuned), focal loss (gamma=2.0),
8-epoch budget, early stopping at epoch 5 (best epoch 3), ~74 min total training wall-clock.

| Metric | Laya/ModernBERT-large | GNN baseline (`data/run_metrics.json`) |
|---|---|---|
| Perfect 4-QN match accuracy | **93.72%** | 94.5-96.9% (across bootstrap generations) |
| Test set | 4,824 MARVEL test rows | same split |

Within ~1-3 points of the GNN across the generation range, using no graph structure at all -- rows
are i.i.d. text strings. Per-QN breakdown (test set, n=4824):

| Target | MAE | F1-Macro | F1-Weighted | Precision | Recall |
|---|---|---|---|---|---|
| AFGL_m1 | 0.0087 | 0.9878 | 0.9940 | 0.9947 | 0.9816 |
| AFGL_m2 | 0.0182 | 0.9732 | 0.9867 | 0.9669 | 0.9798 |
| AFGL_m3 | 0.0073 | 0.9920 | 0.9946 | 0.9960 | 0.9882 |
| AFGL_r  | 0.0674 | **0.8117** | 0.9462 | 0.8151 | 0.8085 |
| MEAN | 0.0254 | 0.9412 | 0.9804 | 0.9432 | 0.9395 |

`r` is clearly the weakest QN (macro-F1 0.81 vs. 0.97-0.99 for m1/m2/m3) -- consistent with `r`
being the isotopologue-crossing degeneracy resolver, the hardest of the 4 QNs for any classifier
(graph or tabular) since it depends on distinguishing states that look nearly identical on the other
3 QNs.

## Compute cost (Step 4)

Batched inference throughput: **~150-160 rows/sec** on the RTX 2070 (measured directly, vs. Laya's
own T4 card figures of 103-332 q/s -- same ballpark). At this rate, a full bootstrap-generation pass
over ~2.5M Ca states would take **~4.3-4.6 hours** on this GPU -- in line with the plan's original
risk-#4 estimate (2-7 hours), now grounded in a real measurement rather than the HF card number.

## Failure-mode inspection (Step 5): near-degenerate states

Directly tested risk #1 (ModernBERT's BPE tokenizer doesn't handle floating point well, so
near-degenerate states -- e.g. two real rows differing only at the 5th-6th decimal of energy --
might get confused). Bucketed per-row test accuracy by each row's nearest-neighbor energy gap
within its `(isotope_id, J, polyad)` group:

| Nearest-neighbor energy gap (cm⁻¹) | Accuracy | n |
|---|---|---|
| < 0.001 | 1.000 | 16 |
| 0.001-0.01 | 1.000 | 8 |
| 0.01-0.1 | 0.969 | 32 |
| 0.1-1 | 0.955 | 44 |
| 1-10 | 1.000 | 36 |
| 10-100 | 0.980 | 49 |
| >100 (incl. singletons) | 0.936 | 4,639 |

**Risk #1 is not confirmed.** The tightest-gap rows (the scenario the risk was specifically about)
classify perfectly or near-perfectly, and every tight-gap bucket outperforms the bulk >100 cm⁻¹
bucket. This suggests the model isn't actually relying on parsing exact energy digits to
disambiguate near-degenerate states -- the other serialized features (t1/t2/t3, dom_coeff, symmetry
flags) likely carry enough signal on their own. Caveat: the tight-gap buckets are small (n=8-49),
so this is suggestive, not conclusive -- would want more near-degenerate examples (or a synthetic
stress test) before fully retiring the risk.

## Decision (Step 7)

- **Does Laya/ModernBERT land within ~5-10 points of the GNN?** Yes -- within ~1-3 points, and with
  zero graph-construction effort (rows are i.i.d.).
- **Is the RTX 2070 compute cost tractable for a real run?** Yes -- ~74 min to fine-tune, ~4.5 hrs
  for a full Ca-state inference pass per bootstrap generation, both practical for this project's
  cadence.
- **Does the H2O desk-check (Step 6, see `h2o_feasibility_notes.md`) support the generalization
  motivation?** Partially. H2O turns out to have its own polyad-like vibrational resonance structure
  (confirmed via a quick literature check), so a row classifier genuinely would save the
  graph-construction effort an H2O port would otherwise need -- a real point in its favor. But H2O's
  asymmetric-top rotational structure (Ka, Kc, not a single J) means the combinatorial class
  representation itself needs redesigning from scratch either way, which is architecture-independent
  and probably the larger piece of work. Net: the motivation survives, but weaker than originally
  framed -- H2O porting stays a multi-month effort regardless of GNN vs. classifier.
- **Recommendation:** the row-classifier approach is empirically viable and worth a second look, not
  a dead end. It doesn't yet beat the GNN, and `r` is the weak point for both approaches to some
  degree, but the near-degeneracy risk that motivated the most caution turned out not to bite. It is
  not, however, a shortcut that avoids the hard parts of an eventual H2O port -- treat it as a
  parallel research thread to keep warm, not a reason to reprioritize away from the JQSRT paper work.
- **Not pursued this round:** bootstrap-loop integration, multi-generation retraining, LoRA
  hyperparameter sweep (r=16 was a single untuned choice), or closing the ~1-3 point gap to the GNN.
