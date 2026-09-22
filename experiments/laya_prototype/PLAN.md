# Phase 3, Step 0 — Laya Classifier Investigation: Build Plan

**Status:** Resumed 2026-09-21. `nvidia-smi` and `torch.cuda.is_available()` now both work in this
session (previously blocked — see git history of this file for the earlier parked note). Step 1
(feasibility spike) **passed**: `torch.cuda.get_device_capability(0)` reports `(7, 5)` as expected
for the RTX 2070 (Turing). One deviation from the original plan: `convaiinnovations/laya` is **not**
loadable via plain `AutoModelForSequenceClassification.from_pretrained(...)` — it ships through a
custom `laya` pip package (`laya.load(...)`) wrapping a proprietary option-marker/RLCD decision
head, not a standard config.json + weights layout (confirmed: no top-level `config.json`; encoder
config lives at `encoder/config.json`, `model.safetensors` bundles encoder + custom head together).
Per the plan's own fallback ("or its declared ModernBERT-large base") and the explicit scope
decision to not replicate Laya's RLCD training recipe, the spike (and the fine-tune to follow) uses
`answerdotai/ModernBERT-large` directly instead — architecturally identical (confirmed matching
`hidden_size=1024`, 28 layers against `encoder/config.json`) and standard AutoModel-loadable, so
this doesn't compromise the objective (testing a ModernBERT-class row classifier vs. the GNN).
Loaded with `attn_implementation="sdpa"`, fp16, `num_labels=221`; forward+backward pass on a dummy
batch succeeded, peak GPU memory 1.61 GB (well under the 8GB budget). Proceeding to step 2.

**Update, same session:** Steps 1-7 all completed. `class_mapping.csv` has 220 classes (not 221) so
`num_labels=220` was used in practice. One training bug hit and fixed: loading base model weights in
fp16 alongside `GradScaler` crashes (`Attempting to unscale FP16 gradients`) -- fixed by loading fp32
weights and letting `autocast` handle fp16 compute only. Headline result: **93.72% perfect 4-QN
match accuracy** on the held-out MARVEL test set vs. the GNN's 94.5-96.9%, with zero graph structure
(rows are i.i.d. text). The near-degeneracy risk (#1) did not materialize -- tight energy-gap rows
classified as well or better than the bulk. Full results in `results.md`; H2O desk-check in
`h2o_feasibility_notes.md`. Not pursued: bootstrap-loop integration, multi-generation retraining,
LoRA hyperparameter tuning. Recommendation: keep as a parallel research thread, not a reprioritization
away from the JQSRT paper work (see `results.md` for the full reasoning).

## Objective

Empirically test whether `convaiinnovations/laya` (HF: https://huggingface.co/convaiinnovations/laya —
ModernBERT-large-backed, calibrated text classifier, 421M params) can serve as a viable
alternative to the GraphSAGE GNN classification stage in the CO2 QN assignment pipeline
(`GNN_CO2_QN_assignment/src/model.py`), and whether doing so would meaningfully ease future
extension to other molecules (H2O first).

This is a step-0 investigation: fast, empirical, time-boxed to "until the first result tells us
something," not a production integration.

## Scope

**In scope:**
- LoRA fine-tune Laya's ModernBERT-large backbone on text-serialized MARVEL CO2 states.
- Compare 4-QN accuracy against the GNN baseline (94.5–96.9% across generations, per
  `data/run_metrics.json`) on the *same* held-out test split (`test_mask` column, already in
  `data/unified_co2_graph_data.csv`).
- Keep the Hungarian solver stage unchanged — Laya's per-class probabilities feed the same
  `1 - prob` cost-matrix assignment `train.py: evaluate_physical_assignment` already does for the GNN.
- A short, code-free desk-check of H2O's actual QN scheme, to test whether "graph vs. tabular
  classifier" is even the thing that matters for H2O generalization.

**Out of scope (do not build this round):**
- Full pipeline replacement or bootstrap-loop integration.
- Any H2O data ingestion or training.
- Laya's own RLCD/typed-decision training recipe — use the pretrained encoder as a
  feature extractor with a plain classification head instead; RLCD is Laya's own training
  method for *its* task (calibrated text Q&A), not something we need to replicate.
- Encoding graph structure (neighbor states, chains) into the text prompt — rows are i.i.d.,
  intentionally, per grilling discussion. A weak result here is itself informative (may indicate
  we've stepped back into the old pre-graph cancellation-trap failure mode the project moved
  away from — see `CLAUDE.md` evolutionary lineage).

## Known risks going in (from grilling, carry forward)

1. **Numeric precision**: ModernBERT's BPE tokenizer doesn't handle floating point well.
   Near-degenerate states differ only in the 5th–6th decimal of energy (e.g. two real rows:
   `0.713887` vs `0.713966` cm⁻¹ are different states). Plan is to serialize plain text and see
   what breaks — do not pre-solve this by binning/rounding, that would cripple the comparison
   before it starts.
2. **Hardware**: RTX 2070 is Turing (compute capability 7.5) — **no bf16 tensor cores, and
   ModernBERT's default attention path assumes flash-attention which typically wants Ampere+.**
   Must explicitly set `attn_implementation="sdpa"` or `"eager"` when loading, and use fp16 (not
   bf16) mixed precision. Verify this in the feasibility spike before writing any training code.
3. **Class count / architecture fit**: ~220 output classes (`data/class_mapping.csv`). Laya's
   "typed-decision" head is designed for small option sets — we are *not* using that head, just
   the pretrained encoder + a standard `num_labels=221` classification head via
   `AutoModelForSequenceClassification`, LoRA-adapted.
4. **Compute for a real (non-prototype) run**: ~2.5M Ca states need inference per bootstrap
   generation. At Laya's stated throughput (103–332 q/s/T4), that's ~2–7 hours per generation on
   a T4-class GPU — worth recording actual RTX 2070 throughput in step 5 below to ground this.
5. **H2O motivation is unverified**: it's genuinely unknown whether the stalled earlier H2O
   attempt (repo no longer on disk) failed on graph-engineering effort or on molecule-specific
   physics/QN-scheme differences. The desk-check (step 6) exists specifically to de-risk this.

## Build steps

### 1. Environment & feasibility spike (~30–60 min)
- Create an **isolated** uv project at `experiments/laya_prototype/` (own `pyproject.toml`) —
  do not add `transformers`/`peft` to the main pipeline's `pyproject.toml`; its `torch`/PyG pins
  are already fiddly per `CLAUDE.md`'s HPC notes, and this is a throwaway experiment.
  - `uv init experiments/laya_prototype && cd experiments/laya_prototype`
  - `uv add torch transformers peft accelerate scikit-learn pandas`
- Confirm CUDA visibility and device capability:
  ```python
  import torch
  print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))  # expect (7, 5)
  ```
- Load `convaiinnovations/laya` (or its declared ModernBERT-large base) with
  `attn_implementation="sdpa"`, run one dummy forward pass, confirm it fits in 8GB and doesn't
  silently fall back to CPU or crash on flash-attention import.
- **Go/no-go**: if the model won't load or fit at all on this GPU, stop here and report back —
  don't proceed to data prep.

### 2. Data prep — serialize MARVEL rows to text (~1–2 hrs)
- Source: `GNN_CO2_QN_assignment/data/unified_co2_graph_data.csv`, filter `is_marvel == True`
  (~48,234 rows). Reuse the existing `train_mask` / `val_mask` / `test_mask` columns directly —
  this is what makes the final comparison to `run_metrics.json` fair.
- Write `prepare_data.py`: one row → one plain-text string, from columns `energy`, `J`,
  `parity_encoded`, `isotope_id`, `polyad`, `dom_coeff`, `t1`, `t2`, `t3`, `is_symmetric`,
  `tot_sym_A1/A2/B1/B2`, masses. E.g.:
  `"isotope 626, J=14, parity=0, polyad=3, energy=1234.567891, dom_coeff=0.87, t1=2, t2=1, t3=0"`
- Label: `combinatorial_class_id` (0–220), used as a plain multiclass target — not routed through
  Laya's typed-decision machinery.
- Output: `data/{train,val,test}.jsonl` under the experiment dir.

### 3. LoRA fine-tune script (~2–3 hrs)
- `train_laya.py`:
  - Load base model + tokenizer, wrap in `peft.LoraConfig` targeting attention
    query/value projections (confirm exact module names for ModernBERT once loaded —
    `print(model)` first).
  - fp16 mixed precision (Turing — no bf16). Batch size start 4–8 with gradient accumulation to
    an effective batch ~32; sequence length is short (rows are single lines) so this should be
    cheap relative to Laya's stated context windows.
  - Loss: cross-entropy over 221 classes. Consider matching `FocalLoss(gamma=2.0)` from
    `src/model.py` for a fairer comparison given the same class imbalance exists here.
  - Fixed small epoch budget (5–10) with early stopping on val accuracy.

### 4. Evaluation & comparison (~1 hr)
- `evaluate.py`: run on `test_mask` rows. Compute exact 4-QN accuracy and per-QN MAE
  (m1, m2, m3, r) using the **same definitions** `src/metrics.py` uses for the GNN, so the numbers
  are directly comparable to `data/run_metrics.json`'s 94.5–96.9%.
- Record training wall-clock, peak GPU memory, and inference throughput (rows/sec) on the RTX
  2070 — needed to sanity-check risk #4 above against a real hardware number instead of the HF
  card's T4 figures.
- Write `results.md`: a simple GNN-vs-Laya table plus caveats.

### 5. Failure-mode inspection (~30–45 min)
- Specifically check accuracy on near-degenerate rows (same isotope+J+polyad group, close
  energies) — does the model actually separate them, or does it systematically confuse them?
  This is the direct test of risk #1.
- Spot-check a handful of misclassified examples qualitatively.

### 6. H2O QN-scheme desk-check (no code, ~1–2 hrs, can run in parallel with steps 1–5)
- Research H2O's actual spectroscopic QN scheme (asymmetric-top rotational numbers J, Ka, Kc;
  normal-mode vibrational (v1,v2,v3)) vs. CO2's linear-molecule, polyad-based combinatorial
  class scheme.
- Determine: does H2O have an analogous near-degenerate-state disambiguation problem, and is the
  actual blocker (a) constructing a graph structure (isotope embeddings, polyad-analogue chains,
  rotational-ladder edges) — which a row-based classifier like Laya would genuinely sidestep —
  or (b) defining the right label/feature representation for an asymmetric top — which is
  unaffected by GNN-vs-Laya and would block either approach equally?
- Write findings to `h2o_feasibility_notes.md`.

### 7. Decision write-up (~30 min)
- Does Laya land within ~5–10 points of the GNN baseline?
- Is training/inference cost on the RTX 2070 tractable for a real (not prototype-scale) run?
- Does the H2O desk-check support or undercut the generalization motivation?
- Recommend: pursue further, or park it — with the reasoning written down either way.

## File layout to create

```
GNN_CO2_QN_assignment/experiments/laya_prototype/
├── pyproject.toml
├── prepare_data.py
├── train_laya.py
├── evaluate.py
├── data/{train,val,test}.jsonl
├── results.md
└── h2o_feasibility_notes.md
```

## Also pending from this session (housekeeping, unrelated to Laya)

- Update project memory / `CLAUDE.md` to mark `ExoMol_CO2_line_list_paper/` as done/published
  (no file moves needed, per user — tracking-only change).

## Time budget

No fixed box (user's call from grilling) — but step 1 is a hard go/no-go gate, and step 7 forces
an explicit stop-or-continue decision rather than open-ended iteration.
