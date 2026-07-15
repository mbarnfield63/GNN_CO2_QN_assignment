# CDSD-2024-PI comparison (isotopologue 626)

Validates the GNN's AFGL assignments against an independent source: the
CDSD-2024-PI effective-Hamiltonian quantum numbers already present in the raw
`12C-16O2__Dozen.states.cut` file (`Source == "EH"`), which the main pipeline
currently drops entirely (`src/dataset.py` keeps only `Ma`/`Ca`). Only 626
uses CDSD-2024-PI as its EH source -- every other isotopologue's `EH`/`HI` tag
comes from HITRAN2020, so this comparison is 626-only by design.

This is a read-only side-run: it never touches the live pipeline's
`data/` outputs, `config.py`, or any published numbers. It reuses the real
`src/` pipeline code in-process via `sys.path` + a runtime `config` patch
(no duplicated pipeline logic, no drift risk).

## Important: run on GPU hardware

This was built and syntax-checked in an environment with no GPU. The current
`src/train.py` trains full-batch (no `NeighborLoader` mini-batching), so a
CPU-only run against the ~2.5M-node 12-isotopologue graph may be impractically
slow. Run this on the same GPU hardware you use for the main pipeline.

Before running: `src/train.py`'s dropped `train_model()` call was restored
(a merge on 2026-07-01 had silently removed it -- the published `data/`
results predate that merge and are unaffected, but HEAD was broken until this
fix). Confirm your checkout includes that fix before running.

## Steps

```bash
# 1. Patch the 626 states file (EH -> Ca, stash true CDSD labels).
#    Only needs to be run once; outputs are already committed.
uv run analysis/cdsd_comparison/prepare_cdsd_data.py

# 2. Run the isolated pipeline (Gen 0 + up to 5 bootstrap generations).
#    This is a full retrain -- expect it to take roughly as long as one
#    full run of src/run_pipeline.py, since it's the same 12-isotopologue
#    graph with ~11,230 extra nodes in 626.
uv run analysis/cdsd_comparison/run_side_pipeline.py

# 3. Score direct 4-QN / per-QN agreement against the CDSD truth.
uv run analysis/cdsd_comparison/compare_to_cdsd.py

# 4. Score B_v (rotational constant) agreement, reusing the paper's existing
#    fit_groups() methodology from scripts/analyse_spectroscopic_constants.py.
uv run analysis/cdsd_comparison/compare_bv_cdsd.py
```

## Files

| File | Purpose |
|---|---|
| `prepare_cdsd_data.py` | Patches 626's raw states file; writes `cdsd_true_labels.csv` |
| `states_patched/` | Scratch copy of 626 with EH rows relabeled to Ca |
| `cdsd_true_labels.csv` | Stashed ground truth (energy, J, parity, m1/m2/m3/r) for the 11,230 CDSD states |
| `run_side_pipeline.py` | Orchestrator: monkeypatches `config`, runs Gen 0 + bootstrap loop in-process, writes to `data/` (this folder) |
| `data/` | Isolated outputs (own `unified_co2_graph_data.csv`, `class_mapping.csv`, `assigned_co2_predictions.csv`, etc.) -- never the main pipeline's `data/` |
| `compare_to_cdsd.py` | Joins predictions back to the CDSD truth; reports per-QN and 4-QN agreement |
| `compare_bv_cdsd.py` | B_v fit comparison (CDSD-labelled bands vs GNN-predicted bands) |

## Known result to expect: the class-space ceiling

Of the 11,026 CDSD states under the standard 15,000 cm⁻¹ cutoff, **56.3%
(6,212 states) carry an (m1, m2, m3, r) combination that does not exist
anywhere in `class_mapping.csv`**, because that mapping is built exclusively
from MARVEL data. The model cannot predict a class it was never given as an
option, regardless of how good it is -- this is a structural ceiling, not a
model failure. `compare_to_cdsd.py` reports the reachable subset (4,814
states) as the headline agreement number and the full-set number as a
footnote; don't read the full-set percentage as raw model accuracy.
