# CO2 Quantum Number Assignment via Graph Neural Network

A bootstrapped GNN pipeline for assigning vibrational quantum numbers (m1, m2, m3, r) to calculated CO2 energy states across 12 isotopologues. Initial proof-of-concept published in the ExoMol CO2 2025 line list "Dozen" ([Barnfield et al. 2025](https://doi.org/10.1093/mnras/staf2135)).

## Problem

CO2 energy level files contain two state types:
- **Ma (MARVEL)**: experimentally validated states with known quantum numbers.
- **Ca (calculated)**: theoretical states with no quantum number labels.

The goal is to assign the four quantum numbers `(m1, m2, m3, r)` to Ca states using the graph structure of the MARVEL dataset as a training signal. The task is transductive — Ma and Ca nodes share the same graph and feature space.

## Repository Structure

```
src/           Core pipeline — library modules and pipeline entry points
  config.py          12 CO2 isotopologue definitions and data paths
  dataset.py         Build unified_co2_graph_data.csv from raw .states.cut files
  graph_builder.py   Construct PyG graph (inter-isotope chains + rotational ladders)
  model.py           CO2AssignmentGNN: GraphSAGE + isotope embeddings + FocalLoss
  train.py           Train GNN, run Hungarian assignment, write assigned_co2_predictions.csv
  bootstrap.py       Harvest confident Ca predictions into the training set
  run_pipeline.py    Orchestrate full 5-generation bootstrap loop
  metrics.py         Calculate F1/MAE on MARVEL test set
  plotting.py        Shared plotting library (thesis-quality figures)
  assignment.py      Assignment summary utility

scripts/       Standalone analysis and publication scripts
  generate_figures.py         Publication-quality figure suite from final predictions
  final_assignment.py         Relaxed MC Dropout assignment (no polyad constraint)
  analyse_predictions.py      Summarise highly-confident Ca assignments post-pipeline
  analyse_uncertainty.py      Compare uncertainty signals as correctness predictors
  analyse_solver_competition.py  Diagnose who wins block conflicts over test states
  analyse_ca_coverage.py      Quantify Ca assignment coverage and energy breakdown
  paper_utils.py              LaTeX table generation for publication

data/          Generated outputs (gitignored)
  unified_co2_graph_data.csv  Master dataset; mutated by bootstrap
  class_mapping.csv           Maps class ID → (m1, m2, m3, r)
  assigned_co2_predictions.csv  All predictions from latest train.py run
  final_relaxed_assignments.csv  Output of final_assignment.py
  run_metrics.json            Per-generation metrics
  figures/                    All generated plots
```

## Architecture

**Graph construction** (`graph_builder.py`): two edge types encode physical structure:
- *Inter-isotope chains*: states linked by energy within the same `(polyad, J, parity)` group — captures perturbation relationships across isotopologues.
- *Intra-isotope rotational ladders*: nearest-energy neighbour within the same `(polyad, isotope)` at adjacent J — captures the ΔJ=1 rotational progression.

**Model** (`model.py`): `CO2AssignmentGNN` — two GraphSAGE layers with residual connections and LayerNorm. Isotope identity is injected via a learned 8-dimensional embedding concatenated to node features before the input projection. Trained with `FocalLoss(gamma=2.0)` for class imbalance. Mini-batched with `NeighborLoader([15, 10], batch_size=2048)`.

**Assignment solver** (`train.py`): states are grouped by `(isotope, J, parity, polyad)` and the Hungarian algorithm enforces physical uniqueness (one state per quantum-number class per group). MARVEL training states are pre-locked to their ground-truth class before the solver runs.

**Confidence metric**: `assigned_margin` — logit of the assigned class minus the runner-up logit, computed post-solver. AUROC = 0.953 on the MARVEL test set. Negative margins flag solver-conflicted states automatically.

**Bootstrap loop** (`run_pipeline.py`): Ca states with `assigned_margin ≥ 1.0` are promoted to the training set each generation. The loop runs for up to 5 generations or until convergence.

## Results

| Metric | Value |
|---|---|
| Raw GNN 4-QN accuracy (MARVEL test set) | 94.71% |
| Post-solver 4-QN accuracy | 90.20% |
| Bootstrap harvest total (5 generations) | ~425,700 Ca states |
| Highly confident Ca assignments (margin ≥ 2.0) | 256,640 |
| Ca assignment coverage | 45.6% |
| Energy coverage: 0–5k cm⁻¹ | 97.2% |
| Energy coverage: 5–10k cm⁻¹ | 75.7% |
| Energy coverage: 10–15k cm⁻¹ | 38.8% |

The 54.4% unassignment rate is physically correct: unassigned Ca states share their GNN-predicted quantum-number class with a MARVEL state in the same `(isotope, J, parity, polyad)` block and are correctly excluded from claiming that slot.

## Installation

```bash
# Requires Python 3.12 and uv
uv sync
```

CUDA 12.6 wheels are fetched automatically via `pyproject.toml`.

## Usage

All scripts must be run via `uv run` from the project root.

```bash
# Full 5-generation bootstrap pipeline (main entry point)
uv run src/run_pipeline.py

# Individual pipeline steps
uv run src/dataset.py          # Rebuild unified_co2_graph_data.csv from raw .states files
uv run src/train.py            # Train GNN + Hungarian assignment → assigned_co2_predictions.csv
uv run src/bootstrap.py        # Harvest confident predictions into the training set
uv run src/metrics.py          # Print final F1/MAE metrics on the MARVEL test set

# Post-pipeline analysis and figure generation
uv run scripts/generate_figures.py         # Publication figures from final predictions
uv run scripts/final_assignment.py         # Relaxed MC Dropout assignment
uv run scripts/analyse_predictions.py      # Summarise confident new Ca assignments
uv run scripts/analyse_uncertainty.py      # Uncertainty signal comparison (AUROC table)
uv run scripts/analyse_solver_competition.py  # Solver competition asymmetry analysis
uv run scripts/analyse_ca_coverage.py      # Ca coverage breakdown by isotopologue/energy
uv run scripts/paper_utils.py              # Generate LaTeX yield table
```
