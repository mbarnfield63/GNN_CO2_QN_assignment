"""
Isolated CDSD-comparison side-run.

Runs the real pipeline (src/dataset.py, src/train.py, src/bootstrap.py --
unmodified, imported in-process) against a copy of the 12-isotopologue
dataset where 626's CDSD-2024-PI (EH) states have been patched to look like
ordinary unlabeled Ca states (see prepare_cdsd_data.py). Everything writes to
its own data/ subfolder here, so the published pipeline outputs under
GNN_CO2_QN_assignment/data/ are never touched.

Must be run from the GNN_CO2_QN_assignment project root:
    uv run analysis/cdsd_comparison/run_side_pipeline.py
"""

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
SIDE_DATA_DIR = os.path.join(HERE, "data")
PATCHED_626_FILE = os.path.join(
    HERE, "states_patched", "12C-16O2__Dozen_cdsd_patched.states.cut"
)

ITERATIONS = 5

sys.path.insert(0, SRC_DIR)

import config  # noqa: E402  (must patch before importing dataset/train/bootstrap)

os.makedirs(SIDE_DATA_DIR, exist_ok=True)
config.DATA_DIR = SIDE_DATA_DIR
config.UNIFIED_DATASET_PATH = os.path.join(SIDE_DATA_DIR, "unified_co2_graph_data.csv")
config.CLASS_MAPPING_PATH = os.path.join(SIDE_DATA_DIR, "class_mapping.csv")
config.GRAPH_CACHE_PATH = os.path.join(SIDE_DATA_DIR, "cached_pyg_graph.pt")
config.PREDICTIONS_PATH = os.path.join(SIDE_DATA_DIR, "assigned_co2_predictions.csv")
config.BOOTSTRAP_METRICS_PATH = os.path.join(SIDE_DATA_DIR, "bootstrap_metrics.json")

# Only 626's file is swapped (for an absolute path, os.path.join(STATES_DIR, file)
# in dataset.py discards STATES_DIR entirely) -- all 11 other isotopologues load
# unmodified from the shared raw data store.
config.ISOTOPES = copy.deepcopy(config.ISOTOPES)
for iso in config.ISOTOPES:
    if iso["id"] == "626":
        iso["file"] = PATCHED_626_FILE

# Imported only now, so they bind the patched config values above.
import dataset  # noqa: E402
import train  # noqa: E402
import bootstrap  # noqa: E402


def main():
    print("=" * 60)
    print("=== CDSD SIDE-RUN: building patched unified dataset ===")
    print("=" * 60)
    dataset.create_unified_dataset()

    print("\n" + "=" * 60)
    print("=== GENERATION 0: INITIAL TRAINING RUN ===")
    print("=" * 60)
    train.main()

    for i in range(1, ITERATIONS + 1):
        print("\n" + "=" * 60)
        print(f"=== BOOTSTRAP CYCLE {i} OF {ITERATIONS} ===")
        print("=" * 60)

        if not bootstrap.run_bootstrap():
            print(f"Bootstrap cycle {i} found no new states. Pipeline converged early.")
            break

        train.main()

    print("\nSide-run complete. Predictions written to", config.PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
