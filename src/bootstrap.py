import sys
import pandas as pd
import os
import time

from config import UNIFIED_DATASET_PATH, PREDICTIONS_PATH, GRAPH_CACHE_PATH

# ── Harvesting threshold ──────────────────────────────────────────────────────
# Uses assigned_prob (softmax probability of the solver-selected class).
# Conflicted nodes receive a lower assigned_prob than their model-top-1 class,
# so they are naturally filtered without an explicit conflict check.
# At PROB_THRESHOLD=0.85: 99.7% precision on the MARVEL test set.
PROB_THRESHOLD = 0.85


def run_bootstrap():
    start_time = time.time()
    print(f"Loading current dataset and recent predictions...")
    df_orig = pd.read_csv(UNIFIED_DATASET_PATH)
    df_preds = pd.read_csv(PREDICTIONS_PATH)

    # ── Initialise tracking columns on first generation ──────────────────────
    if "assignment_generation" not in df_orig.columns:
        df_orig["assignment_generation"] = 0  # 0 = original MARVEL data
        df_orig["locked_margin"] = 10.0  # high sentinel for ground truth

    current_gen = int(df_orig["assignment_generation"].max()) + 1
    print(f"Preparing Generation {current_gen} harvest...")

    # ── Identify confidently-assigned, previously-unharvested inference states ─
    # Conditions:
    #   1. Not MARVEL (those are locked ground truth)
    #   2. Successfully assigned by the solver (pred_class_id != -1)
    #   3. Assigned softmax prob >= threshold (filters low-confidence and conflicts)
    #   4. Not yet harvested in a prior generation
    confident_mask = (
        ~df_preds["is_marvel"]
        & (df_preds["pred_class_id"] != -1)
        & (df_preds["assigned_prob"] >= PROB_THRESHOLD)
        & (df_orig["assignment_generation"] == 0)
    )

    confident_nodes = df_preds[confident_mask]["node_id"].values
    num_new_train = len(confident_nodes)

    if num_new_train == 0:
        print("No highly confident states found to harvest. Pipeline complete.")
        return False  # Signal to run_pipeline.py that we should stop early

    print(f"Harvesting {num_new_train:,} states for Generation {current_gen}...")

    # ── Update masks ─────────────────────────────────────────────────────────
    mask = df_orig["node_id"].isin(confident_nodes)

    df_orig.loc[mask, "train_mask"] = True
    df_orig.loc[mask, "val_mask"] = False
    df_orig.loc[mask, "test_mask"] = False

    # ── Write predicted class back so build_polyad_class_map can extend
    #    the valid class pool in future generations ─────────────────────────
    pred_id_map = df_preds.set_index("node_id")["pred_class_id"]
    prob_map = df_preds.set_index("node_id")["assigned_prob"]

    df_orig.loc[mask, "combinatorial_class_id"] = df_orig.loc[mask, "node_id"].map(
        pred_id_map
    )
    df_orig.loc[mask, "assignment_generation"] = current_gen
    df_orig.loc[mask, "locked_prob"] = df_orig.loc[mask, "node_id"].map(prob_map)

    # ── Per-generation summary statistics ────────────────────────────────────
    harvested_probs = df_orig.loc[mask, "locked_prob"]
    print(f"\n--- Generation {current_gen} Harvest Summary ---")
    print(f"  States harvested : {num_new_train:,}")
    print(f"  Prob  min        : {harvested_probs.min():.3f}")
    print(f"  Prob  median     : {harvested_probs.median():.3f}")
    print(f"  Prob  max        : {harvested_probs.max():.3f}")
    print(f"  Total train set  : {df_orig['train_mask'].sum():,}")

    # ── Save and invalidate graph cache ──────────────────────────────────────
    df_orig.to_csv(UNIFIED_DATASET_PATH, index=False)
    print(f"\nUpdated {UNIFIED_DATASET_PATH}.")

    if os.path.exists(GRAPH_CACHE_PATH):
        os.remove(GRAPH_CACHE_PATH)
        print(f"Deleted stale PyG cache: {GRAPH_CACHE_PATH}")

    elapsed = time.time() - start_time
    print(f"Bootstrap runtime: {elapsed:.2f}s")
    return True  # Signal to run_pipeline.py that harvesting succeeded


if __name__ == "__main__":
    if not run_bootstrap():
        sys.exit(2)
