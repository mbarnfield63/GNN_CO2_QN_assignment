import pandas as pd
import os
import time

DATA_DIR = "data"
UNIFIED_DATASET_PATH = os.path.join(DATA_DIR, "unified_co2_graph_data.csv")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")
CACHE_PATH = os.path.join(DATA_DIR, "cached_pyg_graph.pt")

# ── Harvesting threshold ──────────────────────────────────────────────────────
# Uses assigned_margin (post-Hungarian) rather than raw logit_margin.
# This naturally excludes Hungarian-conflicted nodes since the solver forces
# their assigned_margin toward zero, so they fail the threshold without
# needing an explicit conflict filter.
MARGIN_THRESHOLD = 1.0


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

    # ── Prefer assigned_margin; fall back to logit_margin if absent ──────────
    margin_col = (
        "assigned_margin" if "assigned_margin" in df_preds.columns else "logit_margin"
    )
    if margin_col == "logit_margin":
        print(
            "Warning: assigned_margin not found in predictions — "
            "falling back to logit_margin. Update train.py."
        )

    # ── Identify confidently-assigned, previously-unharvested inference states ─
    # Conditions:
    #   1. Not MARVEL (those are locked ground truth)
    #   2. Successfully assigned by the solver (pred_class_id != -1)
    #   3. Assigned margin >= threshold (filters conflicts automatically)
    #   4. Not yet harvested in a prior generation
    confident_mask = (
        ~df_preds["is_marvel"]
        & (df_preds["pred_class_id"] != -1)
        & (df_preds[margin_col] >= MARGIN_THRESHOLD)
        & (df_orig["assignment_generation"] == 0)
    )

    confident_nodes = df_preds[confident_mask]["node_id"].values
    num_new_train = len(confident_nodes)

    if num_new_train == 0:
        print("No highly confident states found to harvest. Pipeline complete.")
        return False  # Signal to run_pipeline.py that we should stop early

    print(f"Harvesting {num_new_train:,} states for Generation {current_gen}...")

    # ── Update masks ─────────────────────────────────────────────────────────
    node_set = set(confident_nodes)
    mask = df_orig["node_id"].isin(node_set)

    df_orig.loc[mask, "train_mask"] = True
    df_orig.loc[mask, "val_mask"] = False
    df_orig.loc[mask, "test_mask"] = False

    # ── Write predicted class back so build_polyad_class_map can extend
    #    the valid class pool in future generations ─────────────────────────
    pred_id_map = df_preds.set_index("node_id")["pred_class_id"]
    margin_map = df_preds.set_index("node_id")[margin_col]

    df_orig.loc[mask, "combinatorial_class_id"] = df_orig.loc[mask, "node_id"].map(
        pred_id_map
    )
    df_orig.loc[mask, "assignment_generation"] = current_gen
    df_orig.loc[mask, "locked_margin"] = df_orig.loc[mask, "node_id"].map(margin_map)

    # ── Per-generation summary statistics ────────────────────────────────────
    harvested_margins = df_orig.loc[mask, "locked_margin"]
    print(f"\n--- Generation {current_gen} Harvest Summary ---")
    print(f"  States harvested : {num_new_train:,}")
    print(f"  Margin  min      : {harvested_margins.min():.3f}")
    print(f"  Margin  median   : {harvested_margins.median():.3f}")
    print(f"  Margin  max      : {harvested_margins.max():.3f}")
    print(f"  Total train set  : {df_orig['train_mask'].sum():,}")

    # ── Save and invalidate graph cache ──────────────────────────────────────
    df_orig.to_csv(UNIFIED_DATASET_PATH, index=False)
    print(f"\nUpdated {UNIFIED_DATASET_PATH}.")

    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        print(f"Deleted stale PyG cache: {CACHE_PATH}")

    elapsed = time.time() - start_time
    print(f"Bootstrap runtime: {elapsed:.2f}s")
    return True  # Signal to run_pipeline.py that harvesting succeeded


if __name__ == "__main__":
    run_bootstrap()
