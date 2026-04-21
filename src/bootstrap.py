import pandas as pd
import os
import time

DATA_DIR = "data"
UNIFIED_DATASET_PATH = os.path.join(DATA_DIR, "unified_co2_graph_data.csv")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")
CACHE_PATH = os.path.join(DATA_DIR, "cached_pyg_graph.pt")

VARIANCE_THRESHOLD = 0.05


def run_bootstrap():
    start_time = time.time()
    print(f"Loading current dataset and recent predictions...")
    df_orig = pd.read_csv(UNIFIED_DATASET_PATH)
    df_preds = pd.read_csv(PREDICTIONS_PATH)

    # Initialize tracking columns if they don't exist
    if "assignment_generation" not in df_orig.columns:
        df_orig["assignment_generation"] = 0  # 0 means original MARVEL data
        df_orig["locked_variance"] = 0.0  # MARVEL data has 0 uncertainty

    # Determine the current generation number
    current_gen = df_orig["assignment_generation"].max() + 1

    # 1. Identify highly confident new assignments
    # Must be: Not MARVEL originally, successfully mapped, highly confident,
    # and not already harvested in a previous generation!
    confident_mask = (
        (df_preds["is_marvel"] == False)
        & (df_preds["pred_class_id"] != -1)
        & (df_preds["assignment_variance"] <= VARIANCE_THRESHOLD)
        & (df_orig["assignment_generation"] == 0)
    )

    confident_nodes = df_preds[confident_mask]["node_id"].values
    num_new_train = len(confident_nodes)

    if num_new_train == 0:
        print("No highly confident states found to bootstrap. Halting.")
        return

    print(f"Harvesting {num_new_train:,} states for Generation {current_gen}...")

    # 2. Update the original dataframe (Masks and Labels)
    df_orig.loc[df_orig["node_id"].isin(confident_nodes), "train_mask"] = True
    df_orig.loc[df_orig["node_id"].isin(confident_nodes), "val_mask"] = False
    df_orig.loc[df_orig["node_id"].isin(confident_nodes), "test_mask"] = False

    pred_mapping = df_preds.set_index("node_id")["pred_class_id"]
    df_orig.loc[df_orig["node_id"].isin(confident_nodes), "combinatorial_class_id"] = (
        df_orig["node_id"].map(pred_mapping)
    )

    # Lock in the Generation and the exact Variance for these new assignments
    df_orig.loc[df_orig["node_id"].isin(confident_nodes), "assignment_generation"] = (
        current_gen
    )

    variance_mapping = df_preds.set_index("node_id")["assignment_variance"]
    df_orig.loc[df_orig["node_id"].isin(confident_nodes), "locked_variance"] = df_orig[
        "node_id"
    ].map(variance_mapping)

    # 3. Save the new dataset
    df_orig.to_csv(UNIFIED_DATASET_PATH, index=False)
    print(f"Updated {UNIFIED_DATASET_PATH} with Generation {current_gen} data.")

    # 4. Destroy the old cache
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        print(f"Deleted old PyG cache: {CACHE_PATH}")

    print("\nBootstrapping Complete!")
    print(
        f"Generation {current_gen} added {num_new_train:,} new training states."
        f" Total training states: {df_orig['train_mask'].sum():,}."
    )
    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    run_bootstrap()
