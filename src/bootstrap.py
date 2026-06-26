import sys
import json
import pandas as pd
import os
import time

from config import UNIFIED_DATASET_PATH, PREDICTIONS_PATH, GRAPH_CACHE_PATH, CLASS_MAPPING_PATH, BOOTSTRAP_METRICS_PATH

# ── Harvesting threshold ──────────────────────────────────────────────────────
# Uses assigned_prob (softmax probability of the solver-selected class).
# Conflicted nodes receive a lower assigned_prob than their model-top-1 class,
# so they are naturally filtered without an explicit conflict check.
# At PROB_THRESHOLD=0.75: mean macro-F1 drop of 0.01 pp vs t=0.95 baseline (negligible).
PROB_THRESHOLD = 0.75

_GROUP_COLS = ["isotope_id", "polyad_int", "J", "parity_encoded", "m1", "m2", "m3"]


def _apply_r_correction(df_orig, df_preds, class_map):
    """
    Re-rank r within each (isotopologue, polyad, J, parity, m1, m2, m3) group
    by energy descending to enforce the AFGL convention (r=1 = highest energy).

    MARVEL states anchor the ranking via ground-truth QNs. Only bootstrapped
    Ca states have their combinatorial_class_id updated where the energy rank
    disagrees with their current r. Partially-assigned groups are corrected
    relative to all currently assigned states in the group; later generations
    will refine further as additional states are harvested.

    Returns (df_orig, n_swaps).
    """
    qn_to_class_id = {
        (int(row.m1), int(row.m2), int(row.m3), int(row.r)): int(row.class_id)
        for _, row in class_map.iterrows()
    }
    cm = class_map.rename(columns={"class_id": "class_id_int", "r": "current_r"})

    # ── MARVEL states: use ground-truth AFGL QNs as energy anchors ────────────
    mv = df_preds.loc[
        df_preds["is_marvel"],
        ["node_id", "energy", "isotope_id", "polyad_int", "J", "parity_encoded",
         "AFGL_m1", "AFGL_m2", "AFGL_m3"]
    ].copy()
    mv = mv.rename(columns={"AFGL_m1": "m1", "AFGL_m2": "m2", "AFGL_m3": "m3"})
    mv["is_ca"] = False
    mv["current_r"] = -1  # unused; MARVEL r is never updated

    # ── Bootstrapped Ca states: decode current (m1, m2, m3, r) from class_id ──
    ca_orig = df_orig.loc[
        df_orig["assignment_generation"] > 0,
        ["node_id", "combinatorial_class_id"]
    ].dropna().copy()
    ca_orig["class_id_int"] = ca_orig["combinatorial_class_id"].astype(int)
    ca_orig = ca_orig.merge(cm[["class_id_int", "m1", "m2", "m3", "current_r"]],
                            on="class_id_int", how="inner")

    ca_feats = df_preds.loc[
        df_preds["node_id"].isin(ca_orig["node_id"]),
        ["node_id", "energy", "isotope_id", "polyad_int", "J", "parity_encoded"]
    ]
    ca = ca_feats.merge(ca_orig[["node_id", "m1", "m2", "m3", "current_r"]], on="node_id")
    ca["is_ca"] = True

    # ── Rank by energy descending within each vibrational group ───────────────
    combined = pd.concat([mv, ca], ignore_index=True)
    combined["corrected_r"] = (
        combined.groupby(_GROUP_COLS)["energy"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # ── Find Ca rows where energy rank disagrees with current r ───────────────
    ca_rows = combined[combined["is_ca"]].copy()
    needs_fix = ca_rows[ca_rows["corrected_r"] != ca_rows["current_r"].astype(int)].copy()
    needs_fix["key"] = list(zip(
        needs_fix["m1"].astype(int),
        needs_fix["m2"].astype(int),
        needs_fix["m3"].astype(int),
        needs_fix["corrected_r"],
    ))
    needs_fix["new_class_id"] = needs_fix["key"].map(qn_to_class_id)
    valid = needs_fix.dropna(subset=["new_class_id"])

    if not valid.empty:
        update_map = valid.set_index("node_id")["new_class_id"].astype(int)
        ca_mask = df_orig["node_id"].isin(update_map.index)
        df_orig.loc[ca_mask, "combinatorial_class_id"] = (
            df_orig.loc[ca_mask, "node_id"].map(update_map)
        )

    return df_orig, len(valid)


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

    # ── r-ordering correction ────────────────────────────────────────────────
    # Re-rank r by energy descending within each (isotopologue, polyad, J,
    # parity, m1, m2, m3) group, using MARVEL states as anchors.  Applied to
    # all bootstrapped Ca states so that pseudo-labels entering the next
    # generation already satisfy the AFGL energy-ordering convention.
    class_map = pd.read_csv(CLASS_MAPPING_PATH)
    df_orig, n_r_swaps = _apply_r_correction(df_orig, df_preds, class_map)
    print(f"  r-label corrections : {n_r_swaps:,}  (energy-rank disagreements corrected)")

    # ── Persist r-correction count to sidecar ────────────────────────────────
    boot_metrics = []
    if os.path.exists(BOOTSTRAP_METRICS_PATH):
        with open(BOOTSTRAP_METRICS_PATH) as f:
            boot_metrics = json.load(f)
    boot_metrics.append({"generation": current_gen, "n_r_corrections": n_r_swaps})
    with open(BOOTSTRAP_METRICS_PATH, "w") as f:
        json.dump(boot_metrics, f)

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
