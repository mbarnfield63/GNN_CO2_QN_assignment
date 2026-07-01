import sys
import json
import pandas as pd
import os
import time

from config import (
    UNIFIED_DATASET_PATH,
    PREDICTIONS_PATH,
    GRAPH_CACHE_PATH,
    CLASS_MAPPING_PATH,
    BOOTSTRAP_METRICS_PATH,
)

# ── Harvesting threshold ──────────────────────────────────────────────────────
# Uses assigned_prob (softmax probability of the solver-selected class).
# Conflicted nodes receive a lower assigned_prob than their model-top-1 class,
# so they are naturally filtered without an explicit conflict check.
# At PROB_THRESHOLD=0.75: mean macro-F1 drop of 0.01 pp vs t=0.95 baseline (negligible).
PROB_THRESHOLD = 0.80

_GROUP_COLS = ["isotope_id", "polyad_int", "J", "parity_encoded", "m1", "m2", "m3"]
_COLS = [
    "node_id",
    "energy",
    "isotope_id",
    "polyad_int",
    "J",
    "parity_encoded",
    "m1",
    "m2",
    "m3",
    "r",
    "is_new",
]


def _filter_r_inversions(df_orig, df_preds, class_map, candidate_node_ids):
    """
    Return node_ids to exclude from the current harvest due to r-ordering inversions.

    AFGL convention: within each (isotopologue, polyad, J, parity, m1, m2, m3)
    group, E(r=1) > E(r=2) > ...  A consecutive pair that violates this is an
    inversion.  Any candidate state involved in an inversion is withheld from
    this harvest; both members of a candidate-only inversion are removed.  The
    group is reconsidered next generation when more Ca context is available.

    Does not re-check previously bootstrapped states — no cumulative correction.
    """
    cm = class_map[["class_id", "m1", "m2", "m3", "r"]].copy()

    # ── MARVEL anchors ────────────────────────────────────────────────────────
    mv = (
        df_preds.loc[
            df_preds["is_marvel"],
            [
                "node_id",
                "energy",
                "isotope_id",
                "polyad_int",
                "J",
                "parity_encoded",
                "AFGL_m1",
                "AFGL_m2",
                "AFGL_m3",
                "AFGL_r",
            ],
        ]
        .dropna(subset=["AFGL_r"])
        .copy()
        .rename(
            columns={"AFGL_m1": "m1", "AFGL_m2": "m2", "AFGL_m3": "m3", "AFGL_r": "r"}
        )
    )
    mv["is_new"] = False

    # ── Previously bootstrapped Ca states (anchors from prior generations) ───
    prev = (
        df_orig.loc[
            df_orig["assignment_generation"] > 0, ["node_id", "combinatorial_class_id"]
        ]
        .dropna()
        .copy()
    )

    frames = [mv[_COLS]]
    if not prev.empty:
        prev = prev.rename(columns={"combinatorial_class_id": "class_id"})
        prev["class_id"] = prev["class_id"].astype(int)
        prev = prev.merge(cm, on="class_id", how="inner")
        feats = df_preds.loc[
            df_preds["node_id"].isin(prev["node_id"]),
            ["node_id", "energy", "isotope_id", "polyad_int", "J", "parity_encoded"],
        ]
        prev = feats.merge(prev[["node_id", "m1", "m2", "m3", "r"]], on="node_id")
        prev["is_new"] = False
        frames.append(prev[_COLS])

    # ── Candidates for this harvest ───────────────────────────────────────────
    cand = (
        df_preds.loc[
            df_preds["node_id"].isin(candidate_node_ids),
            [
                "node_id",
                "energy",
                "isotope_id",
                "polyad_int",
                "J",
                "parity_encoded",
                "pred_class_id",
            ],
        ]
        .copy()
        .rename(columns={"pred_class_id": "class_id"})
    )
    cand["class_id"] = cand["class_id"].astype(int)
    cand = cand.merge(cm, on="class_id", how="inner")
    cand["is_new"] = True
    frames.append(cand[_COLS])

    combined = pd.concat(frames, ignore_index=True)

    # ── Detect inversions via consecutive-pair check (sorted by r) ────────────
    # Checking consecutive pairs is sufficient: any non-monotonicity shows up
    # in at least one adjacent pair.  Missing r values (gaps) are handled
    # correctly — {r=1, r=3} with E(r=1) > E(r=3) passes with no correction.
    exclude_ids = set()
    for _, grp in combined.groupby(_GROUP_COLS):
        if not grp["is_new"].any():
            continue
        grp_s = grp.sort_values("r").reset_index(drop=True)
        for i in range(len(grp_s) - 1):
            if grp_s.at[i, "energy"] < grp_s.at[i + 1, "energy"]:  # inversion
                if grp_s.at[i, "is_new"]:
                    exclude_ids.add(grp_s.at[i, "node_id"])
                if grp_s.at[i + 1, "is_new"]:
                    exclude_ids.add(grp_s.at[i + 1, "node_id"])

    return exclude_ids


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
    confident_mask = (
        ~df_preds["is_marvel"]
        & (df_preds["pred_class_id"] != -1)
        & (df_preds["assigned_prob"] >= PROB_THRESHOLD)
        & (df_orig["assignment_generation"] == 0)
    )

    candidate_nodes = df_preds[confident_mask]["node_id"].values

    if len(candidate_nodes) == 0:
        print("No highly confident states found to harvest. Pipeline complete.")
        return False

    # ── r-inversion filter ───────────────────────────────────────────────────
    # Withholds candidates whose r label violates energy ordering within their
    # group.  Both members of a candidate-only inversion are excluded.  No
    # cumulative re-check of previously bootstrapped states.
    class_map = pd.read_csv(CLASS_MAPPING_PATH)
    inversion_ids = _filter_r_inversions(df_orig, df_preds, class_map, candidate_nodes)
    n_r_excluded = len(inversion_ids)
    if inversion_ids:
        candidate_nodes = [n for n in candidate_nodes if n not in inversion_ids]

    num_new_train = len(candidate_nodes)
    if num_new_train == 0:
        print("All candidates had r-ordering inversions; nothing harvested.")
        return False

    print(
        f"Harvesting {num_new_train:,} states for Generation {current_gen} "
        f"({n_r_excluded:,} withheld for r-inversion)..."
    )

    # ── Update masks ─────────────────────────────────────────────────────────
    mask = df_orig["node_id"].isin(candidate_nodes)

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
    print(f"  States harvested     : {num_new_train:,}")
    print(f"  r-inversions withheld: {n_r_excluded:,}")
    print(f"  Prob  min            : {harvested_probs.min():.3f}")
    print(f"  Prob  median         : {harvested_probs.median():.3f}")
    print(f"  Prob  max            : {harvested_probs.max():.3f}")
    print(f"  Total train set      : {df_orig['train_mask'].sum():,}")

    # ── Persist metrics to sidecar ───────────────────────────────────────────
    boot_metrics = []
    if os.path.exists(BOOTSTRAP_METRICS_PATH):
        with open(BOOTSTRAP_METRICS_PATH) as f:
            boot_metrics = json.load(f)
    boot_metrics.append({"generation": current_gen, "n_r_excluded": n_r_excluded})
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
    return True


if __name__ == "__main__":
    if not run_bootstrap():
        sys.exit(2)
