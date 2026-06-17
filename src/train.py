import json
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
import os
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import custom modules
from graph_builder import build_pyg_graph
from model import CO2AssignmentGNN, FocalLoss
from plotting import plot_variance_validation

DATA_DIR = "data"
UNIFIED_DATASET_PATH = os.path.join(DATA_DIR, "unified_co2_graph_data.csv")
CLASS_MAPPING_PATH = os.path.join(DATA_DIR, "class_mapping.csv")
GRAPH_CACHE_PATH = os.path.join(DATA_DIR, "cached_pyg_graph.pt")

FEATURE_COLS = [
    "energy",
    "J",
    "parity_encoded",
    "dom_coeff",
    "t1",
    "t2",
    "t3",
    "polyad",
    "is_symmetric",
    "C_mass",
    "O_A_mass",
    "O_B_mass",
]


def load_and_prepare_data():
    print("Loading datasets...")
    df = pd.read_csv(UNIFIED_DATASET_PATH)
    mapping_df = pd.read_csv(CLASS_MAPPING_PATH)

    le_iso = LabelEncoder()
    df["iso_idx_encoded"] = le_iso.fit_transform(df["isotope_id"].astype(str))
    num_isotopes = len(le_iso.classes_)
    num_classes = len(mapping_df)

    scaler = StandardScaler()
    train_df = df[df["train_mask"] == True]
    scaler.fit(train_df[FEATURE_COLS])
    df["polyad_int"] = df["polyad"].copy()
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    if os.path.exists(GRAPH_CACHE_PATH):
        print("Loading cached PyG graph from disk...")
        data = torch.load(GRAPH_CACHE_PATH, weights_only=False)
    else:
        data = build_pyg_graph(df)
        print("Saving PyG graph to disk for future runs...")
        torch.save(data, GRAPH_CACHE_PATH)

    return data, len(FEATURE_COLS), num_isotopes, num_classes, df, mapping_df, scaler


def export_run_metrics(final_df, data_dir="data"):
    """
    Writes key metrics from this training run to a JSON file so
    run_pipeline.py can accumulate them across bootstrap generations.
    """
    test_df = final_df[final_df["test_mask"] == True]

    is_correct = (
        (test_df["AFGL_m1"] == test_df["pred_m1"])
        & (test_df["AFGL_m2"] == test_df["pred_m2"])
        & (test_df["AFGL_m3"] == test_df["pred_m3"])
        & (test_df["AFGL_r"] == test_df["pred_r"])
    )

    current_gen = int(final_df["assignment_generation"].max())
    n_assigned = int((final_df["pred_class_id"] >= 0).sum())
    n_total = len(final_df)
    n_harvested = int((final_df["assignment_generation"] == current_gen).sum())

    metrics = {
        "generation": current_gen,
        "accuracy_4qn": float(is_correct.mean()) if len(test_df) > 0 else 0.0,
        "mae_m1": float(abs(test_df["AFGL_m1"] - test_df["pred_m1"]).mean()),
        "mae_m2": float(abs(test_df["AFGL_m2"] - test_df["pred_m2"]).mean()),
        "mae_m3": float(abs(test_df["AFGL_m3"] - test_df["pred_m3"]).mean()),
        "mae_r": float(abs(test_df["AFGL_r"] - test_df["pred_r"]).mean()),
        "n_assigned": n_assigned,
        "assignment_rate": n_assigned / n_total,
        "n_harvested_this_gen": n_harvested,
        "n_conflicts": int(final_df["hungarian_conflict"].sum()),
        "conflict_accuracy": (
            float(
                (
                    (
                        final_df.loc[
                            final_df["hungarian_conflict"] & final_df["is_marvel"],
                            "AFGL_m1",
                        ]
                        == final_df.loc[
                            final_df["hungarian_conflict"] & final_df["is_marvel"],
                            "pred_m1",
                        ]
                    )
                ).mean()
            )
            if final_df["hungarian_conflict"].any()
            else 0.0
        ),
        "median_margin": (
            float(
                final_df.loc[
                    final_df["assignment_generation"] == current_gen, "locked_margin"
                ].median()
            )
            if current_gen > 0
            else 0.0
        ),
    }

    out_path = os.path.join(data_dir, "run_metrics.json")
    # Append to history list
    history = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            history = json.load(f)
    history.append(metrics)
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Metrics written to {out_path}")
    return metrics


def evaluate_batched(model, loader, device):
    """Calculates Top-1 Accuracy using mini-batches."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.iso_idx)

            # The target nodes are the first 'batch_size' nodes in the sampled graph
            preds = out[: batch.batch_size].argmax(dim=1)
            true = batch.y[: batch.batch_size]

            correct += (preds == true).sum().item()
            total += batch.batch_size

    return correct / total if total > 0 else 0.0


def build_polyad_class_map(df, margin_threshold=0.0):
    """
    Build polyad -> valid class_ids from MARVEL states, optionally extended
    by high-confidence bootstrapped predictions for sparse high-energy polyads.
    """
    # Seed from MARVEL ground truth (always authoritative)
    marvel_states = df[df["is_marvel"] == True][
        ["combinatorial_class_id", "polyad_int"]
    ].dropna()

    polyad_map = (
        marvel_states.groupby("polyad_int")["combinatorial_class_id"]
        .apply(lambda x: set(x.unique().astype(int)))
        .to_dict()
    )

    # Extend from bootstrapped generations if available
    if "assignment_generation" in df.columns and margin_threshold > 0:
        margin_col = (
            "assigned_margin" if "assigned_margin" in df.columns else "locked_margin"
        )

        confident = df[
            (df["assignment_generation"] > 0)
            & (df[margin_col] >= margin_threshold)
            & (df["combinatorial_class_id"] >= 0)
        ][["combinatorial_class_id", "polyad_int"]].dropna()

        for polyad_val, group in confident.groupby("polyad_int"):
            new_classes = set(group["combinatorial_class_id"].unique().astype(int))
            if polyad_val in polyad_map:
                polyad_map[polyad_val] |= new_classes
            else:
                polyad_map[polyad_val] = new_classes

    # Convert sets to sorted lists for deterministic indexing
    return {k: sorted(v) for k, v in polyad_map.items()}


def evaluate_physical_assignment(
    model, loader, device, num_nodes, df, mapping_df, scaler
):
    """Enforces 1-to-1 physical constraints locally and decodes combinatorial classes."""
    model.eval()

    print("\nExtracting logits via standard evaluation pass...")
    all_logits, mean_probs = model.get_logits_and_probs(loader, device, num_nodes)

    print("Extracting raw argmax predictions and margins...")
    all_logits_cpu = all_logits.cpu().numpy()
    mean_probs_cpu = mean_probs.cpu().numpy()

    # Get the Top 1 choices
    raw_class_indices = np.argmax(all_logits_cpu, axis=1)
    df["raw_class_id"] = raw_class_indices

    # Sort the logits along the class dimension
    sorted_logits = np.sort(all_logits_cpu, axis=1)
    # The margin is the highest logit minus the second-highest logit
    logit_margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    df["logit_margin"] = logit_margin

    print("Decoding raw combinatorial classes...")
    class_to_quantum = mapping_df.set_index("class_id")[
        ["m1", "m2", "m3", "r"]
    ].to_dict("index")

    df["raw_m1"] = df["raw_class_id"].map(
        lambda cid: class_to_quantum.get(cid, {"m1": -1})["m1"]
    )
    df["raw_m2"] = df["raw_class_id"].map(
        lambda cid: class_to_quantum.get(cid, {"m2": -1})["m2"]
    )
    df["raw_m3"] = df["raw_class_id"].map(
        lambda cid: class_to_quantum.get(cid, {"m3": -1})["m3"]
    )
    df["raw_r"] = df["raw_class_id"].map(
        lambda cid: class_to_quantum.get(cid, {"r": -1})["r"]
    )

    polyad_to_class_ids = build_polyad_class_map(df, margin_threshold=2.0)
    print(
        f"Polyad-to-class map built from MARVEL states. "
        f"{len(polyad_to_class_ids)} unique polyads. "
        f"Example: polyad 10 has {len(polyad_to_class_ids.get(10, []))} valid classes."
    )

    print("Applying Localized Hungarian Algorithm (per Isotope, J, Parity, Polyad)...")
    print("  MARVEL train/val states are pre-locked to ground-truth classes;")
    print("  solver runs only on Ca + MARVEL test states with remaining class slots.")
    optimal_class_indices = np.full(len(df), -1, dtype=int)

    grouped = df.groupby(["isotope_id", "J", "parity_encoded", "polyad_int"])

    for (iso_id, J_val, parity_val, polyad_val), group in tqdm(
        grouped, desc="Assigning Quantum States"
    ):
        valid_class_ids = polyad_to_class_ids.get(polyad_val, [])
        if not valid_class_ids:
            continue

        valid_class_ids_set = set(valid_class_ids)

        # ── Lock MARVEL train/val to ground-truth classes ─────────────────────
        # These states have known quantum numbers and must never compete with Ca
        # states for class slots. Pre-assigning them removes their classes from
        # the pool before the solver runs.
        marvel_locked = group[
            group["is_marvel"]
            & ~group["test_mask"]
            & group["combinatorial_class_id"].notna()
        ]
        locked_classes = set()
        for midx, mcls in zip(
            marvel_locked.index.values,
            marvel_locked["combinatorial_class_id"].values,
        ):
            mcls = int(mcls)
            if mcls in valid_class_ids_set:
                optimal_class_indices[midx] = mcls
                locked_classes.add(mcls)

        # ── Run solver on remaining states with remaining class slots ─────────
        free_class_ids = np.array(
            sorted(valid_class_ids_set - locked_classes), dtype=int
        )
        free_state_idx = group[~(group["is_marvel"] & ~group["test_mask"])].index.values

        if len(free_state_idx) == 0 or len(free_class_ids) == 0:
            continue

        # For overcrowded blocks (n_states > n_valid), scipy naturally returns
        # only n_valid assignments — the optimal subset. No manual truncation needed.
        block_probs = mean_probs[free_state_idx][:, free_class_ids].cpu().numpy().copy()
        cost_matrix = 1.0 - block_probs
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        optimal_class_indices[free_state_idx[row_ind]] = free_class_ids[col_ind]

    # --- Post-Hungarian: compute assigned-class margin ---
    # For conflicted nodes this reflects the solver's forced class, not the model's top-1
    valid_mask = optimal_class_indices >= 0
    clipped_indices = np.clip(optimal_class_indices, 0, all_logits_cpu.shape[1] - 1)

    assigned_logits = all_logits_cpu[np.arange(len(df)), clipped_indices]
    competitor_logits = all_logits_cpu.copy()
    competitor_logits[valid_mask, optimal_class_indices[valid_mask]] = -np.inf
    runner_up_logits = competitor_logits.max(axis=1)

    df["assigned_margin"] = np.where(
        valid_mask, assigned_logits - runner_up_logits, 0.0
    )

    # Softmax probability of the final assigned class
    assigned_probs_arr = mean_probs_cpu[np.arange(len(df)), clipped_indices]
    df["assigned_prob"] = np.where(valid_mask, assigned_probs_arr, 0.0)

    # Entropy of the full softmax distribution (lower = more confident)
    _eps = 1e-12
    df["entropy"] = -(mean_probs_cpu * np.log(mean_probs_cpu + _eps)).sum(axis=1)

    # --- Conflict root cause diagnostic ---
    duplicate_counts = []
    block_size_counts = []

    for (iso_id, J_val, parity_val, polyad_val), group in df.groupby(
        ["isotope_id", "J", "parity_encoded", "polyad_int"]
    ):
        idx = group.index.values
        top1_preds = raw_class_indices[idx]
        n_states = len(idx)
        n_unique_preds = len(np.unique(top1_preds))
        valid_class_ids = polyad_to_class_ids.get(polyad_val, [])
        n_valid = len(valid_class_ids)

        duplicates = n_states - n_unique_preds
        duplicate_counts.append(duplicates)
        block_size_counts.append((n_states, n_valid, duplicates))

    block_df = pd.DataFrame(
        block_size_counts, columns=["n_states", "n_valid_classes", "n_duplicates"]
    )
    print(f"\n--- Block Diagnostic ---")
    print(f"Total blocks: {len(block_df)}")
    print(
        f"Blocks where n_states > n_valid_classes: {(block_df['n_states'] > block_df['n_valid_classes']).sum()}"
    )
    print(
        f"Blocks with any duplicate top-1 predictions: {(block_df['n_duplicates'] > 0).sum()}"
    )
    print(f"Mean duplicates per block: {block_df['n_duplicates'].mean():.2f}")
    print(f"Total duplicate top-1 predictions: {block_df['n_duplicates'].sum()}")
    print(f"\nBlock size distribution:")
    print(block_df[["n_states", "n_valid_classes"]].describe().round(1))

    print("Decoding final combinatorial classes...")
    df["pred_class_id"] = optimal_class_indices
    df["pred_m1"] = df["pred_class_id"].map(
        lambda cid: class_to_quantum.get(cid, {"m1": -1})["m1"]
    )
    df["pred_m2"] = df["pred_class_id"].map(
        lambda cid: class_to_quantum.get(cid, {"m2": -1})["m2"]
    )
    df["pred_m3"] = df["pred_class_id"].map(
        lambda cid: class_to_quantum.get(cid, {"m3": -1})["m3"]
    )
    df["pred_r"] = df["pred_class_id"].map(
        lambda cid: class_to_quantum.get(cid, {"r": -1})["r"]
    )

    test_df = df[df["test_mask"] == True]
    if not test_df.empty:
        mae_m1 = abs(test_df["AFGL_m1"] - test_df["pred_m1"]).mean()
        mae_m2 = abs(test_df["AFGL_m2"] - test_df["pred_m2"]).mean()
        mae_m3 = abs(test_df["AFGL_m3"] - test_df["pred_m3"]).mean()
        mae_r = abs(test_df["AFGL_r"] - test_df["pred_r"]).mean()

        print(f"\nPhysical MAE on Test Set (Post-Hungarian):")
        print(f"m1 Error: {mae_m1:.4f}")
        print(f"m2 Error: {mae_m2:.4f}")
        print(f"m3 Error: {mae_m3:.4f}")
        print(f"r  Error: {mae_r:.4f}")

        perfect_matches = (
            (test_df["AFGL_m1"] == test_df["pred_m1"])
            & (test_df["AFGL_m2"] == test_df["pred_m2"])
            & (test_df["AFGL_m3"] == test_df["pred_m3"])
            & (test_df["AFGL_r"] == test_df["pred_r"])
        ).sum()
        print(
            f"Perfect 4-QN Match Accuracy: {(perfect_matches / len(test_df)) * 100:.2f}%"
        )

    # --- Conflict diagnostic ---
    df["hungarian_conflict"] = df["raw_class_id"] != df["pred_class_id"]
    conflict_df = df[df["hungarian_conflict"] == True]
    correct_after_conflict = (
        (conflict_df["AFGL_m1"] == conflict_df["pred_m1"])
        & (conflict_df["is_marvel"] == True)
    ).sum() / (conflict_df["is_marvel"].sum() + 1e-9)
    print(
        f"\nHungarian Algorithm caused class changes for {df['hungarian_conflict'].sum()} nodes."
    )
    print(
        f"Accuracy on Hungarian-conflicted MARVEL nodes: {correct_after_conflict:.3f}"
    )

    # Rescale features back to original physical units
    df[FEATURE_COLS] = scaler.inverse_transform(df[FEATURE_COLS])

    unassigned = (optimal_class_indices == -1).sum()
    print(
        f"Unassigned states (physically overcrowded or no valid classes): {unassigned:,}"
    )
    print(f"Assignment rate: {(optimal_class_indices >= 0).sum() / len(df) * 100:.1f}%")

    output_path = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved final assignments to {output_path}")

    return df


def main():
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}")

    data, input_dim, num_isotopes, num_classes, df, mapping_df, scaler = (
        load_and_prepare_data()
    )

    model = CO2AssignmentGNN(
        input_dim=input_dim,
        num_isotopes=num_isotopes,
        num_classes=num_classes,
        hidden_dim=256,
        embed_dim=8,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = FocalLoss(gamma=2.0)

    print("\nInitializing GPU Mini-Batching (NeighborLoader)...")
    # Sample 10 neighbors per node, traversing 4 layers deep
    train_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        batch_size=2048,
        input_nodes=data.train_mask,
        shuffle=True,
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        batch_size=2048,
        input_nodes=data.val_mask,
        shuffle=False,
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        batch_size=2048,
        input_nodes=data.test_mask,
        shuffle=False,
    )

    print("Training Deep Residual GNN via Mini-Batches...")
    epochs = 100

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.iso_idx)
            loss = criterion(out[: batch.batch_size], batch.y[: batch.batch_size])

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.batch_size

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / data.train_mask.sum().item()
            val_acc = evaluate_batched(model, val_loader, device)
            print(
                f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val Top-1 Acc: {val_acc:.4f}"
            )

    test_acc = evaluate_batched(model, test_loader, device)
    print(f"\nTraining Complete. Base Test Top-1 Acc: {test_acc:.4f}")

    print("\nPreparing for global inference (Mini-Batched)...")

    # A loader with input_nodes=None automatically iterates over every single node in the graph safely
    inference_loader = NeighborLoader(
        data,
        num_neighbors=[15, 10],
        batch_size=2048,
        shuffle=False,
    )

    num_total_nodes = data.x.shape[0]
    final_df = evaluate_physical_assignment(
        model, inference_loader, device, num_total_nodes, df, mapping_df, scaler
    )
    plot_variance_validation(final_df)
    export_run_metrics(final_df)

    end = time.time()
    print(f"\nTotal Execution Time: {(end - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
