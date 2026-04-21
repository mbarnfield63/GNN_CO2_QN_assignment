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
from assignment import print_assignment_summary
from graph_builder import build_pyg_graph
from model import CO2AssignmentGNN

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
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    if os.path.exists(GRAPH_CACHE_PATH):
        print("Loading cached PyG graph from disk...")
        data = torch.load(GRAPH_CACHE_PATH, weights_only=False)
    else:
        data = build_pyg_graph(df)
        print("Saving PyG graph to disk for future runs...")
        torch.save(data, GRAPH_CACHE_PATH)

    return data, len(FEATURE_COLS), num_isotopes, num_classes, df, mapping_df, scaler


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


def evaluate_physical_assignment(
    model, loader, device, num_nodes, df, mapping_df, scaler
):
    """Enforces 1-to-1 physical constraints locally and decodes combinatorial classes."""
    model.eval()

    print("\nCalculating epistemic uncertainty via MC Dropout...")
    mean_probs, variance = model.mc_dropout_predict(
        loader, device, num_nodes, num_samples=30
    )

    print("Applying Localized Hungarian Algorithm (per Isotope, J, Parity)...")
    optimal_class_indices = np.full(len(df), -1, dtype=int)

    grouped = df.groupby(["isotope_id", "J", "parity_encoded"])

    for _, group in tqdm(grouped, desc="Assigning Quantum States"):
        idx = group.index.values

        # ADD .copy() here to silence the PyTorch UserWarning
        block_probs = mean_probs[idx].cpu().numpy().copy()

        cost_matrix = 1.0 - block_probs
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        optimal_class_indices[idx[row_ind]] = col_ind

    print("Decoding combinatorial classes...")
    class_to_quantum = mapping_df.set_index("class_id")[
        ["m1", "m2", "m3", "r"]
    ].to_dict("index")

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

    df["assignment_variance"] = (
        variance[np.arange(len(df)), optimal_class_indices].cpu().numpy()
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

    # Rescaling features back to original physical units
    df[FEATURE_COLS] = scaler.inverse_transform(df[FEATURE_COLS])

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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\nInitializing GPU Mini-Batching (NeighborLoader)...")
    # Sample 10 neighbors per node, traversing 4 layers deep
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10, 10, 10],
        batch_size=2048,
        input_nodes=data.train_mask,
        shuffle=True,
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10, 10, 10],
        batch_size=2048,
        input_nodes=data.val_mask,
        shuffle=False,
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10, 10, 10],
        batch_size=2048,
        input_nodes=data.test_mask,
        shuffle=False,
    )

    print("Training Deep Residual GNN via Mini-Batches...")
    epochs = 200

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
        num_neighbors=[10, 10, 10, 10],
        batch_size=2048,
        shuffle=False,
    )

    num_total_nodes = data.x.shape[0]
    final_df = evaluate_physical_assignment(
        model, inference_loader, device, num_total_nodes, df, mapping_df, scaler
    )

    print_assignment_summary(final_df, variance_threshold=0.05)

    end = time.time()
    print(f"\nTotal Execution Time: {(end - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
