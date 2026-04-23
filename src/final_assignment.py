import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from train import load_and_prepare_data, evaluate_batched
from model import CO2AssignmentGNN
from assignment import print_assignment_summary

DATA_DIR = "data"
DUMMY_PENALTY = (
    0.85  # States with <15% confidence will be trashed unless physically forced
)


def evaluate_physical_assignment_relaxed(
    model, loader, device, num_nodes, df, mapping_df, scaler
):
    """Enforces constraints with a dummy 'trash can' class for ghost states."""
    model.eval()

    print("\nCalculating epistemic uncertainty via MC Dropout...")
    mean_probs, variance = model.mc_dropout_predict(
        loader, device, num_nodes, num_samples=30
    )

    print("Applying Relaxed Localized Hungarian Algorithm (per Isotope, J, Parity)...")
    optimal_class_indices = np.full(len(df), -1, dtype=int)

    grouped = df.groupby(["isotope_id", "J", "parity_encoded"])

    for _, group in tqdm(grouped, desc="Assigning Quantum States"):
        idx = group.index.values
        block_probs = mean_probs[idx].cpu().numpy().copy()
        cost_matrix = 1.0 - block_probs

        N, M = cost_matrix.shape

        # Create a "Trash Can" matrix. N dummy classes for N theoretical states.
        dummy_matrix = np.full((N, N), DUMMY_PENALTY)

        # Append the trash can to the physical cost matrix
        relaxed_cost_matrix = np.hstack((cost_matrix, dummy_matrix))

        # Solve the relaxed assignment
        row_ind, col_ind = linear_sum_assignment(relaxed_cost_matrix)

        # Only lock in assignments that went to real physical classes (index < M)
        valid_mask = col_ind < M

        valid_rows = row_ind[valid_mask]
        valid_cols = col_ind[valid_mask]
        optimal_class_indices[idx[valid_rows]] = valid_cols

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

    # Safely assign variance (use 1.0 for unassigned/dummy states)
    df["assignment_variance"] = 1.0
    valid_indices = optimal_class_indices != -1
    if valid_indices.sum() > 0:
        df.loc[valid_indices, "assignment_variance"] = (
            variance[
                np.arange(len(df))[valid_indices], optimal_class_indices[valid_indices]
            ]
            .cpu()
            .numpy()
        )

    test_df = df[df["test_mask"] == True]
    if not test_df.empty:
        mae_m1 = abs(test_df["AFGL_m1"] - test_df["pred_m1"]).mean()
        mae_m2 = abs(test_df["AFGL_m2"] - test_df["pred_m2"]).mean()
        mae_m3 = abs(test_df["AFGL_m3"] - test_df["pred_m3"]).mean()
        mae_r = abs(test_df["AFGL_r"] - test_df["pred_r"]).mean()

        print(f"\nPhysical MAE on Test Set (Post-Hungarian):")
        print(f"m1 Error: {mae_m1:.4f} | m2 Error: {mae_m2:.4f}")
        print(f"m3 Error: {mae_m3:.4f} | r  Error: {mae_r:.4f}")

        perfect_matches = (
            (test_df["AFGL_m1"] == test_df["pred_m1"])
            & (test_df["AFGL_m2"] == test_df["pred_m2"])
            & (test_df["AFGL_m3"] == test_df["pred_m3"])
            & (test_df["AFGL_r"] == test_df["pred_r"])
        ).sum()
        print(
            f"Perfect 4-QN Match Accuracy: {(perfect_matches / len(test_df)) * 100:.2f}%"
        )

    print("\nRescaling features back to original physical units...")
    from train import FEATURE_COLS

    df[FEATURE_COLS] = scaler.inverse_transform(df[FEATURE_COLS])

    output_path = os.path.join(DATA_DIR, "final_relaxed_assignments.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved finalized assignments to {output_path}")

    return df


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware target: {device}")

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

    print("\nInitializing GPU Mini-Batching for Final Run...")
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10, 10, 10],
        batch_size=2048,
        input_nodes=data.train_mask,
        shuffle=True,
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10, 10, 10],
        batch_size=2048,
        input_nodes=data.test_mask,
        shuffle=False,
    )

    print(f"Training Deep Residual GNN on fully-bootstrapped Generation 5 data...")
    epochs = 200

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.iso_idx)
            loss = criterion(out[: batch.batch_size], batch.y[: batch.batch_size])
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} complete.")

    test_acc = evaluate_batched(model, test_loader, device)
    print(f"\nFinal Training Complete. Base Test Top-1 Acc: {test_acc:.4f}")

    print("\nPreparing for global inference (Mini-Batched)...")
    inference_loader = NeighborLoader(
        data, num_neighbors=[10, 10, 10, 10], batch_size=2048, shuffle=False
    )

    num_total_nodes = data.x.shape[0]
    final_df = evaluate_physical_assignment_relaxed(
        model, inference_loader, device, num_total_nodes, df, mapping_df, scaler
    )

    print_assignment_summary(final_df, variance_threshold=0.05)
    print(f"\nTotal Execution Time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
