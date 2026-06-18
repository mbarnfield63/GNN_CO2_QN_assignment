import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from train import load_and_prepare_data, evaluate_batched, train_model, decode_qn_cols, FEATURE_COLS
from model import CO2AssignmentGNN

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
    mean_probs, variance, mean_sample_entropy = model.mc_dropout_predict(
        loader, device, num_nodes, num_samples=30
    )

    print("Decoding combinatorial classes for mapping...")
    class_to_quantum = mapping_df.set_index("class_id")[
        ["m1", "m2", "m3", "r"]
    ].to_dict("index")

    # Extract RAW predictions before the solver touches anything
    print("Capturing raw neural network predictions...")
    raw_class_indices = mean_probs.argmax(dim=1).cpu().numpy()

    df["raw_class_id"] = raw_class_indices
    decode_qn_cols(df, "raw_class_id", class_to_quantum, "raw")

    # Extract the variance specifically for the raw predicted class
    df["raw_variance"] = variance[np.arange(len(df)), raw_class_indices].cpu().numpy()

    # Predictive entropy H(mean_probs) = -sum_c p̄_c log(p̄_c)
    mc_pred_ent = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1).numpy()
    df["mc_predictive_entropy"] = mc_pred_ent

    # BALD (epistemic uncertainty) = H(mean) - mean H(samples)
    df["mc_bald"] = mc_pred_ent - mean_sample_entropy.numpy()

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

    df["pred_class_id"] = optimal_class_indices
    decode_qn_cols(df, "pred_class_id", class_to_quantum, "pred")

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

    test_df = df[df["test_mask"]]
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

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
    train_model(model, train_loader, device, epochs=100, criterion=criterion, optimizer=optimizer, print_every=20)

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

    # Isolate the inference states (!Ma / Ca)
    inference_df = final_df[~final_df["is_marvel"]]
    total_inference = len(inference_df)
    variance_threshold = 0.05  # Define a threshold for high confidence

    if total_inference == 0:
        print("\n=== ASSIGNMENT SUMMARY ===")
        print("No inference (!Ma) states found in the dataset to summarize.")
        return

    # Count how many received a valid mapping from the Hungarian solver
    assigned_df = inference_df[inference_df["pred_class_id"] != -1]
    total_assigned = len(assigned_df)
    assigned_pct = (total_assigned / total_inference) * 100

    # Count how many are highly confident based on the variance threshold
    confident_df = assigned_df[assigned_df["assignment_variance"] <= variance_threshold]
    total_confident = len(confident_df)
    confident_pct = (total_confident / total_inference) * 100

    print("\n" + "=" * 45)
    print("=== FINAL ASSIGNMENT SUMMARY ===")
    print("=" * 45)
    print(f"Total Available Inference (!Ma) States: {total_inference:,}")
    print(
        f"Total States Mapped by Solver:          {total_assigned:,} ({assigned_pct:.2f}%)"
    )
    print(
        f"Highly Confident New Assignments:       {total_confident:,} ({confident_pct:.2f}%)"
    )
    print(f"  *(Confidence defined as variance <= {variance_threshold})*")
    print("=" * 45 + "\n")
    print(f"\nTotal Execution Time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
