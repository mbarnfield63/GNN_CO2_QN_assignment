import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
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


def evaluate_internal(model, data, mask):
    """Calculates internal Top-1 Accuracy."""
    model.eval()
    with torch.no_grad():
        if mask.sum() == 0:
            return 0.0
        out = model(data.x, data.edge_index, data.iso_idx)
        preds = out[mask].argmax(dim=1)
        correct = (preds == data.y[mask]).sum().item()
        return correct / mask.sum().item()


# Add 'scaler' to the arguments
def evaluate_physical_assignment(model, data, df, mapping_df, scaler):
    """Enforces 1-to-1 physical constraints locally and decodes combinatorial classes."""
    model.eval()

    print("\nCalculating epistemic uncertainty via MC Dropout...")
    mean_probs, variance = model.mc_dropout_predict(
        data.x, data.edge_index, data.iso_idx, num_samples=30
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
    data = data.to(device)

    model = CO2AssignmentGNN(
        input_dim=input_dim,
        num_isotopes=num_isotopes,
        num_classes=num_classes,
        hidden_dim=128,
        embed_dim=8,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining GNN Transductively...")
    epochs = 200

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.iso_idx)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            val_acc = evaluate_internal(model, data, data.val_mask)
            print(
                f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Top-1 Acc: {val_acc:.4f}"
            )

    test_acc = evaluate_internal(model, data, data.test_mask)
    print(f"\nTraining Complete. Base Test Top-1 Acc: {test_acc:.4f}")

    # Catch the fully assigned dataframe
    final_df = evaluate_physical_assignment(model, data, df, mapping_df, scaler)

    # Print the assignment callouts
    print_assignment_summary(final_df, variance_threshold=0.05)

    end = time.time()
    print(f"\nTotal Execution Time: {(end - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
