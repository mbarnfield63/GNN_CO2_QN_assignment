# This script calculates the final evaluation metrics (F1-Macro, Precision, Recall) for the GNN + Solver pipeline.
# It reads the final relaxed assignments CSV, filters for the test set, and compares the predicted quantum numbers against the ground truth AFGL values for the MARVEL states.
# The results are printed in a format suitable for inclusion in a LaTeX table.
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def calculate_final_metrics(df=None, csv_path="data/final_relaxed_assignments.csv"):
    if df is None:
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

    # 1. Filter for the Test Set
    # We only want to evaluate on MARVEL states that the model was NOT trained on.
    test_df = df[(df["is_marvel"] == True) & (df["test_mask"] == True)].copy()

    print(f"Evaluating metrics on {len(test_df)} ground truth states...")

    # 2. Define your ground truth and predicted column names
    # Adjust these string names if they differ in your CSV!
    targets = ["m1", "m2", "m3", "r"]

    true_cols = {"m1": "AFGL_m1", "m2": "AFGL_m2", "m3": "AFGL_m3", "r": "AFGL_r"}

    pred_cols = {"m1": "pred_m1", "m2": "pred_m2", "m3": "pred_m3", "r": "pred_r"}

    results = []

    # 3. Calculate metrics for each target independently
    for target in targets:
        y_true = test_df[true_cols[target]]
        y_pred = test_df[pred_cols[target]]

        # Calculate metrics using macro averaging (treats all classes equally, important for rare high QNs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        # Weighted average (Shows operational accuracy based on data volume)
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        results.append(
            {
                "Target": f"AFGL_{target}",
                "F1-Macro": round(f1, 4),
                "F1-Weighted": round(f1_wt, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
            }
        )

    # 4. Display the results formatted for LaTeX table
    results_df = pd.DataFrame(results)

    # Calculate the mean across all targets
    mean_row = pd.DataFrame(
        [
            {
                "Target": "MEAN",
                "F1-Macro": round(results_df["F1-Macro"].mean(), 4),
                "F1-Weighted": round(results_df["F1-Weighted"].mean(), 4),
                "Precision": round(results_df["Precision"].mean(), 4),
                "Recall": round(results_df["Recall"].mean(), 4),
            }
        ]
    )

    results_df = pd.concat([results_df, mean_row], ignore_index=True)

    print("\n=== Final GNN + Solver Metrics ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    calculate_final_metrics()
