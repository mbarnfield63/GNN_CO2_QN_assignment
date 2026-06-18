# This script calculates the final evaluation metrics (F1-Macro, Precision, Recall, MAE)
# for the GNN + Solver pipeline. It evaluates the original MARVEL test set to track
# model performance across bootstrapping generations.

import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support


def calculate_final_metrics(df=None, preds_path="data/assigned_co2_predictions.csv"):
    # If called from run_pipeline.py with the unified dataset, it won't have the pred_m1 columns.
    # In that case, or if no df is provided, load the latest predictions CSV directly.
    if df is None or "pred_m1" not in df.columns:
        print(f"Loading latest predictions from {preds_path}...")
        if os.path.exists(preds_path):
            df = pd.read_csv(preds_path)
        else:
            print(
                f"[ERROR] Could not find {preds_path}. Cannot calculate final metrics."
            )
            return

    # 1. Filter for the Test Set
    # We only want to evaluate on MARVEL states that the model was NOT trained on.
    # This remains pure even during bootstrapping, as bootstrapping only affects inference states.
    test_df = df[df["is_marvel"] & df["test_mask"]].copy()

    if test_df.empty:
        print("Warning: Test set is empty or could not be found.")
        return

    print(f"Evaluating metrics on {len(test_df)} ground-truth MARVEL test states...")

    # 2. Define your ground truth and predicted column names
    targets = ["m1", "m2", "m3", "r"]
    true_cols = {"m1": "AFGL_m1", "m2": "AFGL_m2", "m3": "AFGL_m3", "r": "AFGL_r"}
    pred_cols = {"m1": "pred_m1", "m2": "pred_m2", "m3": "pred_m3", "r": "pred_r"}

    results = []

    # 3. Calculate Perfect 4-QN Match Accuracy
    is_correct = (
        (test_df["AFGL_m1"] == test_df["pred_m1"])
        & (test_df["AFGL_m2"] == test_df["pred_m2"])
        & (test_df["AFGL_m3"] == test_df["pred_m3"])
        & (test_df["AFGL_r"] == test_df["pred_r"])
    )
    perfect_match_acc = is_correct.mean() * 100

    # 4. Calculate metrics for each target independently
    for target in targets:
        y_true = test_df[true_cols[target]]
        y_pred = test_df[pred_cols[target]]

        # Calculate Mean Absolute Error (MAE)
        mae = np.abs(y_true - y_pred).mean()

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
                "MAE": round(mae, 4),
                "F1-Macro": round(f1, 4),
                "F1-Weighted": round(f1_wt, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
            }
        )

    # 5. Display the results formatted for LaTeX table inclusion
    results_df = pd.DataFrame(results)

    # Calculate the mean across all targets
    mean_row = pd.DataFrame(
        [
            {
                "Target": "MEAN",
                "MAE": round(results_df["MAE"].mean(), 4),
                "F1-Macro": round(results_df["F1-Macro"].mean(), 4),
                "F1-Weighted": round(results_df["F1-Weighted"].mean(), 4),
                "Precision": round(results_df["Precision"].mean(), 4),
                "Recall": round(results_df["Recall"].mean(), 4),
            }
        ]
    )

    results_df = pd.concat([results_df, mean_row], ignore_index=True)

    print("\n" + "=" * 50)
    print("=== FINAL GNN + SOLVER METRICS ===")
    print("=" * 50)
    print(f"Perfect 4-QN Match Accuracy: {perfect_match_acc:.2f}%\n")
    print(results_df.to_string(index=False))
    print("=" * 50 + "\n")


if __name__ == "__main__":
    calculate_final_metrics()
