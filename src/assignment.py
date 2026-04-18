import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def strict_1_to_1_assignment(probability_matrix: np.ndarray) -> np.ndarray:
    """
    Takes the (N_nodes, N_classes) probability matrix from the GNN.
    Returns an optimal 1-to-1 mapping array of length N_nodes.
    """
    # The Hungarian algorithm minimizes cost, so we pass negative probabilities
    # (or 1 - probabilities) to maximize the probability sum.
    cost_matrix = 1.0 - probability_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # row_ind corresponds to node indices, col_ind to assigned class IDs
    return col_ind


def print_assignment_summary(df: pd.DataFrame, variance_threshold: float = 0.05):
    """
    Calculates and prints summary statistics for the newly assigned non-MARVEL states.
    """
    # Isolate the inference states (!Ma / Ca)
    inference_df = df[df["is_marvel"] == False]
    total_inference = len(inference_df)

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
