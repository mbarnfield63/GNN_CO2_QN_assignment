import os
import pandas as pd
import numpy as np
from plotting import (
    plot_per_isotopologue,
    plot_energy_distribution,
    plot_j_energy_yield_hexbin,
    plot_polyad_ladders,
    plot_bootstrapping_histogram,
)


DATA_DIR = "data"
FINAL_DATA_PATH = os.path.join(DATA_DIR, "final_relaxed_assignments.csv")
PLOT_DIR = os.path.join(DATA_DIR, "figures")

# Labels
LBL_MARVEL = "MARVEL (Ground Truth)"
LBL_CONFIDENT = "ML Confident (Var ≤ 0.05)"
LBL_CONSTRAINED = "Physically Constrained (Var > 0.05)"
LBL_UNASSIGNED = "Unassigned"

os.makedirs(PLOT_DIR, exist_ok=True)


def load_and_categorize_data(path=FINAL_DATA_PATH):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)

    conditions = [
        (df["is_marvel"] == True),
        (df["is_marvel"] == False)
        & (df["pred_class_id"] != -1)
        & (df["assignment_variance"] <= 0.05),
        (df["is_marvel"] == False)
        & (df["pred_class_id"] != -1)
        & (df["assignment_variance"] > 0.05),
        (df["is_marvel"] == False) & (df["pred_class_id"] == -1),
    ]

    choices = [LBL_MARVEL, LBL_CONFIDENT, LBL_CONSTRAINED, LBL_UNASSIGNED]
    df["Assignment_Category"] = np.select(conditions, choices, default="Unknown")
    df["Assignment_Category"] = pd.Categorical(
        df["Assignment_Category"], categories=choices, ordered=True
    )
    return df


if __name__ == "__main__":
    print("Initializing Publication Figure Generation...")
    df = load_and_categorize_data()
    plot_per_isotopologue(df)
    plot_energy_distribution(df, bin_size=1000)
    plot_polyad_ladders(df)
    print("All figures successfully saved.")
