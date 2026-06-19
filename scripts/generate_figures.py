import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import pandas as pd
import numpy as np
from plotting import (
    plot_per_isotopologue,
    plot_energy_distribution,
    plot_polyad_ladders,
    plot_assignment_rate_by_energy,
    HARVEST_THRESHOLD,
    LBL_MARVEL,
    LBL_CONFIDENT,
    LBL_CONSTRAINED,
    LBL_UNASSIGNED,
)

DATA_DIR = "data"
FINAL_DATA_PATH = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")
PLOT_DIR = os.path.join(DATA_DIR, "figures")

CONFIDENT_THRESHOLD = HARVEST_THRESHOLD  # single source of truth in plotting.py

os.makedirs(PLOT_DIR, exist_ok=True)


def load_and_categorize_data(path=FINAL_DATA_PATH):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)

    if "assigned_prob" not in df.columns:
        raise KeyError(
            "'assigned_prob' column not found. Re-run train.py to regenerate the predictions CSV."
        )

    conditions = [
        df["is_marvel"],
        ~df["is_marvel"]
        & (df["pred_class_id"] != -1)
        & (df["assigned_prob"] >= CONFIDENT_THRESHOLD),
        ~df["is_marvel"]
        & (df["pred_class_id"] != -1)
        & (df["assigned_prob"] < CONFIDENT_THRESHOLD),
        ~df["is_marvel"] & (df["pred_class_id"] == -1),
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
    plot_assignment_rate_by_energy(df)
    print("All figures successfully saved.")
