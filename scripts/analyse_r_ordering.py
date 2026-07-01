"""
MARVEL test-set r-label accuracy by polyad.

Within each (isotopologue, polyad, J, parity) group the AFGL convention
assigns r=1 to the highest-energy state, r=2 to the next highest, etc.
The Hungarian solver enforces uniqueness but not this ordering; r values
of all assigned Ca states are subsequently corrected by energy sort in
bootstrap.py.  This script measures how often the GNN's raw r prediction
already matched ground truth on the MARVEL test set before that correction,
broken down by polyad — quantifying how much work the post-processing step
actually had to do.

Output
------
- data/figures/r_ordering_validation.png
"""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from plotting import thesis_params

DATA_DIR = "data"
PREDICTIONS_PATH = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")

mpl.rcParams.update(thesis_params)
os.makedirs(FIGURES_DIR, exist_ok=True)


def check_marvel_r_accuracy(df):
    """Per-polyad r-label accuracy on the MARVEL test set."""
    test = df[df["test_mask"] & (df["pred_class_id"] != -1)].copy()

    # Restrict to states where the predicted AFGL polyad matches the TROVE polyad
    test["pred_polyad"] = 2 * test["pred_m1"] + test["pred_m2"] + 3 * test["pred_m3"]
    test = test[test["pred_polyad"] == test["polyad_int"]]

    test["r_correct"] = test["pred_r"] == test["AFGL_r"]

    n_correct = int(test["r_correct"].sum())
    n_total = len(test)
    overall = n_correct / n_total * 100

    print(
        f"MARVEL test r-label accuracy (consistent polyad states): "
        f"{overall:.1f}%  ({n_correct:,}/{n_total:,})"
    )

    by_polyad = (
        test.groupby("polyad_int")["r_correct"]
        .agg(accuracy="mean", n_states="count")
        .reset_index()
        .rename(columns={"polyad_int": "polyad"})
    )
    by_polyad["accuracy"] *= 100

    print("\nPer-polyad breakdown:")
    print(by_polyad.to_string(index=False))

    return by_polyad, overall, n_correct, n_total


def plot(by_polyad, overall, n_correct, n_total):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        by_polyad["polyad"],
        by_polyad["accuracy"],
        color="#e67e22",
        alpha=0.8,
        label=f"r accuracy: {overall:.1f}%  ({n_correct:,}/{n_total:,} states)",
    )
    ax.axhline(overall, color="#e67e22", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("Polyad Number $P = 2m_1 + m_2 + 3m_3$")
    ax.set_ylabel("r-label accuracy (%)")
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "r_ordering_validation.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save_path}")


def main():
    print(f"Loading predictions from {PREDICTIONS_PATH}...")
    df = pd.read_csv(PREDICTIONS_PATH)
    by_polyad, overall, n_correct, n_total = check_marvel_r_accuracy(df)
    plot(by_polyad, overall, n_correct, n_total)


if __name__ == "__main__":
    main()
