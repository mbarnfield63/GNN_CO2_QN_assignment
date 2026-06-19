"""
Validate that the GNN has learned the AFGL energy-ordering convention.

Within each (isotopologue, polyad, J, parity) group, the AFGL convention
requires r=1 to label the highest-energy state, r=2 the next highest, etc.
The Hungarian solver enforces one-to-one uniqueness but does NOT enforce this
ordering — whether r-labels align with the correct energy rank is a property
the model must learn from MARVEL supervision.

Two checks are performed:
1. MARVEL test states: pred_r accuracy against ground-truth AFGL_r, by polyad.
2. Ca states (no ground-truth labels): pairwise internal consistency — for each
   pair of assigned Ca states in the same (isotopologue, polyad, J, parity) group,
   check whether the lower pred_r state has the higher energy. This tests whether
   the model applied the AFGL energy-ordering convention to unlabelled states.

Outputs
-------
- Console: overall and per-polyad statistics for both checks
- data/figures/r_ordering_validation.png: dual-panel figure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import combinations
from plotting import thesis_params, HARVEST_THRESHOLD

DATA_DIR = "data"
PREDICTIONS_PATH = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")

mpl.rcParams.update(thesis_params)
os.makedirs(FIGURES_DIR, exist_ok=True)

GROUP_COLS = ["isotope_id", "polyad_int", "J", "parity_encoded"]


def pairwise_ordering_correct(energies, r_vals):
    """
    For every pair (i, j) with different energy and r, check if
    lower r → higher energy (AFGL convention).
    Returns (n_correct, n_pairs).
    """
    n_correct = n_pairs = 0
    for i, j in combinations(range(len(energies)), 2):
        ei, ri = energies[i], r_vals[i]
        ej, rj = energies[j], r_vals[j]
        if ei == ej or ri == rj:
            continue
        n_pairs += 1
        if (ei > ej) == (ri < rj):
            n_correct += 1
    return n_correct, n_pairs


def check_ca_consistency(df):
    """Pairwise r-ordering consistency for assigned Ca states."""
    ca = df[
        ~df["is_marvel"]
        & (df["pred_class_id"] != -1)
        & (df["assigned_prob"] >= HARVEST_THRESHOLD)
    ].copy()

    # Only consistent states (AFGL polyad = TROVE polyad)
    ca["afgl_polyad"] = 2 * ca["pred_m1"] + ca["pred_m2"] + 3 * ca["pred_m3"]
    ca = ca[ca["afgl_polyad"] == ca["polyad_int"]]

    # Groups with >= 2 states
    sizes = ca.groupby(GROUP_COLS).size()
    multi = ca.set_index(GROUP_COLS).index.isin(sizes[sizes >= 2].index)
    ca_multi = ca[multi].copy()

    print(f"Ca states in multi-state groups: {len(ca_multi):,}")

    records = []
    for keys, grp in ca_multi.groupby(GROUP_COLS):
        nc, np_ = pairwise_ordering_correct(
            grp["energy"].values, grp["pred_r"].values
        )
        if np_ > 0:
            records.append({"polyad": keys[1], "n_correct": nc, "n_pairs": np_})

    pairs_df = pd.DataFrame(records)
    total_c = pairs_df["n_correct"].sum()
    total_p = pairs_df["n_pairs"].sum()
    overall = total_c / total_p * 100
    print(f"Ca pairwise r-ordering accuracy: {overall:.1f}%  ({total_c:,}/{total_p:,} pairs)")

    by_polyad = (
        pairs_df.groupby("polyad")
        .agg(n_correct=("n_correct", "sum"), n_pairs=("n_pairs", "sum"))
        .reset_index()
    )
    by_polyad["accuracy"] = by_polyad["n_correct"] / by_polyad["n_pairs"] * 100
    return by_polyad, overall


def check_marvel_r_accuracy(df):
    """Per-polyad r-label accuracy on the MARVEL test set."""
    test = df[
        df["test_mask"]
        & (df["pred_class_id"] != -1)
    ].copy()

    test["afgl_polyad"] = 2 * test["pred_m1"] + test["pred_m2"] + 3 * test["pred_m3"]
    test = test[test["afgl_polyad"] == test["polyad_int"]]
    test["r_correct"] = test["pred_r"] == test["AFGL_r"]

    overall = test["r_correct"].mean() * 100
    print(f"\nMARVEL test r-label accuracy (consistent states): {overall:.1f}%  ({test['r_correct'].sum():,}/{len(test):,})")

    by_polyad = (
        test.groupby("polyad_int")["r_correct"]
        .agg(accuracy="mean", n_states="count")
        .reset_index()
    )
    by_polyad.columns = ["polyad", "accuracy", "n_states"]
    by_polyad["accuracy"] *= 100
    return by_polyad, overall


def plot(ca_by_polyad, ca_overall, marvel_by_polyad, marvel_overall):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), sharex=False)

    # ── Panel A: MARVEL test r-label accuracy by polyad ──────────────────────
    ax_top.bar(
        marvel_by_polyad["polyad"],
        marvel_by_polyad["accuracy"],
        color="#e67e22",
        alpha=0.8,
        label=f"MARVEL test r accuracy (overall {marvel_overall:.1f}%)",
    )
    ax_top.axhline(marvel_overall, color="#e67e22", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_top.set_ylabel("r-label accuracy (%)")
    ax_top.set_ylim(0, 105)
    ax_top.set_title("(a) MARVEL test set: r-label accuracy by polyad")
    ax_top.legend(fontsize=10)
    ax_top.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_top.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Panel B: Ca pairwise ordering consistency ─────────────────────────────
    ax2 = ax_bot.twinx()
    ax2.bar(
        ca_by_polyad["polyad"],
        ca_by_polyad["n_pairs"],
        color="#95a5a6",
        alpha=0.35,
        label="Pairs (right axis)",
        zorder=1,
    )
    ax_bot.plot(
        ca_by_polyad["polyad"],
        ca_by_polyad["accuracy"],
        marker="o",
        color="#2980b9",
        linewidth=2,
        markersize=6,
        label=f"Ca pairwise ordering accuracy (overall {ca_overall:.1f}%)",
        zorder=2,
    )
    ax_bot.axhline(ca_overall, color="#2980b9", linestyle="--", linewidth=1.2, alpha=0.7)

    ax_bot.set_xlabel("Polyad Number $P = 2m_1 + m_2 + 3m_3$")
    ax_bot.set_ylabel("Pairwise r-ordering accuracy (%)", color="#2980b9")
    ax_bot.set_ylim(0, 105)
    ax_bot.tick_params(axis="y", labelcolor="#2980b9")
    ax2.set_ylabel("Number of Ca comparison pairs", color="#95a5a6")
    ax2.tick_params(axis="y", labelcolor="#95a5a6")
    ax_bot.set_title("(b) Ca states: pairwise energy-ordering consistency by polyad")

    lines1, labels1 = ax_bot.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_bot.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)
    ax_bot.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_bot.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "r_ordering_validation.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save_path}")


def main():
    print(f"Loading predictions from {PREDICTIONS_PATH}...")
    df = pd.read_csv(PREDICTIONS_PATH)

    ca_by_polyad, ca_overall = check_ca_consistency(df)
    marvel_by_polyad, marvel_overall = check_marvel_r_accuracy(df)
    plot(ca_by_polyad, ca_overall, marvel_by_polyad, marvel_overall)


if __name__ == "__main__":
    main()
