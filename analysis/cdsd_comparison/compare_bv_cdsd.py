"""
B_v (rotational constant) comparison: CDSD-2024-PI vs GNN, isotopologue 626.

Reuses fit_groups() from scripts/analyse_spectroscopic_constants.py unmodified
-- the exact same fitting methodology already used for the MARVEL vs GNN
comparison in the paper (Sec. "Physical Validation: Spectroscopic Constants"),
just with CDSD's independently-assigned AFGL labels as the reference instead
of MARVEL's.

Run after run_side_pipeline.py and compare_to_cdsd.py:
    uv run analysis/cdsd_comparison/compare_bv_cdsd.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(HERE, "..", "..", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import pandas as pd
from analyse_spectroscopic_constants import (  # noqa: E402
    fit_groups,
    CONFIDENT_MARGIN,
    EXCLUDE_POLYADS,
    MAX_POLYAD,
    RMSE_CLEAN,
)
from compare_to_cdsd import load_and_join, mark_reachable, PREDICTIONS_PATH  # noqa: E402

OUT_PATH = os.path.join(HERE, "spectroscopic_constants_cdsd_vs_gnn.csv")


def build_cdsd_fit_input():
    """CDSD-true-labelled states, restricted to the reachable subset (see
    compare_to_cdsd.py) -- fitting bands from structurally-unreachable states
    would compare against a target the model could never have predicted."""
    merged = mark_reachable(load_and_join())
    return merged[merged["reachable"]].copy()


def build_gnn_fit_input():
    """Confident 626 GNN assignments, same filter as the MARVEL comparison
    (assigned_margin >= CONFIDENT_MARGIN, excluding polyads 15/18, polyad<=10)."""
    preds = pd.read_csv(PREDICTIONS_PATH)
    preds = preds[preds["isotope_id"].astype(str) == "626"].copy()
    return preds[
        (~preds["is_marvel"])
        & (preds["pred_class_id"] != -1)
        & (preds["assigned_margin"] >= CONFIDENT_MARGIN)
        & (~preds["polyad_int"].isin(EXCLUDE_POLYADS))
        & (preds["polyad_int"] <= MAX_POLYAD)
    ].copy()


def main():
    df_cdsd = fit_groups(
        build_cdsd_fit_input(), "m1_true", "m2_true", "m3_true", "r_true", "CDSD"
    )
    df_gnn = fit_groups(
        build_gnn_fit_input(), "pred_m1", "pred_m2", "pred_m3", "pred_r", "GNN"
    )
    print(f"CDSD bands fitted (626, reachable subset): {len(df_cdsd):,}")
    print(f"GNN bands fitted (626, margin>={CONFIDENT_MARGIN}):        {len(df_gnn):,}")

    gnn_clean = df_gnn[df_gnn["rmse"] < RMSE_CLEAN]
    merge_keys = ["isotope_id", "m1", "m2", "m3", "r", "parity"]
    merged = df_cdsd.merge(gnn_clean, on=merge_keys, suffixes=("_cdsd", "_gnn"))

    if merged.empty:
        print("No overlapping clean bands between CDSD and GNN fits.")
        return

    merged["delta_bv"] = (merged["B_v_gnn"] - merged["B_v_cdsd"]).abs()
    merged.to_csv(OUT_PATH, index=False)

    print(f"\nOverlapping clean bands: {len(merged):,}")
    print(f"Median |B_v(GNN) - B_v(CDSD)|: {merged['delta_bv'].median():.6f} cm-1")
    print(f"Max    |B_v(GNN) - B_v(CDSD)|: {merged['delta_bv'].max():.6f} cm-1")
    print(
        f"Bands agreeing to < 1e-3 cm-1: {(merged['delta_bv'] < 1e-3).sum():,} / {len(merged):,}"
    )
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
