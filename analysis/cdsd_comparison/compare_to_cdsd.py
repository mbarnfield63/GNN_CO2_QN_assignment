"""
Scores the isolated CDSD side-run (run_side_pipeline.py) against the true
CDSD-2024-PI AFGL labels stashed by prepare_cdsd_data.py.

Splits results into:
  - "reachable": CDSD states whose true (m1, m2, m3, r) combination exists in
    this run's class_mapping.csv (built only from MARVEL data), i.e. states
    the model could theoretically ever get right.
  - the full set, as a footnote -- most of it is structurally unreachable
    (see README.md), so the full-set percentage alone would understate the
    model's actual performance.

Run after run_side_pipeline.py has completed:
    uv run analysis/cdsd_comparison/compare_to_cdsd.py
"""

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SIDE_DATA_DIR = os.path.join(HERE, "data")

TRUE_LABELS_PATH = os.path.join(HERE, "cdsd_true_labels.csv")
PREDICTIONS_PATH = os.path.join(SIDE_DATA_DIR, "assigned_co2_predictions.csv")
CLASS_MAPPING_PATH = os.path.join(SIDE_DATA_DIR, "class_mapping.csv")

ENERGY_DECIMALS = 4  # absorbs StandardScaler round-trip float noise


def load_and_join():
    true_df = pd.read_csv(TRUE_LABELS_PATH)
    true_df["parity_encoded"] = true_df["parity"].map({"e": 0, "f": 1})
    true_df["energy_key"] = true_df["energy"].round(ENERGY_DECIMALS)
    # Drop the raw energy column before merging: preds carries the same value
    # (both trace back to E_Ca) and fit_groups() downstream expects a single
    # unambiguous "energy" column, not an energy_true/energy_pred split.
    true_df = true_df.drop(columns=["energy"])

    preds = pd.read_csv(PREDICTIONS_PATH)
    preds = preds[preds["isotope_id"].astype(str) == "626"].copy()
    preds["energy_key"] = preds["energy"].round(ENERGY_DECIMALS)

    merged = true_df.merge(
        preds,
        on=["energy_key", "J", "parity_encoded"],
        how="left",
        suffixes=("_true", "_pred"),
    )

    n_unmatched = merged["pred_class_id"].isna().sum()
    if n_unmatched:
        print(
            f"WARNING: {n_unmatched} of {len(merged)} CDSD states did not join to "
            "any node in the side-run predictions (check energy tolerance / cutoff)."
        )
    return merged.dropna(subset=["pred_class_id"])


def mark_reachable(df):
    class_map = pd.read_csv(CLASS_MAPPING_PATH)
    reachable_combos = set(
        zip(class_map["m1"], class_map["m2"], class_map["m3"], class_map["r"])
    )
    combo = list(zip(df["m1_true"], df["m2_true"], df["m3_true"], df["r_true"]))
    df = df.copy()
    df["reachable"] = [c in reachable_combos for c in combo]
    return df


def report(df, label):
    n_total = len(df)
    assigned = df[df["pred_class_id"] >= 0]
    n_assigned = len(assigned)

    per_qn = {}
    for qn, true_col, pred_col in [
        ("m1", "m1_true", "pred_m1"),
        ("m2", "m2_true", "pred_m2"),
        ("m3", "m3_true", "pred_m3"),
        ("r", "r_true", "pred_r"),
    ]:
        match = (assigned[true_col] == assigned[pred_col]).mean() * 100 if n_assigned else float("nan")
        per_qn[qn] = match

    four_qn_match = (
        (assigned["m1_true"] == assigned["pred_m1"])
        & (assigned["m2_true"] == assigned["pred_m2"])
        & (assigned["m3_true"] == assigned["pred_m3"])
        & (assigned["r_true"] == assigned["pred_r"])
    )
    four_qn_pct = four_qn_match.mean() * 100 if n_assigned else float("nan")

    print(f"\n--- {label} (N={n_total:,}) ---")
    print(f"Assigned by solver: {n_assigned:,} / {n_total:,} ({n_assigned/n_total*100:.1f}%)")
    print("Per-QN agreement (of assigned states):")
    for qn, pct in per_qn.items():
        print(f"  {qn}: {pct:.2f}%")
    print(f"4-QN exact match (of assigned states): {four_qn_pct:.2f}%")
    print(f"4-QN exact match (of all {label.lower()} states, unassigned counted as wrong): "
          f"{four_qn_match.sum()/n_total*100:.2f}%")


def main():
    merged = load_and_join()
    merged = mark_reachable(merged)

    n_reachable = merged["reachable"].sum()
    print("=" * 64)
    print("CDSD-2024-PI Comparison (isotopologue 626)")
    print("=" * 64)
    print(f"Total CDSD states matched to side-run: {len(merged):,}")
    print(
        f"Reachable (combo exists in this run's class_mapping.csv): "
        f"{n_reachable:,} ({n_reachable/len(merged)*100:.1f}%)"
    )
    print(
        f"Structurally unreachable (combo absent from MARVEL-derived classes): "
        f"{len(merged)-n_reachable:,} ({(1-n_reachable/len(merged))*100:.1f}%)"
    )

    report(merged[merged["reachable"]], "Reachable subset")
    report(merged, "Full CDSD set (footnote)")


if __name__ == "__main__":
    main()
