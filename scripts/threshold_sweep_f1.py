"""
Sweep assigned_prob thresholds 0.55–0.95 and compute:
  - Ca states assigned at each threshold
  - 4-QN exact match accuracy on retained MARVEL test subset
  - MARVEL test retention (%)
  - Per-QN macro-F1 (m1, m2, m3, r) on retained subset
  - Mean of those 4 F1 scores
  - Drop in mean F1 vs. the 0.95 baseline
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
QN_PAIRS = [
    ("AFGL_m1", "pred_m1"),
    ("AFGL_m2", "pred_m2"),
    ("AFGL_m3", "pred_m3"),
    ("AFGL_r", "pred_r"),
]


def main():
    df = pd.read_csv(PREDICTIONS_PATH)

    test_df = df[df["test_mask"]].copy()
    ca_df = df[~df["is_marvel"]].copy()

    n_test_total = len(test_df)
    print(f"MARVEL test states: {n_test_total}")
    print(f"Ca states total:    {len(ca_df)}\n")

    rows = []
    for t in THRESHOLDS:
        retained = test_df[test_df["assigned_prob"] >= t]
        n_ret = len(retained)

        acc = (
            (
                (retained["AFGL_m1"] == retained["pred_m1"])
                & (retained["AFGL_m2"] == retained["pred_m2"])
                & (retained["AFGL_m3"] == retained["pred_m3"])
                & (retained["AFGL_r"] == retained["pred_r"])
            ).mean()
            * 100
            if n_ret > 0
            else float("nan")
        )

        ret_pct = n_ret / n_test_total * 100

        f1s = []
        for true_col, pred_col in QN_PAIRS:
            if (
                true_col in retained.columns
                and pred_col in retained.columns
                and n_ret > 0
            ):
                f1 = (
                    f1_score(
                        retained[true_col],
                        retained[pred_col],
                        average="macro",
                        zero_division=0,
                    )
                    * 100
                )
                f1s.append(f1)
        mean_f1 = np.mean(f1s) if f1s else float("nan")

        ca_count = (ca_df["assigned_prob"] >= t).sum()

        rows.append(
            {
                "threshold": t,
                "ca_assigned": ca_count,
                "marvel_acc": acc,
                "marvel_ret": ret_pct,
                "mean_f1": mean_f1,
            }
        )

    results = pd.DataFrame(rows)

    # Drop in mean F1 vs. 0.95 baseline
    baseline_f1 = results.loc[results["threshold"] == 0.95, "mean_f1"].values[0]
    results["f1_drop"] = (
        results["mean_f1"] - baseline_f1
    )  # negative = worse than baseline

    print(f"Baseline mean F1 at t=0.95: {baseline_f1:.3f}%\n")
    print(results.to_string(index=False, float_format="%.2f"))

    # LaTeX table rows
    print("\n--- LaTeX rows ---")
    for _, r in results.iterrows():
        t = r["threshold"]
        ca = int(r["ca_assigned"])
        acc = r["marvel_acc"]
        ret = r["marvel_ret"]
        f1 = r["mean_f1"]
        drop = r["f1_drop"]
        highlight = r"        \rowcolor{yellow!20}" if t == 0.75 else "       "
        drop_str = f"{drop:+.2f}" if not np.isnan(drop) else "---"
        print(f"{highlight} {t:.2f} & {ca:,} & {acc:.1f} & {ret:.1f} & {drop_str} \\\\")


if __name__ == "__main__":
    main()
