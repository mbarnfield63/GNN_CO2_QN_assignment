"""
Diagnostic script: compare all available uncertainty signals as predictors
of 4-QN correctness on the MARVEL test set.

Outputs
-------
- Console table: AUROC, KS statistic, Youden threshold, precision @ T=1.0
- data/figures/uncertainty_comparison.png: density + precision-retention plots
  for every signal side-by-side
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp

DATA_DIR = "data"
PREDICTIONS_PATH = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")
RELAXED_PATH = os.path.join(DATA_DIR, "final_relaxed_assignments.csv")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")
HARVEST_THRESHOLD = 0.75

mpl.rcParams.update({"font.size": 11, "legend.frameon": False})


# ── Signal metadata ───────────────────────────────────────────────────────────
# (column_name, display_label, higher_is_more_confident)
SIGNAL_DEFS = [
    ("assigned_prob", "Assigned\nSoftmax Prob", True),
    ("assigned_margin", "Assigned\nLogit Margin", True),
    ("logit_margin", "Raw\nLogit Margin", True),
    ("entropy", "Softmax\nEntropy", False),
    ("assignment_variance", "MC Dropout\nVariance", False),
    ("mc_predictive_entropy", "MC Predictive\nEntropy", False),
    ("mc_bald", "BALD\n(Epistemic Unc.)", False),
]


def _load_test_df():
    """Load and merge prediction CSVs; return MARVEL test rows with correctness flag."""
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"{PREDICTIONS_PATH} not found. Run train.py first.")

    df = pd.read_csv(PREDICTIONS_PATH)
    test_df = df[df["test_mask"]].copy()

    # Merge MC-dropout signals from the relaxed CSV if available
    if os.path.exists(RELAXED_PATH):
        relaxed = pd.read_csv(RELAXED_PATH)
        mc_cols = [
            c
            for c in [
                "assignment_variance",
                "raw_variance",
                "mc_predictive_entropy",
                "mc_bald",
            ]
            if c in relaxed.columns
        ]
        if mc_cols and "node_id" in relaxed.columns and "node_id" in test_df.columns:
            relaxed_sub = relaxed[["node_id"] + mc_cols]
            test_df = test_df.merge(relaxed_sub, on="node_id", how="left")
            print(f"Merged MC dropout columns from {RELAXED_PATH}: {mc_cols}")
        else:
            print(f"Skipping relaxed CSV merge (missing node_id or MC columns).")
    else:
        print(
            f"No relaxed assignments CSV found at {RELAXED_PATH}. Skipping MC dropout."
        )

    # Binary correctness (4-QN perfect match)
    test_df["is_correct"] = (
        (test_df["AFGL_m1"] == test_df["pred_m1"])
        & (test_df["AFGL_m2"] == test_df["pred_m2"])
        & (test_df["AFGL_m3"] == test_df["pred_m3"])
        & (test_df["AFGL_r"] == test_df["pred_r"])
    )

    n = len(test_df)
    nc = test_df["is_correct"].sum()
    print(f"\nMARVEL test set: {n} states | {nc} correct ({nc/n*100:.1f}%)")
    return test_df


def _precision_retention(values, correct, n_total, higher_is_confident):
    """Sweep thresholds and return (thresholds, precision, retention) arrays."""
    lo, hi = np.percentile(values, 1), np.percentile(values, 99)
    thresholds = np.linspace(lo, hi, 300)
    masks = (values[None, :] >= thresholds[:, None]) if higher_is_confident else (values[None, :] <= thresholds[:, None])
    counts = masks.sum(axis=1)
    n_valid = int(np.argmax(counts == 0)) if (counts == 0).any() else len(thresholds)
    thresholds, masks, counts = thresholds[:n_valid], masks[:n_valid], counts[:n_valid]
    prec = (masks * correct[None, :]).sum(axis=1) / np.maximum(counts, 1) * 100
    ret = counts / n_total * 100
    return thresholds, prec, ret


def _youden_threshold(values, correct, higher_is_confident):
    """Return the threshold maximising sensitivity + specificity - 1."""
    score = values if higher_is_confident else -values
    fpr, tpr, thresholds = roc_curve(correct, score)
    opt_score = thresholds[np.argmax(tpr - fpr)]
    return opt_score if higher_is_confident else -opt_score


def analyse(test_df):
    correct = test_df["is_correct"].values
    n_total = len(test_df)

    # Collect results
    rows = []
    available_signals = []
    auroc_by_col = {}

    for col, label, higher in SIGNAL_DEFS:
        if col not in test_df.columns:
            continue
        values = test_df[col].values

        # AUROC (flip for lower-is-confident signals)
        auroc = roc_auc_score(correct, values if higher else -values)
        auroc_by_col[col] = auroc

        # KS statistic
        ks_stat, _ = ks_2samp(values[correct], values[~correct])

        # Youden threshold
        opt_t = _youden_threshold(values, correct, higher)

        # Precision at HARVEST_THRESHOLD (only meaningful for margin-like signals)
        if higher:
            harvest_mask = values >= HARVEST_THRESHOLD
        else:
            harvest_mask = values <= HARVEST_THRESHOLD
        prec_at_harvest = (
            correct[harvest_mask].mean() * 100
            if harvest_mask.sum() > 0
            else float("nan")
        )
        ret_at_harvest = harvest_mask.sum() / n_total * 100

        rows.append(
            {
                "Signal": label.replace("\n", " "),
                "AUROC": f"{auroc:.3f}",
                "KS stat": f"{ks_stat:.3f}",
                "Youden T": f"{opt_t:.3f}",
                f"Prec@T={HARVEST_THRESHOLD}": f"{prec_at_harvest:.1f}%",
                f"Ret@T={HARVEST_THRESHOLD}": f"{ret_at_harvest:.1f}%",
            }
        )
        available_signals.append((col, label, higher, values))

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("=== UNCERTAINTY SIGNAL COMPARISON (MARVEL test set) ===")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("=" * 70 + "\n")

    # Per-QN accuracy breakdown
    qn_pairs = [
        ("AFGL_m1", "pred_m1"),
        ("AFGL_m2", "pred_m2"),
        ("AFGL_m3", "pred_m3"),
        ("AFGL_r", "pred_r"),
    ]
    qn_names = ["m1", "m2", "m3", "r"]
    if all(c in test_df.columns for pair in qn_pairs for c in pair):
        print("Per-QN accuracy on MARVEL test set:")
        for name, (true_col, pred_col) in zip(qn_names, qn_pairs):
            acc = (test_df[true_col] == test_df[pred_col]).mean() * 100
            print(f"  {name}: {acc:.2f}%")
        print()

    return available_signals, auroc_by_col


def plot_comparison(test_df, available_signals, auroc_by_col):
    correct = test_df["is_correct"].values

    # Exclude MC Dropout Variance; order by AUROC descending
    signals = [s for s in available_signals if s[0] != "assignment_variance"]
    signals.sort(key=lambda s: auroc_by_col.get(s[0], 0), reverse=True)
    n_sig = len(signals)

    if n_sig == 0:
        print("No signals available to plot.")
        return

    nrows = min(3, math.ceil(n_sig / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(9, 4 * nrows), squeeze=False)

    c_correct = "#2ecc71"
    c_incorrect = "#e74c3c"

    for col_idx, (col, label, higher, values) in enumerate(signals):
        ax = axes.flat[col_idx]
        auroc = auroc_by_col.get(col, float("nan"))

        bins = np.linspace(np.percentile(values, 0.5), np.percentile(values, 99.5), 50)

        for flag, color, lbl in [
            (correct, c_correct, "Correct"),
            (~correct, c_incorrect, "Incorrect"),
        ]:
            v = values[flag]
            counts, edges = np.histogram(v, bins=bins)
            density = counts / (counts.sum() * np.diff(edges))
            ax.bar(
                edges[:-1],
                density,
                width=np.diff(edges),
                align="edge",
                alpha=0.55,
                color=color,
                label=lbl,
            )

        ax.text(
            0.05,
            0.95,
            f"{label}\nAUROC={auroc:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("Probability Density" if col_idx % 2 == 0 else "")
        ax.set_xlabel(col)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Hide unused axes
    for ax in axes.flat[n_sig:]:
        ax.set_visible(False)

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, "uncertainty_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_calibration(test_df):
    """Reliability diagram for assigned_prob."""
    if "assigned_prob" not in test_df.columns:
        return

    correct = test_df["is_correct"].values
    probs = test_df["assigned_prob"].values

    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(probs[mask].mean())
        else:
            bin_accs.append(np.nan)
            bin_confs.append((lo + hi) / 2)

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)

    # Print per-QN accuracies for the paper table
    qn_pairs = [
        ("AFGL_m1", "pred_m1"),
        ("AFGL_m2", "pred_m2"),
        ("AFGL_m3", "pred_m3"),
        ("AFGL_r", "pred_r"),
    ]
    qn_names = ["m1", "m2", "m3", "r"]
    print("Per-QN accuracy on MARVEL test set (for table):")
    for name, (tc, pc) in zip(qn_names, qn_pairs):
        if tc in test_df.columns and pc in test_df.columns:
            acc = (test_df[tc] == test_df[pc]).mean() * 100
            print(f"  {name}: {acc:.2f}%")
    print()

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

    # ── Reliability diagram ───────────────────────────────────────────────────
    valid = ~np.isnan(bin_accs)
    bin_width = 1.0 / n_bins * 0.8
    ax1.bar(
        bin_confs[valid],
        bin_accs[valid],
        width=bin_width,
        alpha=0.65,
        color="#2980b9",
        label="Observed accuracy",
    )
    ax1.set_xlabel("Mean predicted probability (assigned_prob)")
    ax1.set_ylabel("Observed accuracy")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "calibration.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    test_df = _load_test_df()
    available_signals, auroc_by_col = analyse(test_df)
    plot_comparison(test_df, available_signals, auroc_by_col)
    plot_calibration(test_df)


if __name__ == "__main__":
    main()
