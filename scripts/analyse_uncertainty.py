"""
Diagnostic script: compare all available uncertainty signals as predictors
of 4-QN correctness on the MARVEL test set.

Outputs
-------
- Console table: AUROC, KS statistic, Youden threshold, precision @ T=1.0
- data/figures/uncertainty_comparison.png: density + precision-retention plots
  for every signal side-by-side
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

DATA_DIR = "data"
PREDICTIONS_PATH = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")
RELAXED_PATH = os.path.join(DATA_DIR, "final_relaxed_assignments.csv")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")
HARVEST_THRESHOLD = 1.0

mpl.rcParams.update({"font.size": 11, "legend.frameon": False})


# ── Signal metadata ───────────────────────────────────────────────────────────
# (column_name, display_label, higher_is_more_confident)
SIGNAL_DEFS = [
    ("assigned_margin",       "Assigned\nLogit Margin",         True),
    ("logit_margin",          "Raw\nLogit Margin",              True),
    ("assigned_prob",         "Assigned\nSoftmax Prob",         True),
    ("entropy",               "Softmax\nEntropy",               False),
    ("assignment_variance",   "MC Dropout\nVariance",           False),
    ("mc_predictive_entropy", "MC Predictive\nEntropy",         False),
    ("mc_bald",               "BALD\n(Epistemic Unc.)",         False),
]


def _load_test_df():
    """Load and merge prediction CSVs; return MARVEL test rows with correctness flag."""
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"{PREDICTIONS_PATH} not found. Run train.py first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)
    test_df = df[df["test_mask"]].copy()

    # Merge MC-dropout signals from the relaxed CSV if available
    if os.path.exists(RELAXED_PATH):
        relaxed = pd.read_csv(RELAXED_PATH)
        mc_cols = [
            c for c in [
                "assignment_variance", "raw_variance",
                "mc_predictive_entropy", "mc_bald",
            ] if c in relaxed.columns
        ]
        if mc_cols and "node_id" in relaxed.columns and "node_id" in test_df.columns:
            relaxed_sub = relaxed[["node_id"] + mc_cols]
            test_df = test_df.merge(relaxed_sub, on="node_id", how="left")
            print(f"Merged MC dropout columns from {RELAXED_PATH}: {mc_cols}")
        else:
            print(f"Skipping relaxed CSV merge (missing node_id or MC columns).")
    else:
        print(f"No relaxed assignments CSV found at {RELAXED_PATH}. Skipping MC dropout.")

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
    prec, ret = [], []
    for t in thresholds:
        mask = (values >= t) if higher_is_confident else (values <= t)
        retained = mask.sum()
        if retained == 0:
            break
        prec.append(correct[mask].mean() * 100)
        ret.append(retained / n_total * 100)
    return thresholds[: len(prec)], np.array(prec), np.array(ret)


def _youden_threshold(values, correct, higher_is_confident):
    """Return the threshold maximising sensitivity + specificity - 1."""
    lo, hi = values.min(), values.max()
    thresholds = np.linspace(lo, hi, 500)
    best_j, best_t = -1, lo
    for t in thresholds:
        pos = (values >= t) if higher_is_confident else (values <= t)
        neg = ~pos
        tp = (correct & pos).sum()
        fn = (correct & neg).sum()
        tn = (~correct & neg).sum()
        fp = (~correct & pos).sum()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t


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
        prec_at_harvest = correct[harvest_mask].mean() * 100 if harvest_mask.sum() > 0 else float("nan")
        ret_at_harvest = harvest_mask.sum() / n_total * 100

        rows.append({
            "Signal": label.replace("\n", " "),
            "AUROC": f"{auroc:.3f}",
            "KS stat": f"{ks_stat:.3f}",
            "Youden T": f"{opt_t:.3f}",
            f"Prec@T={HARVEST_THRESHOLD}": f"{prec_at_harvest:.1f}%",
            f"Ret@T={HARVEST_THRESHOLD}": f"{ret_at_harvest:.1f}%",
        })
        available_signals.append((col, label, higher, values))

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("=== UNCERTAINTY SIGNAL COMPARISON (MARVEL test set) ===")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("=" * 70 + "\n")

    # Per-QN accuracy breakdown
    qn_pairs = [("AFGL_m1", "pred_m1"), ("AFGL_m2", "pred_m2"),
                ("AFGL_m3", "pred_m3"), ("AFGL_r",  "pred_r")]
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
    n_total = len(test_df)
    n_sig = len(available_signals)

    if n_sig == 0:
        print("No signals available to plot.")
        return

    fig, axes = plt.subplots(
        2, n_sig,
        figsize=(4.5 * n_sig, 9),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    if n_sig == 1:
        axes = axes.reshape(2, 1)

    c_correct = "#2ecc71"
    c_incorrect = "#e74c3c"
    c_prec = "#2980b9"
    c_ret = "#8e44ad"

    for col_idx, (col, label, higher, values) in enumerate(available_signals):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        auroc = auroc_by_col.get(col, float("nan"))

        # ── Row 1: density histograms ─────────────────────────────────────────
        bins = np.linspace(np.percentile(values, 0.5), np.percentile(values, 99.5), 50)

        for flag, color, lbl in [
            (correct,  c_correct,   "Correct"),
            (~correct, c_incorrect, "Incorrect"),
        ]:
            v = values[flag]
            counts, edges = np.histogram(v, bins=bins)
            density = counts / (counts.sum() * np.diff(edges))
            ax_top.bar(edges[:-1], density, width=np.diff(edges),
                       align="edge", alpha=0.55, color=color, label=lbl)

        ax_top.set_title(f"{label}\nAUROC={auroc:.3f}", fontsize=11, fontweight="bold")
        ax_top.set_ylabel("Probability Density" if col_idx == 0 else "")
        ax_top.set_xlabel(col)
        ax_top.legend(fontsize=9)
        ax_top.grid(axis="y", linestyle="--", alpha=0.4)

        # ── Row 2: precision-retention ────────────────────────────────────────
        thresholds, prec, ret = _precision_retention(values, correct, n_total, higher)

        ax_bot.plot(thresholds, prec, color=c_prec, linewidth=2,
                    label="Precision (%)")
        ax_bot.set_ylabel("Precision (%)", color=c_prec, fontsize=9)
        ax_bot.tick_params(axis="y", labelcolor=c_prec)
        ax_bot.set_ylim(max(0, prec.min() - 3), 101)

        ax_bot2 = ax_bot.twinx()
        ax_bot2.plot(thresholds, ret, color=c_ret, linewidth=2,
                     linestyle=":", label="Retention (%)")
        ax_bot2.set_ylabel("Retention (%)", color=c_ret, fontsize=9)
        ax_bot2.tick_params(axis="y", labelcolor=c_ret)
        ax_bot2.set_ylim(0, 105)

        # Mark harvest threshold line (for margin-like signals only)
        if higher and col in ("assigned_margin", "logit_margin"):
            ax_top.axvline(HARVEST_THRESHOLD, color="black", linestyle="--",
                           linewidth=1.5, label=f"T={HARVEST_THRESHOLD}")
            ax_bot.axvline(HARVEST_THRESHOLD, color="black", linestyle="--",
                           linewidth=1.5)

        ax_bot.set_xlabel(col)
        ax_bot.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Uncertainty Signal Comparison — MARVEL Test Set\n"
        "(Row 1: correct vs incorrect distributions | Row 2: precision-retention)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, "uncertainty_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_calibration(test_df):
    """Reliability diagram for assigned_prob + per-QN accuracy bar chart."""
    if "assigned_prob" not in test_df.columns:
        return

    correct = test_df["is_correct"].values
    probs = test_df["assigned_prob"].values

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(probs[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(np.nan)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)

    qn_pairs = [("AFGL_m1", "pred_m1"), ("AFGL_m2", "pred_m2"),
                ("AFGL_m3", "pred_m3"), ("AFGL_r",  "pred_r")]
    qn_names = ["m1", "m2", "m3", "r"]
    qn_accs = [
        (test_df[tc] == test_df[pc]).mean() * 100
        for tc, pc in qn_pairs
        if tc in test_df.columns and pc in test_df.columns
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # ── Reliability diagram ───────────────────────────────────────────────────
    valid = ~np.isnan(bin_accs)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax1.bar(bin_confs[valid], bin_accs[valid], width=0.08, alpha=0.65,
            color="#2980b9", label="Observed accuracy")
    ax1.set_xlabel("Mean predicted probability (assigned_prob)")
    ax1.set_ylabel("Observed accuracy")
    ax1.set_title("Reliability Diagram (assigned_prob)", fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(linestyle="--", alpha=0.4)

    # Annotate each bar with count
    for conf, acc, cnt in zip(bin_confs[valid], bin_accs[valid], bin_counts[valid]):
        ax1.text(conf, acc + 0.02, str(int(cnt)), ha="center", fontsize=7, color="gray")

    # ── Per-QN accuracy bar chart ─────────────────────────────────────────────
    colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]
    bars = ax2.bar(qn_names, qn_accs, color=colors, alpha=0.8, width=0.5)
    ax2.set_ylim(min(qn_accs) - 3, 101)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Per-Quantum-Number Accuracy (MARVEL test set)", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, acc in zip(bars, qn_accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, acc + 0.3,
                 f"{acc:.2f}%", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Calibration & Per-QN Accuracy — MARVEL Test Set",
                 fontsize=12, fontweight="bold")
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
