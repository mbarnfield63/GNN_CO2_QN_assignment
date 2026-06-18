import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# === Plotting parameters for thesis ready plots ===
thesis_params = {
    "xtick.minor.visible": True,
    "xtick.major.pad": 5,
    "xtick.direction": "in",
    "xtick.top": True,
    "ytick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.right": True,
    "font.family": "DejaVu Sans",
    "font.size": 14.0,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "legend.labelspacing": 0,
    "legend.borderpad": 0.5,
}
mpl.rcParams.update(thesis_params)

# Labels (must match generate_figures.py CONFIDENT_THRESHOLD)
HARVEST_THRESHOLD = 1.0
LBL_MARVEL = "MARVEL (Ground Truth)"
LBL_CONFIDENT = f"ML Confident (Margin \u2265 {HARVEST_THRESHOLD})"
LBL_CONSTRAINED = f"Physically Constrained (Margin < {HARVEST_THRESHOLD})"
LBL_UNASSIGNED = "Unassigned"

# Strict Global Stacking Order (Bottom to Top)
CATEGORY_ORDER = [LBL_MARVEL, LBL_CONFIDENT, LBL_CONSTRAINED, LBL_UNASSIGNED]

# Fixed generation colour map \u2014 keyed by generation number (0\u20135), viridis light\u2192dark.
# Using a fixed map ensures Gen 3 looks the same whether the run had 3 or 5 generations.
_GEN_VIRIDIS = plt.cm.viridis(np.linspace(0.9, 0.1, 6))
GEN_COLORS = {g: _GEN_VIRIDIS[g] for g in range(6)}

# Color palette — viridis-inspired, but MARVEL uses amber (#d4a017) instead of
# near-white yellow (#fde725) so it remains visible on white paper/slides.
colors = {
    LBL_MARVEL: "#d4a017",      # Amber (was #fde725 — invisible on white)
    LBL_CONFIDENT: "#35b779",   # Green
    LBL_CONSTRAINED: "#31688e", # Blue
    LBL_UNASSIGNED: "#440154",  # Purple
}

# Hatching patterns
hatching = {
    LBL_MARVEL: "XX",
    LBL_CONFIDENT: "",
    LBL_CONSTRAINED: "//",
    LBL_UNASSIGNED: ".",
}


def plot_per_isotopologue(df, PLOT_DIR="data/figures"):
    """State distribution per isotopologue."""
    print("Generating Per-Isotopologue Plot...")

    counts = (
        df.groupby(["isotope_id", "Assignment_Category"]).size().unstack(fill_value=0)
    )

    # Enforce strict bottom-to-top stacking order
    ordered_cols = [c for c in CATEGORY_ORDER if c in counts.columns]
    counts = counts[ordered_cols]

    counts["Total"] = counts.sum(axis=1)
    counts = counts.sort_values("isotope_id", ascending=True).drop("Total", axis=1)

    fig, ax = plt.subplots(figsize=(12, 7))

    counts.plot(
        kind="bar",
        stacked=True,
        color=[colors[c] for c in counts.columns],
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
    )

    # Apply hatches manually
    for container, category_name in zip(ax.containers, counts.columns):
        hatch_pattern = hatching[category_name]
        for patch in container:
            patch.set_hatch(hatch_pattern)

    ax.set_xlabel("Isotopologue")
    ax.set_ylabel("Number of Energy States")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Shrink current axis's height by 10% to make room for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(
        title="Assignment Type", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2
    )

    save_path = os.path.join(PLOT_DIR, "yield_per_isotopologue.png")
    plt.savefig(save_path)
    plt.close()


def plot_energy_distribution(df, bin_size=1000, PLOT_DIR="data/figures"):
    """Stacked histogram of assignment success across the energy spectrum."""
    print("Generating Energy Distribution Histogram...")

    fig, ax = plt.subplots(figsize=(12, 6))

    plot_df = df[df["Assignment_Category"] != LBL_UNASSIGNED].copy()

    # Bottom-to-top visual order
    active_cats = [
        c for c in CATEGORY_ORDER if c in plot_df["Assignment_Category"].unique()
    ]

    # Seaborn puts the LAST item in hue_order at the bottom of the visual stack
    hue_order = active_cats[::-1]

    sns.histplot(
        data=plot_df,
        x="energy",
        hue="Assignment_Category",
        hue_order=hue_order,
        multiple="stack",
        binwidth=bin_size,
        palette=[colors[c] for c in hue_order],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
        ax=ax,
        legend=False,
    )

    patch_drawing_order = active_cats

    num_hues = len(patch_drawing_order)
    total_patches = len(ax.patches)
    if num_hues > 0 and total_patches > 0:
        bins_per_hue = total_patches // num_hues
        for i, patch in enumerate(ax.patches):
            hue_idx = i // bins_per_hue
            if hue_idx < num_hues:
                category = patch_drawing_order[hue_idx]
                patch.set_hatch(hatching[category])

    ax.set_xlabel("Energy (cm$^{-1}$)")
    ax.set_ylabel("Number of Assigned States")

    legend_handles = [
        mpatches.Patch(
            facecolor=colors[c], hatch=hatching[c], edgecolor="black", label=c
        )
        for c in active_cats[
            ::-1
        ]  # Reverse so Constrained is at the top of the legend box
    ]
    ax.legend(handles=legend_handles, loc="best", title="Assignment Type")

    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_xlim(left=0)

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "energy_distribution.png")
    plt.savefig(save_path)
    plt.close()


def plot_polyad_ladders(df, PLOT_DIR="data/figures"):
    """Polyad ladder scatter plot (J small)."""
    print("Generating Polyad Ladder Plot...")

    if "polyad" not in df.columns:
        print("Missing 'polyad' column. Making polyad...")
        if df["is_marvel"] == True:
            df["polyad"] = 2 * df["AFGL_m1"] + df["AFGL_m2"] + 3 * df["AFGL_m3"]
        else:
            df["polyad"] = 2 * df["pred_m1"] + df["pred_m2"] + 3 * df["pred_m3"]

    subset_df = df[df["J"] == 2].copy()

    marvel = subset_df[subset_df["is_marvel"]]
    confident = subset_df[subset_df["Assignment_Category"] == LBL_CONFIDENT]
    constrained = subset_df[subset_df["Assignment_Category"] == LBL_CONSTRAINED]

    fig, ax = plt.subplots(figsize=(12, 8))

    # 1. ML Confident
    ax.scatter(
        confident["polyad"],
        confident["energy"],
        color=colors[LBL_CONFIDENT],
        label=LBL_CONFIDENT,
        alpha=0.6,
        s=20,
        marker="x",
    )

    # 2. Physically Constrained
    ax.scatter(
        constrained["polyad"],
        constrained["energy"],
        color=colors[LBL_CONSTRAINED],
        label=LBL_CONSTRAINED,
        alpha=0.6,
        s=20,
        marker="*",
    )

    # 3. MARVEL Last
    ax.scatter(
        marvel["polyad"],
        marvel["energy"],
        color=colors[LBL_MARVEL],
        label=LBL_MARVEL,
        alpha=1.0,
        s=25,
        marker="o",
        edgecolor="black",
        linewidth=0.25,
    )

    ax.set_xlabel("Polyad Number ($P = 2v_1 + v_2 + 3v_3$), J = 2")
    ax.set_ylabel("Energy (cm$^{-1}$)")

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "polyad_ladders.png")
    plt.savefig(save_path)
    plt.close()


def plot_margin_boxplot(df, save_path):
    """
    Boxplot of assigned logit margin per bootstrap generation.
    Shows the confidence floor rising as the model learns from its own
    high-quality predictions.
    """
    print("Generating Logit Margin Boxplot...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    margin_col = "locked_margin" if "locked_margin" in df.columns else "logit_margin"

    gen_order = sorted(df["assignment_generation"].unique())
    palette = [GEN_COLORS.get(g, GEN_COLORS[5]) for g in gen_order]

    sns.boxplot(
        data=df,
        x="assignment_generation",
        y=margin_col,
        order=gen_order,
        palette=palette,
        ax=ax,
        showfliers=False,
        linewidth=1.5,
        width=0.55,
    )

    ax.axhline(
        1.0,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        label="Harvest threshold (margin = 1.0)",
    )

    # Annotate median values above each box
    for i, gen in enumerate(gen_order):
        median = df.loc[df["assignment_generation"] == gen, margin_col].median()
        ax.text(
            i,
            median + 0.08,
            f"{median:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            fontweight="bold",
        )

    ax.set_xlabel("Bootstrap Generation")
    ax.set_ylabel("Assigned Logit Margin")
    ax.set_title(
        "Evolution of Model Confidence Across Bootstrap Generations\n"
        "(Assigned logit margin of harvested inference states)"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_pipeline_progression(metrics_history, save_path):
    """
    Three-panel figure showing the evolution of key pipeline metrics
    across bootstrap generations.

    Panel A: 4-QN accuracy on the held-out MARVEL test set
    Panel B: Total assigned states and cumulative harvest
    Panel C: Mean absolute error per quantum number
    """
    print("Generating Pipeline Progression Plot...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    gens = [m["generation"] for m in metrics_history]
    acc = [m["accuracy_4qn"] * 100 for m in metrics_history]
    n_assigned = [m["n_assigned"] for m in metrics_history]
    assign_rate = [m["assignment_rate"] * 100 for m in metrics_history]
    mae_m1 = [m.get("mae_m1", 0) for m in metrics_history]
    mae_m2 = [m.get("mae_m2", 0) for m in metrics_history]
    mae_m3 = [m.get("mae_m3", 0) for m in metrics_history]
    mae_r = [m.get("mae_r", 0) for m in metrics_history]

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 1, hspace=0.42)

    marker_kw = dict(marker="o", markersize=7, linewidth=2)

    # ── Panel A: 4-QN Accuracy ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(gens, acc, color="#2ecc71", **marker_kw, label="4-QN accuracy")
    ax1.fill_between(gens, acc, alpha=0.12, color="#2ecc71")

    ax1.set_ylabel("4-QN Perfect Match (%)")
    ax1.set_title(
        "A  —  Model Accuracy on Held-out MARVEL Test Set",
        loc="left",
        fontweight="bold",
        fontsize=12,
    )
    ax1.set_xticks(gens)
    ax1.set_ylim(
        max(0, min(acc) - 3),
        min(100, max(acc) + 3),
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Annotate each point
    for g, a in zip(gens, acc):
        ax1.annotate(
            f"{a:.1f}%",
            (g, a),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
        )

    # ── Panel B: Assignment yield ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])

    ax2_r = ax2.twinx()
    ax2.bar(
        gens,
        n_assigned,
        color="#3498db",
        alpha=0.65,
        label="Total assigned states",
        width=0.5,
        zorder=2,
    )
    ax2_r.plot(
        gens,
        assign_rate,
        color="#e67e22",
        **marker_kw,
        label="Assignment rate (%)",
        zorder=3,
    )

    ax2.set_ylabel("Total Assigned States", color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")
    ax2_r.set_ylabel("Assignment Rate (%)", color="#e67e22")
    ax2_r.tick_params(axis="y", labelcolor="#e67e22")
    ax2_r.set_ylim(0, 105)

    ax2.set_title(
        "B  —  Yield: Total Assigned States & Assignment Rate",
        loc="left",
        fontweight="bold",
        fontsize=12,
    )
    ax2.set_xticks(gens)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    # ── Panel C: MAE per quantum number ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])

    qn_colors = {"m₁": "#e74c3c", "m₂": "#9b59b6", "m₃": "#2980b9", "r": "#27ae60"}
    for label, data, color in [
        ("m₁", mae_m1, qn_colors["m₁"]),
        ("m₂", mae_m2, qn_colors["m₂"]),
        ("m₃", mae_m3, qn_colors["m₃"]),
        ("r", mae_r, qn_colors["r"]),
    ]:
        ax3.plot(gens, data, color=color, label=label, **marker_kw)

    ax3.set_xlabel("Bootstrap Generation")
    ax3.set_ylabel("Mean Absolute Error")
    ax3.set_title(
        "C  —  Physical MAE per Quantum Number (MARVEL Test Set)",
        loc="left",
        fontweight="bold",
        fontsize=12,
    )
    ax3.set_xticks(gens)
    ax3.set_ylim(bottom=0)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", ncol=4)

    fig.suptitle(
        "CO₂ Quantum Number Assignment — Bootstrap Pipeline Progression",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_energy_coverage_by_generation(df, save_path, bin_size=500):
    """
    Stacked histogram showing which energy range each bootstrap generation covers.

    Expects df_preds (assigned_co2_predictions.csv) which has both pred_class_id
    and assignment_generation. Generation 0 Ca is hatched to distinguish it as the
    initial GNN pass (not a bootstrapped generation). Unassigned Ca states are shown
    on top in a colour outside the viridis scale.
    """
    print("Generating Energy Coverage by Generation Plot...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = df[df["energy"] <= 15000].copy()

    marvel_mask = df["is_marvel"]
    has_pred = "pred_class_id" in df.columns

    # Each segment: (mask, label, color, hatch)
    segments = []
    segments.append((marvel_mask, "MARVEL (Ground Truth)", colors[LBL_MARVEL], None))

    UNASSIGNED_COLOR = "#c0c0c0"  # silver-gray — outside viridis scale

    if has_pred:
        ca_gen0_mask = (
            (~df["is_marvel"]) & (df["assignment_generation"] == 0) & (df["pred_class_id"] >= 0)
        )
        # Hatched to signal "initial GNN inference, not yet bootstrapped"
        segments.append((ca_gen0_mask, "GNN Initial (Gen 0)", "#b5cf6b", "///"))

        max_gen = int(df.loc[~df["is_marvel"], "assignment_generation"].max())
        for g in range(1, max_gen + 1):
            ca_gen_mask = (
                (~df["is_marvel"]) & (df["assignment_generation"] == g) & (df["pred_class_id"] >= 0)
            )
            segments.append((ca_gen_mask, f"Bootstrap Gen {g}", GEN_COLORS.get(g, GEN_COLORS[5]), None))

        # Unassigned Ca on top — outside viridis scale
        unassigned_mask = (~df["is_marvel"]) & (df["pred_class_id"] < 0)
        segments.append((unassigned_mask, "Unassigned Ca", UNASSIGNED_COLOR, None))
    else:
        print("  Warning: pred_class_id not found; only MARVEL states will be shown.")

    bins = np.arange(0, 15001, bin_size)

    fig, ax_main = plt.subplots(figsize=(13, 6))

    bottoms = np.zeros(len(bins) - 1)
    for mask, label, color, hatch in segments:
        counts, _ = np.histogram(df.loc[mask, "energy"], bins=bins)
        ec = "black" if hatch else "none"
        ax_main.bar(
            bins[:-1], counts, width=bin_size, bottom=bottoms,
            color=color, label=label, alpha=0.85, align="edge",
            edgecolor=ec, linewidth=0.4, hatch=hatch,
        )
        bottoms += counts

    ax_main.set_xlabel("Energy (cm⁻¹)")
    ax_main.set_ylabel("Number of States")
    ax_main.set_title(
        "CO₂ Pipeline — Energy Coverage Across Bootstrap Generations",
        loc="left", fontweight="bold", fontsize=12,
    )
    ax_main.legend(loc="upper left", fontsize=10)
    ax_main.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_variance_validation(df, PLOT_DIR="data/figures"):
    """
    Validates the logit margin as a confidence proxy.
    Panel A: Normalised density distributions (correct vs incorrect).
    Panel B: Precision curve — fraction correct as a function of margin threshold.
    """
    print("Generating Confidence Validation Plot on Ground Truth Test Set...")

    test_df = df[df["test_mask"]].copy()

    # --- Grade against POST-Hungarian assignments (consistent with reported MAE) ---
    is_correct = (
        (test_df["AFGL_m1"] == test_df["pred_m1"])
        & (test_df["AFGL_m2"] == test_df["pred_m2"])
        & (test_df["AFGL_m3"] == test_df["pred_m3"])
        & (test_df["AFGL_r"] == test_df["pred_r"])
    )
    test_df["Accuracy"] = np.where(is_correct, "Correct (4-QN)", "Incorrect (4-QN)")

    # Use assigned_margin (post-Hungarian) if available, fall back to logit_margin
    margin_col = (
        "assigned_margin" if "assigned_margin" in test_df.columns else "logit_margin"
    )
    n_correct = is_correct.sum()
    n_incorrect = (~is_correct).sum()
    n_total = len(test_df)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [2, 1]}
    )

    # ── Panel A: Normalised density ──────────────────────────────────────────
    sns.histplot(
        data=test_df,
        x=margin_col,
        hue="Accuracy",
        hue_order=["Correct (4-QN)", "Incorrect (4-QN)"],
        palette={"Correct (4-QN)": "#2ecc71", "Incorrect (4-QN)": "#e74c3c"},
        bins=60,
        stat="density",  # <-- normalise each distribution independently
        common_norm=False,  # <-- critical: each hue normalised to its own area
        alpha=0.6,
        ax=ax1,
    )

    threshold = HARVEST_THRESHOLD
    ax1.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Harvest threshold = {threshold}",
    )

    # Annotate counts so the reader understands the class ratio
    ax1.text(
        0.98,
        0.96,
        f"Correct: {n_correct:,}  |  Incorrect: {n_incorrect:,}\n"
        f"Overall accuracy: {n_correct/n_total*100:.1f}%",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    ax1.set_xlabel("")
    ax1.set_ylabel("Probability Density")
    ax1.set_title(
        "Logit Margin Distribution: Correct vs. Incorrect 4-QN Predictions\n"
        "(Normalised independently — shape comparison, not count comparison)"
    )
    ax1.legend(title="Prediction")
    ax1.set_xlim(left=0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # ── Panel B: Precision curve ─────────────────────────────────────────────
    # For each threshold T: among all nodes with margin >= T, what fraction correct?
    margins = test_df[margin_col].values
    correct = is_correct.values

    thresholds = np.linspace(0, margins.max() * 0.95, 200)
    precision_vals = []
    retention_vals = []

    for t in thresholds:
        mask = margins >= t
        retained = mask.sum()
        if retained == 0:
            break
        precision_vals.append(correct[mask].mean())
        retention_vals.append(retained / n_total)

    thresholds = thresholds[: len(precision_vals)]
    precision_vals = np.array(precision_vals)
    retention_vals = np.array(retention_vals)

    color_prec = "#2980b9"
    color_ret = "#8e44ad"

    ax2.plot(
        thresholds,
        precision_vals * 100,
        color=color_prec,
        linewidth=2,
        label="Precision (% correct among retained)",
    )
    ax2.set_ylabel("Precision (%)", color=color_prec)
    ax2.tick_params(axis="y", labelcolor=color_prec)
    ax2.set_ylim(bottom=max(0, precision_vals.min() * 100 - 2), top=101)

    ax2b = ax2.twinx()
    ax2b.plot(
        thresholds,
        retention_vals * 100,
        color=color_ret,
        linewidth=2,
        linestyle=":",
        label="Retention (% of test set)",
    )
    ax2b.set_ylabel("Retention (%)", color=color_ret)
    ax2b.tick_params(axis="y", labelcolor=color_ret)
    ax2b.set_ylim(0, 105)

    ax2.axvline(threshold, color="black", linestyle="--", linewidth=2)

    # Annotate the operating point
    t_idx = np.searchsorted(thresholds, threshold)
    if t_idx < len(precision_vals):
        op_prec = precision_vals[t_idx] * 100
        op_ret = retention_vals[t_idx] * 100
        ax2.annotate(
            f"T={threshold}: {op_prec:.1f}% precision\n{op_ret:.1f}% retained",
            xy=(threshold, op_prec),
            xytext=(threshold + 0.3, op_prec - 5),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

    ax2.set_xlabel("Assigned Logit Margin Threshold")
    ax2.set_title("Precision–Retention Trade-off vs. Margin Threshold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    save_path = os.path.join(PLOT_DIR, "confidence_validation.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_confidence_validation_final(df_preds, save_path):
    """
    Final publishable version of the confidence validation figure.

    Panel A: Normalised density of assigned_prob — correct vs. incorrect predictions.
             Threshold line at PROB_THRESHOLD (0.6, Youden-optimal for assigned_prob).
    Panel B: Precision–Retention trade-off vs. assigned_margin threshold.
             Threshold line at PAPER_MARGIN_THRESHOLD (1.5, removes visible bump of
             incorrect predictions seen above T=1.0).
    """
    PROB_THRESHOLD = 0.6
    PAPER_MARGIN_THRESHOLD = 1.5

    print("Generating Final Confidence Validation Plot...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    test_df = df_preds[df_preds["test_mask"]].copy()

    # Grade against post-Hungarian assignments
    is_correct = (
        (test_df["AFGL_m1"] == test_df["pred_m1"])
        & (test_df["AFGL_m2"] == test_df["pred_m2"])
        & (test_df["AFGL_m3"] == test_df["pred_m3"])
        & (test_df["AFGL_r"] == test_df["pred_r"])
    )
    test_df["Accuracy"] = np.where(is_correct, "Correct (4-QN)", "Incorrect (4-QN)")

    margin_col = (
        "assigned_margin" if "assigned_margin" in test_df.columns else "logit_margin"
    )
    prob_col = "assigned_prob" if "assigned_prob" in test_df.columns else margin_col

    n_correct = is_correct.sum()
    n_incorrect = (~is_correct).sum()
    n_total = len(test_df)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ── Normalised density of assigned_prob ───────────────────────────────────
    sns.histplot(
        data=test_df,
        x=prob_col,
        hue="Accuracy",
        hue_order=["Correct (4-QN)", "Incorrect (4-QN)"],
        palette={"Correct (4-QN)": "#2ecc71", "Incorrect (4-QN)": "#e74c3c"},
        bins=60,
        stat="density",
        common_norm=False,
        alpha=0.65,
        ax=ax1,
    )

    threshold_line = ax1.axvline(
        PROB_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=2,
    )

    ax1.set_xlabel("Assigned Softmax Probability")
    ax1.set_ylabel("Probability Density")
    ax1.set_xlim(0, 1)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Rebuild legend to include threshold line alongside seaborn hue entries
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(threshold_line)
    labels.append(f"Reporting threshold (p = {PROB_THRESHOLD})")
    ax1.legend(handles=handles, labels=labels, title="Prediction", loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
