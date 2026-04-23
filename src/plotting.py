import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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

# Labels
LBL_MARVEL = "MARVEL (Ground Truth)"
LBL_CONFIDENT = "ML Confident (Var \u2264 0.05)"
LBL_CONSTRAINED = "Physically Constrained (Var > 0.05)"
LBL_UNASSIGNED = "Unassigned"

# Strict Global Stacking Order (Bottom to Top)
CATEGORY_ORDER = [LBL_MARVEL, LBL_CONFIDENT, LBL_CONSTRAINED, LBL_UNASSIGNED]

# Color palette - viridis
colors = {
    LBL_MARVEL: "#fde725",  # Yellow
    LBL_CONFIDENT: "#35b779",  # Green
    LBL_CONSTRAINED: "#31688e",  # Blue
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

    marvel = subset_df[subset_df["is_marvel"] == True]
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


def plot_variance_boxplot(df, PLOT_DIR="data/figures"):
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="assignment_generation",
        y="locked_variance",
        palette="viridis",
        hue="assignment_generation",
        legend=False,
    )

    plt.xlabel("Bootstrapping Generation")
    plt.ylabel("Monte Carlo Dropout Variance")

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tick_params(axis="x", which="both", bottom=False, top=False)

    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "variance_boxplot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
