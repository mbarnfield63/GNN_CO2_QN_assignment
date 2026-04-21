import matplotlib as mpl
import matplotlib.pyplot as plt
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


def plot_variance_boxplot(df, save_path):
    # Set up the visualization
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

    # Add gridlines for easier reading
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tick_params(axis="x", which="both", bottom=False, top=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
