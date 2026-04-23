import subprocess
import os
import pandas as pd

from plotting import plot_variance_boxplot

# ==========================================
# CONFIGURATION
# ==========================================
ITERATIONS = 5
DATA_DIR = "data"
UNIFIED_DATASET_PATH = os.path.join(DATA_DIR, "unified_co2_graph_data.csv")
PLOT_PATH = os.path.join(DATA_DIR, "figures", "variance_boxplot.png")


def main():
    print(f"Starting Automated Bootstrapping Pipeline for {ITERATIONS} Iterations...\n")

    # ---------------------------------------------------------
    # 1. THE AUTOMATED LOOP
    # ---------------------------------------------------------
    for i in range(1, ITERATIONS + 1):
        print("=" * 60)
        print(f"=== STARTING BOOTSTRAP CYCLE {i} OF {ITERATIONS} ===")
        print("=" * 60)

        # Step A: Train the Network
        print(f"\n>>> Running Training Phase (Generation {i})...")
        train_process = subprocess.run(["uv", "run", "src/train.py"])

        if train_process.returncode != 0:
            print(f"\n[ERROR] Training failed on cycle {i}. Halting pipeline.")
            return

        # Step B: Harvest Confident Predictions
        print(f"\n>>> Running Bootstrapping Phase (Generation {i})...")
        boot_process = subprocess.run(["uv", "run", "src/bootstrap.py"])

        if boot_process.returncode != 0:
            print(f"\n[ERROR] Bootstrapping failed on cycle {i}. Halting pipeline.")
            return

    print("\n" + "=" * 60)
    print("=== PIPELINE COMPLETE. GENERATING VISUALIZATIONS ===")
    print("=" * 60)

    # ---------------------------------------------------------
    # 2. GENERATE THE VARIANCE BOXPLOT
    # ---------------------------------------------------------
    print(f"Loading data from {UNIFIED_DATASET_PATH}...")
    df = pd.read_csv(UNIFIED_DATASET_PATH)

    # We only want to plot the new states the model harvested
    # not the original MARVEL data (Gen 0)
    plot_df = df[df["assignment_generation"] > 0].copy()

    if plot_df.empty:
        print("No new assignments were harvested. Cannot generate plot.")
        return
    else:
        plot_variance_boxplot(plot_df, PLOT_PATH)

    print(f"\nSuccess! Boxplot saved to: {PLOT_PATH}")

    # Print a quick text summary of the plot
    summary_stats = plot_df.groupby("assignment_generation").agg(
        States_Harvested=("node_id", "count"),
        Median_Variance=("locked_variance", "median"),
        Max_Variance=("locked_variance", "max"),
    )
    print("\n--- Final Yield Summary ---")
    print(summary_stats.to_string())
    summary_stats.to_csv(
        os.path.join(DATA_DIR, f"pipeline_summary_{ITERATIONS}iterations.csv")
    )


if __name__ == "__main__":
    main()
