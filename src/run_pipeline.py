import subprocess
import os
import json
import pandas as pd
import numpy as np
import time

from plotting import (
    plot_margin_boxplot,
    plot_pipeline_progression,
    plot_energy_coverage_by_generation,
    plot_confidence_validation_final,
)
from metrics import calculate_final_metrics

# ── Configuration ─────────────────────────────────────────────────────────────
ITERATIONS = 5
DATA_DIR = "data"
FIGURES_DIR = os.path.join(DATA_DIR, "figures")
UNIFIED_DATASET_PATH = os.path.join(DATA_DIR, "unified_co2_graph_data.csv")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")
METRICS_PATH = os.path.join(DATA_DIR, "run_metrics.json")
SUMMARY_PATH = os.path.join(DATA_DIR, "pipeline_summary.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)


def _run_step(cmd, label):
    """Runs a subprocess step; returns True on success."""
    print(f"\n>>> {label}...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {label} failed (exit code {result.returncode}).")
        return False
    return True


def main():
    pipeline_start = time.time()

    # ── Clear stale metrics from a previous full run ──────────────────────────
    if os.path.exists(METRICS_PATH):
        os.remove(METRICS_PATH)
        print(f"Cleared stale metrics from {METRICS_PATH}.")

    print(f"\nStarting Bootstrapping Pipeline — {ITERATIONS} generation(s).\n")

    # ── Generation 0: initial training run (no harvested states yet) ──────────
    print("=" * 60)
    print("=== GENERATION 0: INITIAL TRAINING RUN ===")
    print("=" * 60)
    if not _run_step(["uv", "run", "src/train.py"], "Initial Training"):
        return

    # ── Generations 1–N: bootstrap loop ──────────────────────────────────────
    for i in range(1, ITERATIONS + 1):
        print("\n" + "=" * 60)
        print(f"=== BOOTSTRAP CYCLE {i} OF {ITERATIONS} ===")
        print("=" * 60)

        # Harvest confident predictions into the training set
        boot_result = subprocess.run(
            ["uv", "run", "src/bootstrap.py"], capture_output=True
        )
        if boot_result.returncode != 0:
            print(f"[ERROR] Bootstrap step failed on cycle {i}. Halting.")
            return

        # Check whether bootstrap found anything to harvest
        df_check = pd.read_csv(UNIFIED_DATASET_PATH)
        if df_check["assignment_generation"].max() < i:
            print(f"Bootstrap cycle {i} found no new states. Pipeline converged early.")
            break

        # Retrain on the expanded dataset
        if not _run_step(["uv", "run", "src/train.py"], f"Training — Generation {i}"):
            return

    # ── Load accumulated metrics ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("=== PIPELINE COMPLETE. GENERATING FIGURES ===")
    print("=" * 60)

    if not os.path.exists(METRICS_PATH):
        print("[ERROR] run_metrics.json not found. Cannot generate progression plots.")
        metrics_history = []
    else:
        with open(METRICS_PATH) as f:
            metrics_history = json.load(f)
        print(f"Loaded {len(metrics_history)} generation metric records.")

    # ── Load final dataframes ─────────────────────────────────────────────────
    df_final = pd.read_csv(UNIFIED_DATASET_PATH)
    df_preds = pd.read_csv(PREDICTIONS_PATH)

    # ── Figure 1: Pipeline Progression (multi-panel) ──────────────────────────
    if metrics_history:
        plot_pipeline_progression(
            metrics_history,
            save_path=os.path.join(FIGURES_DIR, "pipeline_progression.png"),
        )

    # ── Figure 2: Margin Boxplot (per generation) ─────────────────────────────
    plot_df = df_final[df_final["assignment_generation"] > 0].copy()
    if not plot_df.empty:
        plot_margin_boxplot(
            plot_df,
            save_path=os.path.join(FIGURES_DIR, "margin_boxplot.png"),
        )

    # ── Figure 3: Energy Coverage by Generation ───────────────────────────────
    # Must use df_preds (assigned_co2_predictions.csv): df_final (unified dataset)
    # has assignment_generation but lacks pred_class_id, so only MARVEL states pass
    # the plot filter.
    plot_energy_coverage_by_generation(
        df_preds,
        save_path=os.path.join(FIGURES_DIR, "energy_coverage_by_generation.png"),
    )

    # ── Figure 4: Final Confidence Validation (from last prediction run) ──────
    plot_confidence_validation_final(
        df_preds,
        save_path=os.path.join(FIGURES_DIR, "confidence_validation_final.png"),
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    if metrics_history:
        summary_df = pd.DataFrame(metrics_history)
        summary_df.to_csv(SUMMARY_PATH, index=False)

        print("\n--- Pipeline Summary ---")
        display_cols = [
            "generation",
            "accuracy_4qn",
            "n_assigned",
            "assignment_rate",
            "n_harvested_this_gen",
            "median_margin",
        ]
        display_cols = [c for c in display_cols if c in summary_df.columns]
        print(summary_df[display_cols].to_string(index=False))

    # ── Final metrics on test set ──────────────────────────────────────────────
    calculate_final_metrics(df=df_final)

    elapsed = time.time() - pipeline_start
    print(f"\nTotal Pipeline Runtime: {elapsed:.2f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
