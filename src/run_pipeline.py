import subprocess
import os
import json
import pandas as pd
import time

from config import (
    DATA_DIR,
    UNIFIED_DATASET_PATH,
    PREDICTIONS_PATH,
    GRAPH_CACHE_PATH,
    BOOTSTRAP_METRICS_PATH,
)
from plotting import (
    plot_assignment_rate_by_energy,
    plot_bootstrap_margin_gain,
    plot_margin_boxplot,
    plot_pipeline_progression,
    plot_energy_coverage_by_generation,
    plot_confidence_validation_final,
)
from metrics import calculate_final_metrics

# ── Configuration ─────────────────────────────────────────────────────────────
ITERATIONS = 5
FIGURES_DIR = os.path.join(DATA_DIR, "figures")
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


def _reset_dataset():
    """
    Reset all bootstrapped Ca states to their pre-pipeline state so that
    each run starts cleanly from MARVEL-only training data.

    Resets Ca rows (is_marvel == False) that were promoted in a prior run:
      - assignment_generation  → 0
      - train_mask             → False
      - combinatorial_class_id → -1
      - locked_margin          → 10.0  (sentinel used by bootstrap.py)

    Also deletes the PyG graph cache so it is rebuilt from the clean dataset.
    Does nothing (and reports so) if no bootstrapped Ca states are present.
    """
    if not os.path.exists(UNIFIED_DATASET_PATH):
        print("No unified dataset found — skipping reset.")
        return

    df = pd.read_csv(UNIFIED_DATASET_PATH)

    gen_col = (
        df["assignment_generation"]
        if "assignment_generation" in df.columns
        else pd.Series(0, index=df.index)
    )
    ca_bootstrapped = (~df["is_marvel"]) & (gen_col > 0)
    n_to_reset = ca_bootstrapped.sum()

    if n_to_reset == 0:
        print("Dataset already clean — no bootstrapped Ca states to reset.")
    else:
        print(
            f"Resetting {n_to_reset:,} bootstrapped Ca states to pre-pipeline defaults..."
        )
        df.loc[ca_bootstrapped, "assignment_generation"] = 0
        df.loc[ca_bootstrapped, "train_mask"] = False
        df.loc[ca_bootstrapped, "combinatorial_class_id"] = -1
        if "locked_margin" in df.columns:
            df.loc[ca_bootstrapped, "locked_margin"] = 10.0
        df.to_csv(UNIFIED_DATASET_PATH, index=False)
        print(f"Dataset reset. MARVEL training states: {df['train_mask'].sum():,}")

    if os.path.exists(GRAPH_CACHE_PATH):
        os.remove(GRAPH_CACHE_PATH)
        print(f"Deleted stale PyG cache: {GRAPH_CACHE_PATH}")


def main():
    pipeline_start = time.time()

    # ── Reset dataset to MARVEL-only baseline ────────────────────────────────
    print("=" * 60)
    print("=== RESETTING DATASET ===")
    print("=" * 60)
    _reset_dataset()

    # ── Clear stale metrics from a previous full run ──────────────────────────
    if os.path.exists(METRICS_PATH):
        os.remove(METRICS_PATH)
        print(f"Cleared stale metrics from {METRICS_PATH}.")
    if os.path.exists(BOOTSTRAP_METRICS_PATH):
        os.remove(BOOTSTRAP_METRICS_PATH)
        print(f"Cleared stale bootstrap metrics from {BOOTSTRAP_METRICS_PATH}.")

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
        boot_result = subprocess.run(["uv", "run", "src/bootstrap.py"])
        if boot_result.returncode == 2:
            print(f"Bootstrap cycle {i} found no new states. Pipeline converged early.")
            break
        if boot_result.returncode != 0:
            print(f"[ERROR] Bootstrap step failed on cycle {i}. Halting.")
            return

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

    # ── Figure 5: Assignment Rate by Energy (stacked by bootstrap cohort) ─────
    # Generated only at end of pipeline so intermediate bootstrap steps do not
    # cause the Gen 0 band to appear to shrink mid-run.
    plot_assignment_rate_by_energy(
        df_preds,
        save_path=os.path.join(FIGURES_DIR, "assignment_rate_by_energy.png"),
    )

    # ── Figure 6: Bootstrap Margin Gain (violin by bootstrap cohort) ──────────
    plot_bootstrap_margin_gain(
        df_preds,
        save_path=os.path.join(FIGURES_DIR, "bootstrap_margin_gain.png"),
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    if metrics_history:
        summary_df = pd.DataFrame(metrics_history)
        if os.path.exists(BOOTSTRAP_METRICS_PATH):
            with open(BOOTSTRAP_METRICS_PATH) as f:
                boot_records = {
                    r["generation"]: r["n_r_excluded"] for r in json.load(f)
                }
            summary_df["n_r_excluded"] = (
                summary_df["generation"].map(boot_records).fillna(0).astype(int)
            )
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
