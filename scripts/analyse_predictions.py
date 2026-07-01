import pandas as pd
import os

DATA_DIR = "data"
PREDICTIONS_PATH = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "confident_new_assignments.csv")

# assigned_prob (softmax probability of solver-selected class) is the authoritative
# confidence signal: AUROC=0.967. Conflicted states receive lower assigned_prob,
# so no explicit conflict filter is needed.
PROB_THRESHOLD = 0.95


def analyze_new_assignments():
    print(f"Loading predictions from {PREDICTIONS_PATH}...\n")
    df = pd.read_csv(PREDICTIONS_PATH)

    # 1. Isolate the Inference States
    # We only care about states that were NOT originally MARVEL assigned
    inference_df = df[~df["is_marvel"]].copy()

    # 2. Drop failed assignments (if the Hungarian solver couldn't fit them)
    assigned_df = inference_df[inference_df["pred_class_id"] != -1].copy()

    # 3. Identify Conflicts and Confidence
    # assigned_prob already suppresses Hungarian-conflicted nodes (lower prob),
    # so no explicit conflict filter is required here.
    assigned_df["hungarian_conflict"] = (
        assigned_df["raw_class_id"] != assigned_df["pred_class_id"]
    )

    confident_df = assigned_df[assigned_df["assigned_prob"] >= PROB_THRESHOLD].copy()

    low_confidence_df = assigned_df[
        assigned_df["assigned_prob"] < PROB_THRESHOLD
    ].copy()

    # 4. Print Summary Statistics
    print("=== ASSIGNMENT SUMMARY ===")
    print(f"Total Target Inference States (Ca): {len(inference_df):,}")
    print(f"Successfully Mapped via Hungarian:  {len(assigned_df):,}")
    print(
        f"Highly Confident New Assignments:   {len(confident_df):,} (Prob >= {PROB_THRESHOLD})"
    )
    print(f"Low Confidence / Ambiguous States:  {len(low_confidence_df):,}")

    print("\n=== CONFIDENT NEW ASSIGNMENTS PER ISOTOPE ===")
    isotope_counts = (
        confident_df.groupby("isotope_id").size().reset_index(name="New Levels")
    )
    print(isotope_counts.to_string(index=False))

    # Keep only the essential columns for readability
    keep_cols = [
        "isotope_id",
        "energy",
        "J",
        "parity_encoded",
        "t1",
        "t2",
        "t3",
        "dom_coeff",
        "pred_m1",
        "pred_m2",
        "pred_m3",
        "pred_r",
        "assigned_prob",
    ]

    final_output = confident_df[keep_cols].sort_values(["isotope_id", "energy"])
    final_output.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved clean, highly confident assignments to {OUTPUT_PATH}")


if __name__ == "__main__":
    analyze_new_assignments()
