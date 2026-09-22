"""Serialize MARVEL CO2 rows to plain text for Laya/ModernBERT fine-tuning.

Source: ../../data/unified_co2_graph_data.csv, filtered to is_marvel == True.
Reuses the existing train_mask/val_mask/test_mask columns so the eventual
comparison against data/run_metrics.json (the GNN baseline) is apples-to-apples.
"""

import json
from pathlib import Path

import pandas as pd

DATA_CSV = Path(__file__).resolve().parents[2] / "data" / "unified_co2_graph_data.csv"
OUT_DIR = Path(__file__).resolve().parent / "data"


def serialize_row(row: pd.Series) -> str:
    return (
        f"isotope {row.isotope_id}, J={row.J}, parity={row.parity_encoded}, "
        f"polyad={row.polyad}, energy={row.energy:.6f}, dom_coeff={row.dom_coeff:.4f}, "
        f"t1={row.t1}, t2={row.t2}, t3={row.t3}, is_symmetric={int(row.is_symmetric)}, "
        f"sym_A1={row.tot_sym_A1:.2f}, sym_A2={row.tot_sym_A2:.2f}, "
        f"sym_B1={row.tot_sym_B1:.2f}, sym_B2={row.tot_sym_B2:.2f}, "
        f"masses={row.C_mass}/{row.O_A_mass}/{row.O_B_mass}"
    )


def main():
    df = pd.read_csv(DATA_CSV)
    marvel = df[df["is_marvel"] == True].copy()  # noqa: E712
    print(f"marvel rows: {len(marvel)}")

    OUT_DIR.mkdir(exist_ok=True)
    for split, mask_col in [("train", "train_mask"), ("val", "val_mask"), ("test", "test_mask")]:
        split_df = marvel[marvel[mask_col] == True]  # noqa: E712
        out_path = OUT_DIR / f"{split}.jsonl"
        with out_path.open("w") as f:
            for _, row in split_df.iterrows():
                rec = {"text": serialize_row(row), "label": int(row.combinatorial_class_id)}
                f.write(json.dumps(rec) + "\n")
        print(f"{split}: {len(split_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
