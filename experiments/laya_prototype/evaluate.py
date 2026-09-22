"""Evaluate the LoRA-tuned ModernBERT-large classifier on the held-out MARVEL
test split, using the same metric definitions as ../../src/metrics.py so the
numbers are directly comparable to data/run_metrics.json (the GNN baseline).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "answerdotai/ModernBERT-large"
NUM_LABELS = 220
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CKPT_DIR = BASE_DIR / "checkpoints" / "best"
CLASS_MAPPING = BASE_DIR.parents[1] / "data" / "class_mapping.csv"
MAX_LEN = int(os.environ.get("LAYA_MAX_LEN", 128))
BATCH_SIZE = int(os.environ.get("LAYA_EVAL_BATCH_SIZE", 64))

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


def load_model(device):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=NUM_LABELS, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(base_model, CKPT_DIR).to(device)
    model.eval()
    return model


def main():
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_model(device)

    records = [json.loads(l) for l in (DATA_DIR / "test.jsonl").open()]
    texts = [r["text"] for r in records]
    true_class_ids = np.array([r["label"] for r in records])

    pred_class_ids = np.zeros(len(records), dtype=np.int64)
    t_start = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch_texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt"
            ).to(device)
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                logits = model(**enc).logits
            pred_class_ids[i : i + len(batch_texts)] = logits.argmax(dim=-1).cpu().numpy()
    inference_elapsed = time.time() - t_start
    throughput = len(texts) / inference_elapsed

    class_map = pd.read_csv(CLASS_MAPPING).set_index("class_id")
    true_qn = class_map.loc[true_class_ids, ["m1", "m2", "m3", "r"]].to_numpy()
    pred_qn = class_map.loc[pred_class_ids, ["m1", "m2", "m3", "r"]].to_numpy()

    perfect_match_acc = (true_qn == pred_qn).all(axis=1).mean() * 100

    targets = ["m1", "m2", "m3", "r"]
    rows = []
    for j, target in enumerate(targets):
        y_true, y_pred = true_qn[:, j], pred_qn[:, j]
        mae = np.abs(y_true - y_pred).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        rows.append(
            {
                "Target": f"AFGL_{target}",
                "MAE": round(mae, 4),
                "F1-Macro": round(f1, 4),
                "F1-Weighted": round(f1_wt, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
            }
        )

    results_df = pd.DataFrame(rows)
    mean_row = pd.DataFrame(
        [{"Target": "MEAN", **{c: round(results_df[c].mean(), 4) for c in results_df.columns[1:]}}]
    )
    results_df = pd.concat([results_df, mean_row], ignore_index=True)

    print("=" * 60)
    print("=== LAYA (ModernBERT-large LoRA) TEST SET METRICS ===")
    print("=" * 60)
    print(f"Test set size: {len(records)}")
    print(f"Perfect 4-QN Match Accuracy: {perfect_match_acc:.2f}%")
    print(f"GNN baseline (data/run_metrics.json): 94.5-96.9% across generations")
    print(f"Inference throughput: {throughput:.1f} rows/sec (peak GPU mem: {torch.cuda.max_memory_allocated()/1e6:.1f} MB)")
    print()
    print(results_df.to_string(index=False))
    print("=" * 60)

    marvel_df = pd.read_csv(BASE_DIR.parents[1] / "data" / "unified_co2_graph_data.csv")
    test_meta = marvel_df[(marvel_df["is_marvel"] == True) & (marvel_df["test_mask"] == True)].reset_index(drop=True)  # noqa: E712
    per_row = test_meta[["isotope_id", "J", "polyad", "energy"]].copy()
    per_row["true_class_id"] = true_class_ids
    per_row["pred_class_id"] = pred_class_ids
    per_row["correct"] = (true_qn == pred_qn).all(axis=1)
    per_row.to_csv(BASE_DIR / "per_row_predictions.csv", index=False)

    (BASE_DIR / "eval_results.json").write_text(
        json.dumps(
            {
                "test_size": len(records),
                "perfect_4qn_match_accuracy": perfect_match_acc,
                "inference_throughput_rows_per_sec": throughput,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1e6,
                "per_qn_metrics": results_df.to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
