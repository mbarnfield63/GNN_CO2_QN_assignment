"""LoRA fine-tune of ModernBERT-large (Laya's backbone) as a text row classifier,
generalised from the CO2 prototype (GNN_CO2_QN_assignment, branch CDSD-Analysis,
experiments/laya_prototype/train_laya.py). On CO2 this reached 94.88% exact match.

Input: data/train.jsonl and data/val.jsonl, one {"text": str, "label": int} per line.
How you serialise a level into "text" and encode its label is up to you.

    uv run src/train_classifier.py train
    uv run src/train_classifier.py predict data/test.jsonl logits.npy

`predict` writes raw logits (n, C) for src/assignment.py.

Precision auto-detects: bf16 on Ampere+ GPUs, fp16 + GradScaler on older cards
(weights stay fp32; autocast handles fp16 compute). CPU works but is slow.
Hyperparameters are overridable via LAYA_* env vars.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("LAYA_MODEL", "answerdotai/ModernBERT-large")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints" / "best"
MAX_LEN = int(os.environ.get("LAYA_MAX_LEN", 128))
BATCH_SIZE = int(os.environ.get("LAYA_BATCH_SIZE", 16))
GRAD_ACCUM = int(os.environ.get("LAYA_GRAD_ACCUM", 2))
EPOCHS = int(os.environ.get("LAYA_EPOCHS", 8))
LR = float(os.environ.get("LAYA_LR", 2e-4))
PATIENCE = int(os.environ.get("LAYA_PATIENCE", 2))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open()]


def batches(records, tokenizer, batch_size, shuffle=False):
    order = np.random.permutation(len(records)) if shuffle else np.arange(len(records))
    for i in range(0, len(order), batch_size):
        chunk = [records[j] for j in order[i : i + batch_size]]
        enc = tokenizer([r["text"] for r in chunk], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(DEVICE)
        labels = torch.tensor([r.get("label", -1) for r in chunk], device=DEVICE)
        yield enc, labels


def logits_of(model, enc):
    with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
        return model(**enc).logits.float()


def focal_loss(logits, labels, gamma=2.0):
    # Down-weights easy examples; helps with rare high-quantum-number labels.
    logpt = F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    return -((1 - logpt.exp()) ** gamma * logpt).mean()


@torch.no_grad()
def predict_logits(model, tokenizer, records):
    model.eval()
    return np.concatenate([logits_of(model, enc).cpu().numpy()
                           for enc, _ in batches(records, tokenizer, BATCH_SIZE * 4)])


def train():
    train_recs, val_recs = read_jsonl(DATA_DIR / "train.jsonl"), read_jsonl(DATA_DIR / "val.jsonl")
    num_labels = max(r["label"] for r in train_recs + val_recs) + 1
    print(f"device={DEVICE} precision={'bf16' if USE_BF16 else 'fp16'} labels={num_labels} "
          f"train={len(train_recs)} val={len(val_recs)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=num_labels, attn_implementation="sdpa")
    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["Wqkv", "Wo"], modules_to_save=["classifier"])).to(DEVICE)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda" and not USE_BF16)
    val_labels = np.array([r["label"] for r in val_recs])
    best, stale, t0 = 0.0, 0, time.time()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        for step, (enc, labels) in enumerate(batches(train_recs, tokenizer, BATCH_SIZE, shuffle=True)):
            scaler.scale(focal_loss(logits_of(model, enc), labels) / GRAD_ACCUM).backward()
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        val_acc = (predict_logits(model, tokenizer, val_recs).argmax(1) == val_labels).mean()
        print(f"epoch {epoch}: val_acc={val_acc:.4f} elapsed={time.time() - t0:.0f}s")
        if val_acc > best:
            best, stale = val_acc, 0
            model.save_pretrained(CKPT_DIR)
            (CKPT_DIR / "num_labels.json").write_text(json.dumps(num_labels))
        elif (stale := stale + 1) >= PATIENCE:
            break
    print(f"best val_acc={best:.4f}")


def predict(in_path, out_path):
    num_labels = json.loads((CKPT_DIR / "num_labels.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=num_labels, attn_implementation="sdpa")
    # The base load reports classifier.* as MISSING; the adapter restores them.
    model = PeftModel.from_pretrained(base, CKPT_DIR).to(DEVICE)
    np.save(out_path, predict_logits(model, tokenizer, read_jsonl(in_path)))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["train"]:
        train()
    elif sys.argv[1:2] == ["predict"] and len(sys.argv) == 4:
        predict(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
