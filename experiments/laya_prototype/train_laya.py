"""LoRA fine-tune of ModernBERT-large (Laya's backbone architecture) on
text-serialized MARVEL CO2 rows -> combinatorial_class_id (220 classes).

sdpa attention throughout (portable, and PyTorch dispatches it to a flash-attention
kernel automatically on Ampere+ hardware). Precision auto-detects per-GPU: bf16 on
Ampere+ (A100/L40S) with no loss-scaling needed, fp16 + GradScaler on older cards
(RTX 2070, V100) which lack bf16 tensor cores.
Loss: focal loss (gamma=2.0), matching src/model.py, for a fairer comparison
given the same class imbalance exists here.

Hyperparameters are overridable via env vars (see LAYA_* below) so the same script
runs unmodified on a laptop GPU or a cluster A100 with a bigger batch size.
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "answerdotai/ModernBERT-large"
NUM_LABELS = 220
DATA_DIR = Path(__file__).resolve().parent / "data"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
MAX_LEN = int(os.environ.get("LAYA_MAX_LEN", 128))
BATCH_SIZE = int(os.environ.get("LAYA_BATCH_SIZE", 16))
GRAD_ACCUM = int(os.environ.get("LAYA_GRAD_ACCUM", 2))
EPOCHS = int(os.environ.get("LAYA_EPOCHS", 8))
LR = float(os.environ.get("LAYA_LR", 2e-4))
PATIENCE = int(os.environ.get("LAYA_PATIENCE", 2))
NUM_WORKERS = int(os.environ.get("LAYA_NUM_WORKERS", 2))

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


class RowDataset(Dataset):
    def __init__(self, path: Path, tokenizer):
        self.records = [json.loads(l) for l in path.open()]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(rec["label"], dtype=torch.long),
        }


def focal_loss(logits, labels, gamma=2.0):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    logpt = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    return -((1 - pt) ** gamma * logpt).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    model.train()
    return correct / total


def main():
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)} (cc {torch.cuda.get_device_capability(0)}), "
          f"precision: {'bf16' if USE_BF16 else 'fp16'}, "
          f"batch_size={BATCH_SIZE} grad_accum={GRAD_ACCUM} (effective {BATCH_SIZE * GRAD_ACCUM})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=NUM_LABELS, attn_implementation="sdpa"
    ).to(device)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["Wqkv", "Wo"], modules_to_save=["classifier"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_ds = RowDataset(DATA_DIR / "train.jsonl", tokenizer)
    val_ds = RowDataset(DATA_DIR / "val.jsonl", tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    # bf16 doesn't need loss scaling; GradScaler is fp16-only (it would otherwise raise
    # "Attempting to unscale FP16 gradients" style errors when params/grads aren't fp16).
    scaler = torch.amp.GradScaler("cuda", enabled=not USE_BF16)

    best_val_acc = 0.0
    epochs_no_improve = 0
    CKPT_DIR.mkdir(exist_ok=True)

    t_start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.autocast("cuda", dtype=AMP_DTYPE):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = focal_loss(logits, labels) / GRAD_ACCUM

            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={running_loss / len(train_loader):.4f} val_acc={val_acc:.4f} "
              f"elapsed={time.time() - t_start:.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            model.save_pretrained(CKPT_DIR / "best")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"early stopping at epoch {epoch}, best val_acc={best_val_acc:.4f}")
                break

    print(f"peak GPU memory (MB): {torch.cuda.max_memory_allocated() / 1e6:.1f}")
    print(f"total train wall-clock (s): {time.time() - t_start:.0f}")
    print(f"best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
