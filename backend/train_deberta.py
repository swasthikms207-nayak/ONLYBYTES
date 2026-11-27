"""
train_deberta.py – Fine-tune microsoft/deberta-v3-small as a multi-label
classifier on /data/text_dataset.csv (train.csv / val.csv / test.csv).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "deberta-ambiguity"
BASE_MODEL = "microsoft/deberta-v3-small"

LABELS = [
    "lexical",
    "syntactic",
    "referential",
    "scope",
    "semantic",
    "pragmatic",
    "underspecified",
    "ellipsis",
    "temporal_numeric",
    "cultural_domain",
]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}


class AmbiguityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: Any, max_len: int = 96) -> None:
        self.texts = df["text"].fillna("").astype(str).tolist()
        self.contexts = df["context"].fillna("").astype(str).tolist()
        self.labels = []
        for raw in df["types"].fillna("[]"):
            try:
                t_list = json.loads(raw) if isinstance(raw, str) else list(raw)
            except Exception:
                t_list = [x.strip() for x in str(raw).split(",") if x.strip()]
            vec = np.zeros(len(LABELS), dtype=np.float32)
            for t in t_list:
                if t in LABEL2ID:
                    vec[LABEL2ID[t]] = 1.0
            self.labels.append(vec)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        ctx = self.contexts[idx]
        enc = self.tokenizer(
            text,
            ctx if ctx else None,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def _eval_split(model: Any, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds, all_gold = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            all_preds.append((probs >= 0.5).astype(int))
            all_gold.append(labels.astype(int))
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_gold)
    return {
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def train(epochs: int = 2, batch_size: int = 32, lr: float = 5e-5, max_len: int = 48) -> dict[str, Any]:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id=LABEL2ID,
        torch_dtype=torch.float32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # On CPU, freeze the DeBERTa backbone and fine-tune pooler + multi-label classifier head
    # (pre-extracting DeBERTa hidden states makes 25 epochs run in ~1s after extraction).
    if device.type == "cpu":
        for name, param in model.named_parameters():
            if name.startswith("deberta."):
                param.requires_grad = False
        epochs = max(epochs, 80)
        lr = 1e-3

    train_loader = DataLoader(AmbiguityDataset(train_df, tokenizer, max_len=max_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AmbiguityDataset(val_df, tokenizer, max_len=max_len), batch_size=batch_size)
    test_loader = DataLoader(AmbiguityDataset(test_df, tokenizer, max_len=max_len), batch_size=batch_size)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.full((len(LABELS),), 4.0, device=device))

    if device.type == "cpu":
        # Fast CPU path: extract frozen DeBERTa CLS hidden states once, then train pooler+classifier
        def _extract_cls(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
            ds = AmbiguityDataset(df, tokenizer, max_len=max_len)
            loader = DataLoader(ds, batch_size=batch_size)
            feats, lbls = [], []
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    lbls.append(batch.pop("labels"))
                    out = model.deberta(**batch).last_hidden_state[:, 0, :]
                    feats.append(out)
            return torch.cat(feats, dim=0), torch.cat(lbls, dim=0)

        X_tr, y_tr = _extract_cls(train_df)
        X_va, y_va = _extract_cls(val_df)
        X_te, y_te = _extract_cls(test_df)

        for epoch in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(X_tr.size(0))
            total_loss = 0.0
            for i in range(0, X_tr.size(0), batch_size):
                idx = perm[i : i + batch_size]
                optimizer.zero_grad()
                pooled = model.pooler(X_tr[idx].unsqueeze(1)) if hasattr(model, "pooler") else X_tr[idx]
                logits = model.classifier(pooled)
                loss = loss_fn(logits, y_tr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                total_loss += float(loss.item())

            model.eval()
            with torch.no_grad():
                va_pooled = model.pooler(X_va.unsqueeze(1)) if hasattr(model, "pooler") else X_va
                va_probs = torch.sigmoid(model.classifier(va_pooled)).numpy()
                va_pred = (va_probs >= 0.5).astype(int)
                y_va_np = y_va.numpy().astype(int)
                val_metrics = {
                    "micro_f1": round(float(f1_score(y_va_np, va_pred, average="micro", zero_division=0)), 4),
                    "macro_f1": round(float(f1_score(y_va_np, va_pred, average="macro", zero_division=0)), 4),
                }
            if epoch % 5 == 0 or epoch == epochs:
                print(f"Epoch {epoch}/{epochs} | loss={total_loss:.4f} | val={val_metrics}")

        model.eval()
        with torch.no_grad():
            te_pooled = model.pooler(X_te.unsqueeze(1)) if hasattr(model, "pooler") else X_te
            te_probs = torch.sigmoid(model.classifier(te_pooled)).numpy()
            te_pred = (te_probs >= 0.5).astype(int)
            y_te_np = y_te.numpy().astype(int)
            test_metrics = {
                "micro_f1": round(float(f1_score(y_te_np, te_pred, average="micro", zero_division=0)), 4),
                "macro_f1": round(float(f1_score(y_te_np, te_pred, average="macro", zero_division=0)), 4),
            }
    else:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            val_metrics = _eval_split(model, val_loader, device)
            print(f"Epoch {epoch}/{epochs} | loss={total_loss/len(train_loader):.4f} | val={val_metrics}")

        test_metrics = _eval_split(model, test_loader, device)
    print(f"Test metrics: {test_metrics}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    meta = {
        "base_model": BASE_MODEL,
        "labels": LABELS,
        "epochs": epochs,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved fine-tuned model to {MODEL_DIR}")
    return meta


if __name__ == "__main__":
    train()
