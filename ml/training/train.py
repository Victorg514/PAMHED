"""
One training script for every task and feature set (v2 pipeline).

  python ml/training/train.py --task mental_health --features none      # text-only model
  python ml/training/train.py --task mental_health --features default   # text + behavioural features
  python ml/training/train.py --task depression --features default,-timing --seed 43

Compared to train_binary_*_manual.py:
- Text-only and delta models share the same architecture, data and training loop; the text-only
  model is the same network with no feature stream, so differences come from the features alone.
- Explicit feature list (ml/features.py), scaled with train-split statistics, saved in config.json.
- Class imbalance is corrected once (weighted loss by default), not twice.
- Best epoch chosen by validation ROC-AUC (threshold-free); decision thresholds tuned on validation.
- bf16 mixed precision and dynamic padding; early stopping.
- Writes val/test predictions and metrics.json for ml/evaluation/summarize.py.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, RobertaModel, get_scheduler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, f1_score
import functools
from tqdm.auto import tqdm
if not sys.stderr.isatty():  # writing to a log file: update once a minute so the log stays small
    tqdm = functools.partial(tqdm, mininterval=60)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from features import (PROJECT_ROOT, resolve_features, task_labels, num_labels_for, load_split,
                      fit_scaler, apply_scaler, TASKS)
from evaluation.metrics import evaluate_predictions

MODEL_NAME = "mental/mental-roberta-base"


class FusionModel(nn.Module):
    """
    Gated fusion of RoBERTa's [CLS] vector with a behavioural feature vector (same design as the
    v1 AdvancedDeltaModel). With num_features=0 it is a plain text classifier on [CLS].
    """
    def __init__(self, base_name=MODEL_NAME, num_labels=2, num_features=0, dropout_rate=0.3, feature_dim=64):
        super().__init__()
        self.num_features = num_features
        self.base = RobertaModel.from_pretrained(base_name, add_pooling_layer=False)
        hidden = self.base.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        if num_features > 0:
            self.feature_encoder = nn.Sequential(nn.Linear(num_features, feature_dim), nn.ReLU(), nn.Dropout(dropout_rate))
            self.fusion_gate = nn.Sequential(nn.Linear(hidden + feature_dim, feature_dim), nn.ReLU(),
                                             nn.Linear(feature_dim, feature_dim), nn.Sigmoid())
            self.classifier = nn.Linear(hidden + feature_dim, num_labels)
        else:
            self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, delta=None):
        text = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        text = self.dropout(text)
        if self.num_features == 0:
            return self.classifier(text)
        encoded = self.feature_encoder(delta.float())
        gated = encoded * self.fusion_gate(torch.cat([text, encoded], dim=1))
        return self.classifier(torch.cat([text, gated], dim=1))


class TweetDataset(Dataset):
    def __init__(self, texts, feats, labels):
        self.texts, self.feats, self.labels = texts, feats, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.feats[i], self.labels[i]


def make_collate(tokenizer, max_len):
    def collate(batch):
        texts, feats, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["delta"] = torch.from_numpy(np.stack(feats))
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate


def predict(model, loader, device, amp_dtype):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["delta"].to(device))
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probs)


def selection_score(y, proba):
    """Threshold-free model-selection metric: ROC-AUC (binary) or macro F1 (4-class)."""
    if proba.shape[1] == 2:
        return roc_auc_score(y, proba[:, 1])
    return f1_score(y, proba.argmax(1), average="macro")


def prepare(split, task, features, limit, seed):
    df = load_split(split, features)
    if limit:
        # Sample whole users, alternating between labels, so every class appears in smoke tests
        users = df[["user_id", "label"]].drop_duplicates("user_id").sample(frac=1, random_state=seed)
        users = users.assign(rank=users.groupby("label").cumcount()).sort_values("rank", kind="stable")["user_id"]
        keep, total = [], 0
        sizes = df.groupby("user_id").size()
        for u in users:
            keep.append(u); total += sizes[u]
            if total >= limit: break
        df = df[df["user_id"].isin(keep)]
    df = df.reset_index(drop=True)
    y = task_labels(task, df["label"].to_numpy())
    return df, y


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if amp_dtype == torch.float16:
        print("Warning: GPU has no bf16 support; fp16 without a GradScaler can underflow. Consider --no_amp.")

    features = resolve_features(args.features)
    num_labels = num_labels_for(args.task)
    run_name = args.name or f"{args.task}__{args.features.replace(',', '+')}__s{args.seed}"
    out_dir = pathlib.Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {run_name} | device={device} amp={amp_dtype} | {len(features)} features: {features}")

    # --- Data ---
    train_df, y_train = prepare("train", args.task, features, args.limit, args.seed)
    val_df, y_val = prepare("val", args.task, features, args.limit and args.limit // 3, args.seed)
    test_df, y_test = prepare("test", args.task, features, args.limit and args.limit // 3, args.seed)
    scaler = fit_scaler(train_df, features)
    x_train, x_val, x_test = (apply_scaler(d, scaler) for d in (train_df, val_df, test_df))
    print(f"Rows: train={len(train_df)} val={len(val_df)} test={len(test_df)} | "
          f"positive rate (train)={np.mean(y_train == 1):.3f}" if num_labels == 2 else
          f"Rows: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = make_collate(tokenizer, args.max_len)
    train_ds = TweetDataset(train_df["text"].tolist(), x_train, y_train)
    eval_loader = lambda df, x, y: DataLoader(TweetDataset(df["text"].tolist(), x, y), batch_size=args.batch * 4,
                                              collate_fn=collate, num_workers=0)
    val_loader, test_loader = eval_loader(val_df, x_val, y_val), eval_loader(test_df, x_test, y_test)

    # --- Class imbalance: corrected ONCE ---
    classes = np.arange(num_labels)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    loss_weights = torch.tensor(class_weights, dtype=torch.float32, device=device) if args.balance == "loss" else None
    if args.balance == "sampler":
        sample_w = class_weights[y_train]
        sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double), len(sample_w),
                                        generator=torch.Generator().manual_seed(args.seed))
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, collate_fn=collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                                  generator=torch.Generator().manual_seed(args.seed))
    print(f"Balancing: {args.balance} | class weights {np.round(class_weights, 3).tolist()}")

    # --- Model ---
    model = FusionModel(MODEL_NAME, num_labels, len(features), args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * total_steps),
                              num_training_steps=total_steps)

    config = {
        "run_name": run_name, "task": args.task, "num_labels": num_labels, "base_model": MODEL_NAME,
        "feature_spec": args.features, **scaler, "dropout": args.dropout, "max_len": args.max_len,
        "args": vars(args), "torch": torch.__version__,
    }
    best_score, best_epoch, bad_epochs, log = -np.inf, -1, 0, []
    weights_path = out_dir / "model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(bar):
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["delta"].to(device))
            loss = loss_fn(logits.float(), batch["labels"].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}; rerun with --no_amp")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                bar.set_postfix(loss=f"{loss.item():.4f}")

        val_proba = predict(model, val_loader, device, amp_dtype)
        score = selection_score(y_val, val_proba)
        entry = {"epoch": epoch, "val_selection_score": score, "minutes": (time.time() - t0) / 60}
        log.append(entry)
        print(f"Epoch {epoch}: val {'ROC-AUC' if num_labels == 2 else 'macro F1'} = {score:.4f} "
              f"({entry['minutes']:.1f} min)")

        # The first epoch is always kept, so a NaN score (validation has one class) cannot leave no model
        if best_epoch < 0 or score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save(model.state_dict(), weights_path)
        else:
            bad_epochs += 1
            if bad_epochs > args.patience:
                print(f"Early stopping: no improvement for {bad_epochs} epoch(s).")
                break

    # --- Final predictions with the best epoch ---
    model.load_state_dict(torch.load(weights_path, map_location=device))
    preds = {}
    for split, loader, df, y in (("val", val_loader, val_df, y_val), ("test", test_loader, test_df, y_test)):
        proba = predict(model, loader, device, amp_dtype)
        pred_df = pd.DataFrame({"tweet_id": df["tweet_id"], "user_id": df["user_id"], "y": y})
        for i in range(num_labels):
            pred_df[f"p{i}"] = proba[:, i]
        pred_df.to_parquet(out_dir / f"pred_{split}.parquet", index=False)
        preds[split] = pred_df

    metrics = evaluate_predictions(preds["val"], preds["test"], num_labels, n_boot=args.n_boot)
    metrics.update({"best_epoch": best_epoch, "best_val_selection_score": best_score, "epochs": log})
    config.update({"best_epoch": best_epoch})
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    if not args.save_model:
        os.remove(weights_path)
    t, u = metrics["test_tweet"], metrics["test_user"]
    print(f"\nTest (tweet): macro F1 {t['macro_f1']:.4f}" + (f" | ROC-AUC {t['roc_auc']:.4f}" if num_labels == 2 else ""))
    print(f"Test (user):  macro F1 {u['macro_f1']:.4f}" + (f" | ROC-AUC {u['roc_auc']:.4f}" if num_labels == 2 else ""))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--features", default="default", help="'none' (text only), 'default', 'all', or groups, e.g. 'default,-timing'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=1, help="Epochs without val improvement before stopping")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--balance", choices=["loss", "sampler", "none"], default="loss")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Smoke test: ~N train rows (whole users)")
    p.add_argument("--n_boot", type=int, default=1000, help="Bootstrap resamples for CIs (0 = skip)")
    p.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out_root", default=str(PROJECT_ROOT / "out" / "v2"))
    p.add_argument("--name", default=None)
    main(p.parse_args())
