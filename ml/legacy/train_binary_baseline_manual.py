
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, RobertaModel, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
import pathlib
import warnings
from tqdm.auto import tqdm
import os
import json

warnings.filterwarnings('ignore')

def main(args):
    POSITIVE_CLASS_NAME = args.positive_class
    print(f"\n--- MANUAL PyTorch Training for BINARY BASELINE (with Stratified Sampling): {POSITIVE_CLASS_NAME.upper()} ---")

    # --- Configuration ---
    DATA_FILE = pathlib.Path("data/final.parquet")
    MODEL_NAME = "mental/mental-roberta-base"
    OUT_DIR = f"out/baseline_binary_{POSITIVE_CLASS_NAME}_manual"
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    MAX_LEN, BATCH, EPOCHS, SEED = 128, 16, 5, 42
    LEARNING_RATE, WEIGHT_DECAY = 3e-5, 0.01 
    DATA_FRACTION = 1
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load and Relabel Data ---
    df = pd.read_parquet(DATA_FILE)
    df = df[["text", "label", "label_name", "split"]]

    LABEL_MAP = {"control": 0, "depression": 1, "anxiety": 2, "bipolar": 3}
    if POSITIVE_CLASS_NAME == 'mental_health':
        df['binary_label'] = df['label'].apply(lambda x: 0 if x == LABEL_MAP['control'] else 1)
    else:
        positive_label_id = LABEL_MAP[POSITIVE_CLASS_NAME]
        df['binary_label'] = df['label'].apply(lambda x: 1 if x == positive_label_id else 0)

    train_df = df[df['split'] == 'train'].sample(frac=DATA_FRACTION, random_state=SEED)
    val_df = df[df['split'] == 'val'].sample(frac=DATA_FRACTION, random_state=SEED)

    # Class weights are still useful as an extra signal to the loss function
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df['binary_label'])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Binary class weights: {class_weights}")

    # --- Create Datasets & Tokenize ---
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        encoding["labels"] = batch["binary_label"]
        return encoding
    
    tokenized_train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    tokenized_val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)
    tokenized_train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # --- THIS IS THE DEFINITIVE FIX: Stratified Batching ---
    # 1. Calculate weights for each individual sample in the training set.
    train_labels_np = np.array(train_df['binary_label'])
    class_sample_count = np.array([len(np.where(train_labels_np == t)[0]) for t in np.unique(train_labels_np)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_labels_np])
    samples_weight = torch.from_numpy(samples_weight).double()

    # 2. Create the sampler which will yield balanced batches.
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # 3. Create the DataLoader using this sampler.
    # CRITICAL: When using a sampler, you MUST set shuffle=False. The sampler handles the shuffling.
    train_dataloader = DataLoader(tokenized_train_ds, batch_size=BATCH, sampler=sampler)
    # -----------------------------------------------------------------
    
    # The evaluation dataloader should not be sampled; we want to evaluate on the true distribution.
    eval_dataloader = DataLoader(tokenized_val_ds, batch_size=BATCH)

    # --- Manual PyTorch Training Setup ---
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(num_training_steps * 0.1), num_training_steps=num_training_steps)
    
    best_eval_f1 = 0.0
    
    print(f"\n--- Starting Manual Training for {EPOCHS} Epochs ---")
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(EPOCHS):
        # --- Training Loop ---
        model.train()
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Use the correct, external weighted loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

        # --- Evaluation Loop ---
        model.eval()
        all_preds, all_labels = [], []
        for batch in eval_dataloader:
            # Pop labels before passing to model to be safe
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        eval_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        print(f"\n--- Epoch {epoch+1} Evaluation ---")
        print(f"Validation F1 Score: {eval_f1:.4f}")

        # --- Save Best Model ---
        if eval_f1 > best_eval_f1:
            best_eval_f1 = eval_f1
            print(f"New best model found! Saving to {OUT_DIR}")
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)

    print("\n--- Manual Training Complete ---")
    print(f"Best validation F1 score: {best_eval_f1:.4f}")
    print(f"Best model saved in {OUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_class", type=str, required=True, choices=['mental_health', 'depression', 'anxiety', 'bipolar'])
    args = parser.parse_args()
    main(args)