import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, RobertaModel, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
import pathlib
import warnings
from tqdm import tqdm
import os
import json

warnings.filterwarnings('ignore')

# Use stable gated model architecture 
class AdvancedDeltaModel(nn.Module):
   
    def __init__(self, base_name="mental/mental-roberta-base", num_labels=2, num_features=50, dropout_rate=0.3):
        super().__init__(); self.num_labels = num_labels; self.base = RobertaModel.from_pretrained(base_name); hidden_size = self.base.config.hidden_size; self.feature_encoder = nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(dropout_rate)); self.fusion_gate = nn.Sequential(nn.Linear(hidden_size + 64, 64), nn.ReLU(), nn.Linear(64, 64), nn.Sigmoid()); self.classifier = nn.Linear(hidden_size + 64, num_labels)
    def forward(self, input_ids=None, attention_mask=None, delta=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask); text_features = outputs.last_hidden_state[:, 0]; behavioral_features_encoded = self.feature_encoder(delta.float()); gate_input = torch.cat([text_features, behavioral_features_encoded], dim=1); gate_values = self.fusion_gate(gate_input); gated_behavioral_features = behavioral_features_encoded * gate_values; final_combined_features = torch.cat([text_features, gated_behavioral_features], dim=1); logits = self.classifier(final_combined_features)
        return logits

def select_best_features(df, n_features, binary_y):
    """
    Selects features, prioritizing the powerful deviation and z-scored columns.
    """
    exclude_cols = ['tweet_id', 'user_id', 'text', 'label', 'label_name', 'split', 'created', 'binary_label']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
    
    if not feature_cols:
        print("CRITICAL WARNING: No numerical feature columns found!")
        return []

    # Get the data needed for calculation
    X = df[feature_cols].fillna(0).values
    y = binary_y
    
    mi_scores = mutual_info_classif(X, y, random_state=42)

    feature_scores = pd.DataFrame({'feature': feature_cols, 'score': mi_scores}).sort_values('score', ascending=False)
    
    selected = feature_scores.head(n_features)['feature'].tolist()
    
    print(f"Found {len(feature_cols)} total numerical features.")
    print(f"Selected {len(selected)} best features for the binary task.")
    return selected


def main(args):
    POSITIVE_CLASS_NAME = args.positive_class
    print(f"\n--- MANUAL PyTorch Training for DELTA (with Stratified Sampling): {POSITIVE_CLASS_NAME.upper()} ---")

    # --- Configuration (Identical to Baseline Manual Script) ---
    DATA_FILE = pathlib.Path("data/final.parquet")
    MODEL_NAME = "mental/mental-roberta-base"
    OUT_DIR = f"out/delta_binary_{POSITIVE_CLASS_NAME}_manual"
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    MAX_LEN, BATCH, EPOCHS, SEED = 128, 16, 5, 42
    LEARNING_RATE, WEIGHT_DECAY = 3e-5, 0.01
    MAX_FEATURES_TO_SELECT = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)
    
    # --- Load and Relabel Data ---
    df = pd.read_parquet(DATA_FILE)
    LABEL_MAP = {"control": 0, "depression": 1, "anxiety": 2, "bipolar": 3}
    if POSITIVE_CLASS_NAME == 'mental_health':
        df['binary_label'] = df['label'].apply(lambda x: 0 if x == LABEL_MAP['control'] else 1)
    else:
        positive_label_id = LABEL_MAP[POSITIVE_CLASS_NAME]
        df['binary_label'] = df['label'].apply(lambda x: 1 if x == positive_label_id else 0)

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df['binary_label'])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Binary class weights: {class_weights}")
    
    selected_features = select_best_features(train_df, n_features=MAX_FEATURES_TO_SELECT, binary_y=train_df['binary_label'].values)
    actual_num_features = len(selected_features)
    
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_with_features_safe(batch):
        # (tokenizer is the same as before)
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        deltas = [[float(batch[c][i]) if pd.notna(batch[c][i]) and np.isfinite(batch[c][i]) else 0.0 for c in selected_features] for i in range(len(batch["text"]))]
        encoding["delta"] = deltas; encoding["labels"] = batch["binary_label"]
        return encoding
        
    tokenized_train_ds = train_ds.map(tokenize_with_features_safe, batched=True, remove_columns=train_ds.column_names)
    tokenized_val_ds = val_ds.map(tokenize_with_features_safe, batched=True, remove_columns=val_ds.column_names)
    tokenized_train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'delta'])
    tokenized_val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'delta'])
    
    # --- ADD THE STRATIFIED SAMPLER (Identical to Baseline) ---
    train_labels_np = np.array(train_df['binary_label'])
    class_sample_count = np.array([len(np.where(train_labels_np == t)[0]) for t in np.unique(train_labels_np)])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in train_labels_np])).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_dataloader = DataLoader(tokenized_train_ds, batch_size=BATCH, sampler=sampler)
    eval_dataloader = DataLoader(tokenized_val_ds, batch_size=BATCH)
    # -----------------------------------------------------------

    # --- Manual PyTorch Training Setup ---
    # Use the stable, gated model
    model = AdvancedDeltaModel(base_name=MODEL_NAME, num_labels=2, num_features=actual_num_features, dropout_rate=0.4).to(device)
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
            # (Training step logic is the same)
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device)
            delta = batch['delta'].to(device); labels = batch['labels'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, delta=delta)
            loss = loss_fn(logits, labels)
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
            # (Evaluation step logic is the same)
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                logits = model(**inputs)
            predictions = torch.argmax(logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        eval_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        print(f"\n--- Epoch {epoch+1} Evaluation ---")
        print(f"Validation F1 Score: {eval_f1:.4f}")

        # --- Save Best Model ---
        if eval_f1 > best_eval_f1:
            best_eval_f1 = eval_f1
            print(f"New best model found! Saving to {OUT_DIR}")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "pytorch_model.bin"))
            config_dict = {
                "_name_or_path": MODEL_NAME, "num_labels": 2, 
                "custom_num_features": actual_num_features, "custom_dropout_rate": 0.4
            }
            with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
                json.dump(config_dict, f)

    print("\n--- Manual Training Complete ---")
    print(f"Best validation F1 score: {best_eval_f1:.4f}")
    print(f"Best model saved in {OUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_class", type=str, required=True, choices=['mental_health', 'depression', 'anxiety', 'bipolar'])
    args = parser.parse_args()
    main(args)