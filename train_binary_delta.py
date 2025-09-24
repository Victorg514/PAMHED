

import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    RobertaModel,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
import pathlib
import warnings
warnings.filterwarnings('ignore')

# --- THE DEFINITIVE, STABLE GATED MODEL ---
class AdvancedDeltaModel(nn.Module):
    def __init__(self, base_name="mental/mental-roberta-base", num_labels=2, num_features=50, dropout_rate=0.3):
        super().__init__()
        self.num_labels = num_labels
        self.base = RobertaModel.from_pretrained(base_name)
        hidden_size = self.base.config.hidden_size
        
        # Stream 1: Behavioral Feature Encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # The "Gate": A small network that learns how much to trust the behavioral features
        # It takes both text and behavior as input to make its decision
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid() # Sigmoid squashes the output between 0 and 1
        )
        
        # Final classifier only takes the text features and the gated behavioral features
        self.classifier = nn.Linear(hidden_size + 64, num_labels)

    def forward(self, input_ids=None, attention_mask=None, delta=None, labels=None):
        # Get stable [CLS] token from RoBERTa
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state[:, 0]
        
        # Get encoded behavioral features
        behavioral_features_encoded = self.feature_encoder(delta.float())
        
        # --- GATED FUSION LOGIC ---
        # 1. Create a combined vector to feed into the gate
        gate_input = torch.cat([text_features, behavioral_features_encoded], dim=1)
        
        # 2. Calculate the gate values (a vector of numbers between 0 and 1)
        gate_values = self.fusion_gate(gate_input)
        
        # 3. Apply the gate to the behavioral features
        gated_behavioral_features = behavioral_features_encoded * gate_values
        # -------------------------

        # Concatenate the original text features with the SAFE, gated behavioral features
        final_combined_features = torch.cat([text_features, gated_behavioral_features], dim=1)
        
        logits = self.classifier(final_combined_features)
        
        return logits
# ----------------------------------------------------


class WeightedDeltaTrainer(Trainer):
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs); self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels"); logits = model(**inputs)
        loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
        return (loss, {"logits": logits}) if return_outputs else loss
    
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
    DATA_FILE = pathlib.Path("data/final.parquet")
    MODEL_NAME = "mental/mental-roberta-base"
    OUT_DIR = f"out/delta_binary_{POSITIVE_CLASS_NAME}2"
    MAX_LEN, BATCH, EPOCHS, SEED = 128, 16, 5, 42
    LEARNING_RATE, WARMUP_RATIO, WEIGHT_DECAY = 3e-5, 0.1, 0.05
    MAX_FEATURES_TO_SELECT = 50
    
    torch.manual_seed(SEED); np.random.seed(SEED); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); df = pd.read_parquet(DATA_FILE);
    LABEL_MAP = {"control": 0, "depression": 1, "anxiety": 2, "bipolar": 3}
    if POSITIVE_CLASS_NAME == 'mental_health': df['binary_label'] = df['label'].apply(lambda x: 0 if x == LABEL_MAP['control'] else 1)
    else: positive_label_id = LABEL_MAP[POSITIVE_CLASS_NAME]; df['binary_label'] = df['label'].apply(lambda x: 1 if x == positive_label_id else 0)
    train_df = df[df['split'] == 'train']; class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df['binary_label'])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device); print(f"Binary class weights: {class_weights}")
    selected_features = select_best_features(train_df, n_features=MAX_FEATURES_TO_SELECT, binary_y=train_df['binary_label'].values)
    actual_num_features = len(selected_features)
    def create_dataset(split): return Dataset.from_pandas(df[df['split'] == split].reset_index(drop=True))
    train_ds, val_ds = create_dataset("train"), create_dataset("val"); tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_with_features(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        encoding["delta"] = [[float(batch[c][i]) for c in selected_features] for i in range(len(batch["text"]))]
        encoding["labels"] = batch["binary_label"]; return encoding
    train_ds = train_ds.map(tokenize_with_features, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize_with_features, batched=True, remove_columns=val_ds.column_names)
    
    # Instantiate the new, stable model
    model = AdvancedDeltaModel(base_name=MODEL_NAME, num_labels=2, num_features=actual_num_features, dropout_rate=0.4).to(device)
    
    # Use the universal compute_metrics
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions['logits']; labels = eval_pred.label_ids
        preds = np.argmax(logits, axis=1); f1 = f1_score(labels, preds, average='binary', zero_division=0)
        return {"f1": f1}

    # --- THIS IS THE FINAL, STABLE TrainingArguments BLOCK ---
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        
        learning_rate=3e-5, # A safe and effective learning rate
        warmup_ratio=0.1,
        weight_decay=0.05,
        
        
        fp16=False,                  # Disable mixed-precision to prevent underflow
        max_grad_norm=1.0,           # Add gradient clipping as a safeguard
        lr_scheduler_type="cosine_with_restarts", # Use a more stable scheduler
        

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )
    # ----------------------------------------------------
    
    trainer = WeightedDeltaTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights_tensor
    )
    
    trainer.train()
    trainer.save_model(OUT_DIR)
    print(f"--- Finished training and saved model to {OUT_DIR} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_class", type=str, required=True, choices=['mental_health', 'depression', 'anxiety', 'bipolar'], help="The class to treat as the positive label (1).")
    args = parser.parse_args()
    main(args)