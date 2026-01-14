"""
Final delta training script with:
- All advanced deviation features
- Attention mechanism for feature fusion
- Proper class weights
- Early stopping and learning rate scheduling
- Feature selection capability
"""
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import mutual_info_classif
import pathlib
import warnings
warnings.filterwarnings('ignore')

# The AdvancedDeltaModel class is unchanged.
class AdvancedDeltaModel(nn.Module):
    def __init__(self, base_name="mental/mental-roberta-base", num_labels=4, num_features=50, dropout_rate=0.3):
        super().__init__()
        self.num_labels = num_labels
        self.base = RobertaModel.from_pretrained(base_name)
        hidden_size = self.base.config.hidden_size
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.text_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=dropout_rate, batch_first=True)
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_size + 64, hidden_size + 64), nn.Sigmoid())
        self.pre_classifier = nn.Linear(hidden_size + 64, hidden_size)
        self.classifier_dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.feature_importance = nn.Parameter(torch.tensor(0.5))
    def forward(self, input_ids=None, attention_mask=None, delta=None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask); sequence_output = outputs.last_hidden_state
        attended_output, _ = self.text_attention(query=sequence_output,key=sequence_output,value=sequence_output,key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        cls_token = attended_output[:, 0]; mean_pooled = attended_output.mean(dim=1)
        text_features = self.feature_importance * cls_token + (1 - self.feature_importance) * mean_pooled
        behavioral_features = self.feature_encoder(delta.float())
        combined_features = torch.cat([text_features, behavioral_features], dim=1)
        gate_values = self.fusion_gate(combined_features); fused_features = combined_features * gate_values
        pooled_output = self.pre_classifier(fused_features); dropped_output = self.classifier_dropout(pooled_output)
        logits = self.classifier(dropped_output); loss = None
        if labels is not None: loss = nn.functional.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), weight=getattr(self, 'class_weights', None))
        return {"loss": loss, "logits": logits}


def select_best_features(df, n_features=50):
    # Ensure all necessary columns exist to be excluded
    exclude_cols = ['tweet_id', 'user_id', 'text', 'label', 'label_name', 'split', 'created']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
    
    if not feature_cols:
        print("CRITICAL WARNING: No numerical feature columns found!")
        return []
        
    train_df = df[df['split'] == 'train'].copy()
    X = train_df[feature_cols].fillna(0).values; y = train_df['label'].values
    mi_scores = mutual_info_classif(X, y, random_state=42)
    feature_scores = pd.DataFrame({'feature': feature_cols, 'score': mi_scores}).sort_values('score', ascending=False)
    
    # Select the top N features, or fewer if not enough are available
    top_features = feature_scores.head(n_features)['feature'].tolist()
    
    print(f"Found {len(feature_cols)} total numerical features.")
    print(f"Selected {len(top_features)} best features.")
    return top_features

def main():
    # --- Configuration ---
    DATA_FILE = pathlib.Path("data/final.parquet")
    WEIGHTS_FILE = pathlib.Path("data/class_weights.npy")
    MODEL_NAME = "mental/mental-roberta-base"
    OUT_DIR = "out/delta_final2"
    MAX_LEN, BATCH, EPOCHS, SEED = 128, 16, 5, 42
    MAX_FEATURES_TO_SELECT = 50 # Keep this as the desired maximum
    LEARNING_RATE, WARMUP_RATIO = 2e-5, 0.1

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df = pd.read_parquet(DATA_FILE)

    # --- FIX: Step 1 - Select features FIRST ---
    selected_features = select_best_features(df, n_features=MAX_FEATURES_TO_SELECT)
    # --- FIX: Step 2 - Get the ACTUAL number of features found ---
    actual_num_features = len(selected_features)

    if actual_num_features == 0:
        raise ValueError("No features were selected. Halting training.")

    # Load class weights
    class_weights = np.load(WEIGHTS_FILE)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Create Datasets
    def create_dataset(split): return Dataset.from_pandas(df[df['split'] == split].reset_index(drop=True))
    train_ds, val_ds, test_ds = create_dataset("train"), create_dataset("val"), create_dataset("test")

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_with_features(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        # Use the globally defined selected_features list
        encoding["delta"] = [[float(batch[c][i] if pd.notna(batch[c][i]) else 0.0) for c in selected_features] for i in range(len(batch["text"]))]
        encoding["labels"] = batch["label"]
        return encoding
    
    train_ds = train_ds.map(tokenize_with_features, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize_with_features, batched=True, remove_columns=val_ds.column_names)
    test_ds = test_ds.map(tokenize_with_features, batched=True, remove_columns=test_ds.column_names)

    # --- FIX: Step 3 - Initialize the model with the DYNAMIC feature count ---
    model = AdvancedDeltaModel(
        base_name=MODEL_NAME, 
        num_labels=4, 
        num_features=actual_num_features, # Use the actual count
        dropout_rate=0.4
    ).to(device)
    model.class_weights = class_weights_tensor

    # Metrics
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        
        predictions = np.argmax(predictions, axis=1)
        f1 = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)[2]
        return {"macro_f1": f1}

    # Hardened training args
    training_args = TrainingArguments(
        output_dir=OUT_DIR, overwrite_output_dir=True, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=2, learning_rate=LEARNING_RATE, warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        logging_strategy="steps", logging_steps=200, fp16=False,
        report_to="none", seed=SEED,
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    test_results = trainer.evaluate(eval_dataset=test_ds)
    print("Delta Model Test Results:", test_results)

if __name__ == "__main__":
    main()