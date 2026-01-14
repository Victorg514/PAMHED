"""
Final baseline training script with:
- Proper class weights from data analysis
- Early stopping
- More epochs
- Learning rate scheduling
- Comprehensive evaluation metrics
"""

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import pandas as pd
import numpy as np
import pathlib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

def main():
    # --- Configuration ---
    # UPDATED: Point to the final, correct data file
    DATA_FILE = pathlib.Path("data/final.parquet")
    WEIGHTS_FILE = pathlib.Path("data/class_weights.npy")
    MODEL_NAME = "mental/mental-roberta-base"
    OUT_DIR = "out/baseline_final2"
    MAX_LEN = 128
    BATCH = 16
    EPOCHS = 5
    NUM_LABELS = 4
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    SEED = 42

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    df = pd.read_parquet(DATA_FILE)
    df = df[["text", "label", "label_name", "split", "user_id", "tweet_id"]]

    # Load Class Weights
    class_weights = np.load(WEIGHTS_FILE)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Using class weights: {class_weights_tensor.cpu().numpy()}")

    # Create Datasets
    def make_dataset(split):
        return Dataset.from_pandas(df[df["split"] == split].reset_index(drop=True))
    train_ds, val_ds, test_ds = make_dataset("train"), make_dataset("val"), make_dataset("test")

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
    
    train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=["text", "label_name", "split", "user_id", "tweet_id"])
    val_ds = val_ds.map(tokenize_function, batched=True, remove_columns=["text", "label_name", "split", "user_id", "tweet_id"])
    test_ds = test_ds.map(tokenize_function, batched=True, remove_columns=["text", "label_name", "split", "user_id", "tweet_id"])
    train_ds, val_ds, test_ds = train_ds.rename_column("label", "labels"), val_ds.rename_column("label", "labels"), test_ds.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2,).to(device)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred; predictions = np.argmax(predictions, axis=1)
        f1 = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)[2]
        return {"macro_f1": f1}

    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = nn.functional.cross_entropy(logits, labels, weight=class_weights_tensor)
            return (loss, outputs) if return_outputs else loss

    # Hardened Training Arguments for stability
    training_args = TrainingArguments(
        output_dir=OUT_DIR, overwrite_output_dir=True, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH, # Keep eval batch size same as train
        gradient_accumulation_steps=2, learning_rate=LEARNING_RATE, warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        logging_strategy="steps", logging_steps=200, fp16=False, # Keep FP16 off for stability
        report_to="none", seed=SEED,
        label_smoothing_factor=0.1,
    )

    trainer = WeightedLossTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    test_results = trainer.evaluate(eval_dataset=test_ds)
    print("Baseline Test Results:", test_results)

if __name__ == "__main__":
    main()