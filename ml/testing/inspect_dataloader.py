

import argparse
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.feature_selection import mutual_info_classif
import pathlib
import warnings
warnings.filterwarnings('ignore')

def select_best_features(df, n_features, binary_y):
    """This function must be identical to the one in your training script."""
    exclude_cols = ['tweet_id', 'user_id', 'text', 'label', 'label_name', 'split', 'created', 'binary_label']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
    if not feature_cols: return []
    X = df[feature_cols].fillna(0).values; y = binary_y
    mi_scores = mutual_info_classif(X, y, random_state=42)
    feature_scores = pd.DataFrame({'feature': feature_cols, 'score': mi_scores}).sort_values('score', ascending=False)
    selected = feature_scores.head(n_features)['feature'].tolist()
    return selected

def main(args):
    POSITIVE_CLASS_NAME = args.positive_class
    print(f"\n--- Inspecting DataLoader for: {POSITIVE_CLASS_NAME.upper()} vs. OTHERS ---")

    # --- Use the same configuration as the training script ---
    DATA_FILE = pathlib.Path("data/final.parquet")
    MODEL_NAME = "mental/mental-roberta-base"
    MAX_LEN, BATCH = 128, 16
    MAX_FEATURES_TO_SELECT = 50
    
    # --- Load and prepare data exactly as in the training script ---
    df = pd.read_parquet(DATA_FILE)
    LABEL_MAP = {"control": 0, "depression": 1, "anxiety": 2, "bipolar": 3}
    if POSITIVE_CLASS_NAME == 'mental_health':
        df['binary_label'] = df['label'].apply(lambda x: 0 if x == LABEL_MAP['control'] else 1)
    else:
        positive_label_id = LABEL_MAP[POSITIVE_CLASS_NAME]
        df['binary_label'] = df['label'].apply(lambda x: 1 if x == positive_label_id else 0)
    
    train_df = df[df['split'] == 'train']
    selected_features = select_best_features(train_df, n_features=MAX_FEATURES_TO_SELECT, binary_y=train_df['binary_label'].values)
    
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Use the bulletproof tokenizer from the final script ---
    def tokenize_with_features_safe(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
        deltas = []
        for i in range(len(batch["text"])):
            feature_vector = []
            for col_name in selected_features:
                value = batch[col_name][i]
                if value is None or np.isnan(value) or np.isinf(value):
                    feature_vector.append(0.0)
                else:
                    feature_vector.append(float(value))
            deltas.append(feature_vector)
        encoding["delta"] = deltas
        encoding["labels"] = batch["binary_label"]
        return encoding
    
    tokenized_train_ds = train_ds.map(tokenize_with_features_safe, batched=True, remove_columns=train_ds.column_names)
    tokenized_train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'delta'])
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(tokenized_train_ds, batch_size=BATCH)

    # --- The Inspection Loop ---
    print(f"\nInspecting the first 100 batches from the DataLoader...")
    
    found_issue = False
    for i, batch in enumerate(dataloader):
        if i >= 100: # We only need to check a few batches
            break

        delta_tensor = batch['delta']

        # Check 1: NaN values
        if torch.isnan(delta_tensor).any():
            print(f"!!!!!! CRITICAL ERROR in Batch {i}: NaN value detected! !!!!!!")
            found_issue = True
            break
            
        # Check 2: Infinite values
        if torch.isinf(delta_tensor).any():
            print(f"!!!!!! CRITICAL ERROR in Batch {i}: Infinite value detected! !!!!!!")
            found_issue = True
            break
            
        # Check 3: Check the range of the data
        if i % 20 == 0: # Print stats every 20 batches
            print(f"  Batch {i}: OK. Delta tensor shape: {delta_tensor.shape}. Min: {delta_tensor.min():.2f}, Max: {delta_tensor.max():.2f}, Mean: {delta_tensor.mean():.2f}")

    print("\n--- Inspection Complete ---")
    if not found_issue:
        print("✓ No NaN or Infinite values were found in the first 100 batches.")
        print("The data being passed to the model appears to be clean.")
    else:
        print("✗ An issue was found in the data pipeline. Please check the error message above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--positive_class", type=str, required=True,
        choices=['mental_health', 'depression', 'anxiety', 'bipolar'],
        help="The binary task to inspect."
    )
    args = parser.parse_args()
    main(args)