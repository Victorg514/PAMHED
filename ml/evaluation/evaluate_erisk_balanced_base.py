

import argparse
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

def main(args):
    POSITIVE_CLASS_NAME = args.positive_class
    print(f"\n--- Evaluating Twitter-Trained BASELINE '{POSITIVE_CLASS_NAME}' Model on eRisk Reddit Data ---")
    
    # --- Configuration ---
    DECISION_THRESHOLD = 0.95 
    ERISK_FILE = "../../data/erisk_processed_for_testing.parquet"
    MODEL_PATH = os.path.abspath(f"../../out/baseline_binary_{POSITIVE_CLASS_NAME}_manual")
    
    print(f"Loading baseline model from LOCAL path: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if the folder exists
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f" The model directory was not found at: {MODEL_PATH}\n Please check your training script output folder.")
        
    # Check if config.json exists
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        raise FileNotFoundError(f" config.json not found in {MODEL_PATH}. The model may not have saved correctly.")

    try:
        # Load from the verified local path
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model architecture. Error: {e}")
    
    
    
    # 1. Load the Baseline Model
    # AutoModel handles safetensors automatically!
    print(f"Loading baseline model from: {MODEL_PATH}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    except Exception as e:
        raise FileNotFoundError(f"Could not load model from {MODEL_PATH}. Error: {e}")
        
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH) # Load tokenizer from the same saved folder

    # 2. Load and Balance eRisk Data
    full_df = pd.read_parquet(ERISK_FILE)
    depressed_df = full_df[full_df['label'] == 1]
    control_df = full_df[full_df['label'] == 0]
    
    print(f"Original Count -> Depressed: {len(depressed_df)}, Control: {len(control_df)}")
    
    # Sample controls to match the number of depressed users (50/50 split)
    control_df_balanced = control_df.sample(n=len(depressed_df), random_state=42)
    test_df = pd.concat([depressed_df, control_df_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced Count -> Depressed: {len(test_df[test_df['label']==1])}, Control: {len(test_df[test_df['label']==0])}")
    print(f"Total Test Set Size: {len(test_df)}")

    # 3. Tokenize (Text Only)
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True, remove_columns=test_df.columns.tolist())
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=64)
    
    # 4. Predict
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Baseline"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            # Convert logits to probabilities (0.0 to 1.0)
            probs = torch.softmax(logits, dim=1)[:, 1] # Probability of class 1 (Depression)
            all_probs.append(probs.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    
    # --- Apply Threshold ---
    final_preds = (all_probs > DECISION_THRESHOLD).astype(int)
    true_labels = test_df['label'].values
    
    print("\n" + "="*70)
    print("--- FINAL CROSS-PLATFORM REPORT ---")
    print("="*70)
    print(classification_report(
        true_labels,
        final_preds,
        target_names=["control", "depression"],
        digits=4
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_class", type=str, required=True, choices=['mental_health', 'depression'])
    args = parser.parse_args()
    main(args)