import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, RobertaModel, AutoConfig
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# --- MODEL CLASS ---
class AdvancedDeltaModel(nn.Module):
    def __init__(self, base_name="mental/mental-roberta-base", num_labels=2, num_features=50, dropout_rate=0.3):
        super().__init__(); self.num_labels = num_labels; self.base = RobertaModel.from_pretrained(base_name); hidden_size = self.base.config.hidden_size; self.feature_encoder = nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(dropout_rate)); self.fusion_gate = nn.Sequential(nn.Linear(hidden_size + 64, 64), nn.ReLU(), nn.Linear(64, 64), nn.Sigmoid()); self.classifier = nn.Linear(hidden_size + 64, num_labels)
    def forward(self, input_ids=None, attention_mask=None, delta=None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask); text_features = outputs.last_hidden_state[:, 0]; behavioral_features_encoded = self.feature_encoder(delta.float()); gate_input = torch.cat([text_features, behavioral_features_encoded], dim=1); gate_values = self.fusion_gate(gate_input); gated_behavioral_features = behavioral_features_encoded * gate_values; final_combined_features = torch.cat([text_features, gated_behavioral_features], dim=1); logits = self.classifier(final_combined_features)
        return {"logits": logits}

# --- RECONSTRUCT TRAINING FEATURES ---
def get_training_feature_list(training_data_path, n_features, positive_class):
    """
    Re-runs feature selection on Twitter data to get the exact list and order of features 
    the model expects.
    """
    print("Re-calculating feature list from training data to ensure alignment...")
    df = pd.read_parquet(training_data_path)
    train_df = df[df['split'] == 'train']
    
    LABEL_MAP = {"control": 0, "depression": 1, "anxiety": 2, "bipolar": 3}
    if positive_class == 'mental_health':
        binary_y = train_df['label'].apply(lambda x: 0 if x == LABEL_MAP['control'] else 1).values
    else:
        pid = LABEL_MAP[positive_class]
        binary_y = train_df['label'].apply(lambda x: 1 if x == pid else 0).values

    exclude_cols = ['tweet_id', 'user_id', 'text', 'label', 'label_name', 'split', 'created', 'binary_label']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and train_df[c].dtype in ['float64', 'int64']]
    
    X = train_df[feature_cols].fillna(0).values
    mi_scores = mutual_info_classif(X, binary_y, random_state=42)
    feature_scores = pd.DataFrame({'feature': feature_cols, 'score': mi_scores}).sort_values('score', ascending=False)
    
    selected = feature_scores.head(n_features)['feature'].tolist()
    print(f"Identified {len(selected)} features used in training: {selected}")
    return selected

def main(args):
    POSITIVE_CLASS_NAME = args.positive_class
    print(f"\n--- Evaluating Twitter-Trained '{POSITIVE_CLASS_NAME}' Model on eRisk Reddit Data ---")
    DECISION_THRESHOLD = 0.95 
    ERISK_FILE = "data/erisk_processed_for_testing.parquet"
    TWITTER_FILE = "data/final.parquet" 
    TOKENIZER_NAME = "mental/mental-roberta-base"
    MODEL_PATH = f"../out/delta_binary_{POSITIVE_CLASS_NAME}_manual"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model Config
    print(f"Loading model config from: {MODEL_PATH}")
    weights_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    if not os.path.exists(weights_path): raise FileNotFoundError(f"Missing weights at {weights_path}")
    
    # Introspect weight shape to get N features
    state_dict = torch.load(weights_path, map_location='cpu')
    num_features_trained = state_dict['feature_encoder.0.weight'].shape[1]
    print(f"Model expects exactly {num_features_trained} features.")

    # 2. Get the Ordered Feature List
    # We MUST use the same columns in the same order as training
    selected_features = get_training_feature_list(TWITTER_FILE, num_features_trained, POSITIVE_CLASS_NAME)
    
    # 3. Load eRisk Data and Verify Columns
    full_df = pd.read_parquet(ERISK_FILE)
    depressed_df = full_df[full_df['label'] == 1]
    control_df = full_df[full_df['label'] == 0]
    
    print(f"Original Count -> Depressed: {len(depressed_df)}, Control: {len(control_df)}")
    
    # Sample controls to match the number of depressed users
    control_df_balanced = control_df.sample(n=len(depressed_df), random_state=42)
    
    # Combine and shuffle
    test_df = pd.concat([depressed_df, control_df_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced Count -> Depressed: {len(test_df[test_df['label']==1])}, Control: {len(test_df[test_df['label']==0])}")
    print(f"Total Test Set Size: {len(test_df)}")
    
    # Ensure all selected features exist in eRisk data (fill 0 if missing)
    for feat in selected_features:
        if feat not in test_df.columns:
            print(f"Warning: Feature '{feat}' missing in eRisk data. Filling with 0.")
            test_df[feat] = 0.0
            
    # 4. Initialize Model
    model = AdvancedDeltaModel(base_name=TOKENIZER_NAME, num_labels=2, num_features=num_features_trained).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Tokenize and Batch
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    def tokenize_with_features(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        # Construct delta vector using the strict ordered list
        encoding["delta"] = [[float(batch[c][i]) for c in selected_features] for i in range(len(batch["text"]))]
        return encoding

    test_ds = Dataset.from_pandas(test_df).map(
        tokenize_with_features, 
        batched=True, 
        remove_columns=test_df.columns.tolist()
    )
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'delta'])
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=32)
    
    # 6. Predict
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)['logits']
            # Convert logits to probabilities (0.0 to 1.0)
            probs = torch.softmax(logits, dim=1)[:, 1] # Probability of class 1 (Depression)
            all_probs.append(probs.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    
    # --- Apply Custom Threshold ---
    final_preds = (all_probs > DECISION_THRESHOLD).astype(int)
    true_labels = test_df['label'].values
    
    print("\n" + "="*70)
    print("--- FINAL CROSS-PLATFORM PERFORMANCE REPORT (eRisk) ---")
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