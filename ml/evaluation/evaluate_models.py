import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from tqdm import tqdm
import pathlib
import warnings

from transformers import (
    AutoTokenizer,
    RobertaModel,
    AutoModelForSequenceClassification,
    AutoConfig,
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_FILE    = "data/final.parquet" 
BASELINE_DIR = "out/baseline_final2"
DELTA_DIR    = "out/delta_final2" 
MODEL_NAME   = "mental/mental-roberta-base"
NUM_LABELS   = 4
MAX_FEATURES_TO_SELECT = 50

# Restored Caching System
CACHE = {
    "proba_plain": "cache/proba_plain_4class.npy",
    "proba_delta": "cache/proba_delta_4class.npy",
    "pred_plain":  "cache/pred_plain_4class.npy",
    "pred_delta":  "cache/pred_delta_4class.npy",
    "labels":      "cache/labels_4class.npy"
}

# --- THE CORRECT DELTA MODEL CLASS ---
# This MUST be identical to the class used in your train_delta_final_v3.py script.
class AdvancedDeltaModel(nn.Module):
    def __init__(self, base_name="mental/mental-roberta-base", num_labels=4, num_features=50, dropout_rate=0.5):
        super().__init__(); self.num_labels = num_labels; self.base = RobertaModel.from_pretrained(base_name); hidden_size = self.base.config.hidden_size; self.feature_encoder = nn.Sequential(nn.Linear(num_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()); self.text_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=dropout_rate, batch_first=True); self.fusion_gate = nn.Sequential(nn.Linear(hidden_size + 64, hidden_size + 64), nn.Sigmoid()); self.pre_classifier = nn.Linear(hidden_size + 64, hidden_size); self.classifier_dropout = nn.Dropout(dropout_rate); self.classifier = nn.Linear(hidden_size, num_labels); self.feature_importance = nn.Parameter(torch.tensor(0.5))
    def forward(self, input_ids=None, attention_mask=None, delta=None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask); sequence_output = outputs.last_hidden_state; attended_output, _ = self.text_attention(query=sequence_output,key=sequence_output,value=sequence_output,key_padding_mask=~attention_mask.bool() if attention_mask is not None else None); cls_token = attended_output[:, 0]; mean_pooled = attended_output.mean(dim=1); text_features = self.feature_importance * cls_token + (1 - self.feature_importance) * mean_pooled; behavioral_features = self.feature_encoder(delta.float()); combined_features = torch.cat([text_features, behavioral_features], dim=1); gate_values = self.fusion_gate(combined_features); fused_features = combined_features * gate_values; pooled_output = self.pre_classifier(fused_features); dropped_output = self.classifier_dropout(pooled_output); logits = self.classifier(dropped_output)
        return {"logits": logits}

# --- DATA & PREDICTION FUNCTIONS ---

def get_feature_list(df, n_features):
    """
    This function now just gets the deterministic list of top N features.
    """
    exclude_cols = ['tweet_id', 'user_id', 'text', 'label', 'label_name', 'split', 'created']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
    if not feature_cols: return []
    X = df[feature_cols].fillna(0).values; y = df['label'].values
    mi_scores = mutual_info_classif(X, y, random_state=42)
    feature_scores = pd.DataFrame({'feature': feature_cols, 'score': mi_scores}).sort_values('score', ascending=False)
    top_features = feature_scores.head(n_features)['feature'].tolist()
    return top_features

def load_all():
    """
    This is the main function your notebook calls.
    It runs inference or loads from cache, then returns a dictionary
    with probabilities, predictions, and labels.
    """
    os.makedirs("cache", exist_ok=True)
    
    # Check if cached predictions exist
    # Comment out for new dataset
    """
    if all(os.path.exists(CACHE[f]) for f in CACHE):
        print("--- Loading all predictions from cache ---")
        return {
            "proba_plain": np.load(CACHE["proba_plain"]),
            "proba_delta": np.load(CACHE["proba_delta"]),
            "pred_plain":  np.load(CACHE["pred_plain"]),
            "pred_delta":  np.load(CACHE["pred_delta"]),
            "labels":      np.load(CACHE["labels"])
        }
    """
    print("--- Running Inference (Cache not found) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(DATA_FILE)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # --- Baseline Prediction ---
    print("\nEvaluating Baseline Model...")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(BASELINE_DIR).to(device)
    baseline_model.eval()
    
    def tok_only(batch): return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    plain_ds = Dataset.from_pandas(test_df).map(
        tok_only, 
        batched=True, 
        # Use test_df.columns.tolist() instead of test_df.column_names
        remove_columns=test_df.columns.tolist()
    )
    plain_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    from torch.utils.data import DataLoader
    plain_loader = DataLoader(plain_ds, batch_size=64)
    all_baseline_logits = []
    with torch.no_grad():
        for batch in tqdm(plain_loader, desc="Predicting with Baseline Model"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = baseline_model(**batch).logits
            all_baseline_logits.append(logits.cpu().numpy())
    proba_plain = np.concatenate(all_baseline_logits)
    
    # --- Delta Prediction ---
    print("\nEvaluating Delta Model...")
    # Load the config file first
    config = AutoConfig.from_pretrained(DELTA_DIR)
    # Get the custom parameters from the config
    num_features_trained = config.custom_num_features
    dropout_rate = config.custom_dropout_rate
    print(f"Model was trained with {num_features_trained} features. Replicating setup.")

    # Get the EXACT SAME list of N features that was used in training.
    selected_features = get_feature_list(df, n_features=num_features_trained)
    
    delta_model = AdvancedDeltaModel(
        base_name=config._name_or_path, 
        num_labels=config.num_labels, 
        num_features=num_features_trained,
        dropout_rate=dropout_rate
    ).to(device)
    
    # --- Loading logic ---
    safetensors_path = os.path.join(DELTA_DIR, "model.safetensors")
    pytorch_bin_path = os.path.join(DELTA_DIR, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        print(f"Loading weights from: {safetensors_path}")
        state_dict = load_file(safetensors_path, device=str(device))
    elif os.path.exists(pytorch_bin_path):
        print(f"Loading weights from: {pytorch_bin_path}")
        state_dict = torch.load(pytorch_bin_path, map_location=device)
    else:
        raise FileNotFoundError(f"Could not find model weights ('model.safetensors' or 'pytorch_model.bin') in {DELTA_DIR}")
        
    delta_model.load_state_dict(state_dict)
    delta_model.eval()

    def tok_plus(batch):
        encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
        encoding["delta"] = [[float(batch[c][i]) for c in selected_features] for i in range(len(batch["text"]))]
        return encoding

    delta_ds = Dataset.from_pandas(test_df).map(
        tok_plus, 
        batched=True,
        # Use test_df.columns.tolist() again here
        remove_columns=test_df.columns.tolist()
    )
    delta_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'delta'])
    
    delta_loader = DataLoader(delta_ds, batch_size=32)
    all_delta_logits = []
    with torch.no_grad():
        for batch in tqdm(delta_loader, desc="Predicting with Delta Model"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = delta_model(**batch)['logits']
            all_delta_logits.append(logits.cpu().numpy())
    proba_delta = np.concatenate(all_delta_logits)

    # --- Process and Cache All Results ---
    labels = test_df['label'].values
    pred_plain = proba_plain.argmax(axis=1)
    pred_delta = proba_delta.argmax(axis=1)

    print("\nCaching predictions for future runs...")
    np.save(CACHE["proba_plain"], proba_plain)
    np.save(CACHE["proba_delta"], proba_delta)
    np.save(CACHE["labels"], labels)
    np.save(CACHE["pred_plain"], pred_plain)
    np.save(CACHE["pred_delta"], pred_delta)

    return {
        "proba_plain": proba_plain,
        "proba_delta": proba_delta,
        "pred_plain":  pred_plain,
        "pred_delta":  pred_delta,
        "labels":      labels
    }

def main():
    """When run as a script, this function produces the detailed text reports."""
    
    # Run predictions (uses cache if available)
    d = load_all()
    
    # Unpack the dictionary for clarity
    labels = d["labels"]
    pred_plain = d["pred_plain"]
    pred_delta = d["pred_delta"]
    class_names = ["control", "depression", "anxiety", "bipolar"]
    
    # --- Generate All Reports ---
    print("\n" + "="*70)
    print("BASELINE MODEL RESULTS (4-CLASS):")
    print("="*70)
    print(classification_report(labels, pred_plain, target_names=class_names, digits=4))
    
    print("\n" + "="*70)
    print("PERSONALIZED (DELTA) MODEL RESULTS (4-CLASS):")
    print("="*70)
    print(classification_report(labels, pred_delta, target_names=class_names, digits=4))
    
    # --- Detailed Comparison (Restored from your original script) ---
    acc_plain, prec_plain, rec_plain, f1_plain = (
        accuracy_score(labels, pred_plain),
        *precision_recall_fscore_support(labels, pred_plain, average='macro')[:3]
    )
    acc_delta, prec_delta, rec_delta, f1_delta = (
        accuracy_score(labels, pred_delta),
        *precision_recall_fscore_support(labels, pred_delta, average='macro')[:3]
    )
    
    print("\n" + "="*70)
    print("SUMMARY COMPARISON:")
    print("="*70)
    print(f"{'Metric':<20} {'Baseline':>12} {'Delta':>12} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Accuracy':<20} {acc_plain:>12.4f} {acc_delta:>12.4f} {acc_delta - acc_plain:>+15.4f}")
    print(f"{'Macro Precision':<20} {prec_plain:>12.4f} {prec_delta:>12.4f} {prec_delta - prec_plain:>+15.4f}")
    print(f"{'Macro Recall':<20} {rec_plain:>12.4f} {rec_delta:>12.4f} {rec_delta - rec_plain:>+15.4f}")
    print(f"{'Macro F1':<20} {f1_plain:>12.4f} {f1_delta:>12.4f} {f1_delta - f1_plain:>+15.4f}")
    
    f1_plain_per_class = precision_recall_fscore_support(labels, pred_plain, average=None, zero_division=0)[2]
    f1_delta_per_class = precision_recall_fscore_support(labels, pred_delta, average=None, zero_division=0)[2]
    
    print("\n" + "="*70)
    print("PER-CLASS F1 SCORE COMPARISON:")
    print("="*70)
    print(f"{'Class':<15} {'Baseline':>12} {'Delta':>12} {'Improvement':>15}")
    print("-" * 70)
    for i, class_name in enumerate(class_names):
        improvement = f1_delta_per_class[i] - f1_plain_per_class[i]
        print(f"{class_name:<15} {f1_plain_per_class[i]:>12.4f} {f1_delta_per_class[i]:>12.4f} {improvement:>+15.4f}")
    
    print("\n" + "="*70)
    print("CONFUSION MATRICES:")
    print("="*70)
    print("\nBaseline Model:")
    cm_plain = confusion_matrix(labels, pred_plain)
    print(pd.DataFrame(cm_plain, index=class_names, columns=class_names))
    
    print("\nPersonalized (Delta) Model:")
    cm_delta = confusion_matrix(labels, pred_delta)
    print(pd.DataFrame(cm_delta, index=class_names, columns=class_names))

if __name__ == "__main__":
    main()