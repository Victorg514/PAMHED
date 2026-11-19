import argparse
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
import warnings

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from evaluate_binary_models import AdvancedDeltaModel, get_feature_list

warnings.filterwarnings('ignore')

# Picks the decision threshold on the VALIDATION split, then applies it to the cached TEST logits
# produced by evaluate_binary_models.py. Never tune on test.

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "final.parquet")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
MODEL_NAME = "mental/mental-roberta-base"
LABEL_MAP = {"control": 0, "depression": 1, "anxiety": 2, "bipolar": 3}
EXCLUDE_COLS = ['tweet_id', 'user_id', 'text', 'label', 'label_name', 'split', 'created', 'binary_label']


def binarize(labels, positive_class_name):
    if positive_class_name == 'mental_health':
        return (labels != LABEL_MAP['control']).astype(int)
    return (labels == LABEL_MAP[positive_class_name]).astype(int)


def numeric_feature_cols():
    # Same candidate set and order as get_feature_list() sees on the full DataFrame
    schema = pq.read_schema(DATA_FILE)
    return [f.name for f in schema if f.name not in EXCLUDE_COLS and str(f.type) in ('double', 'int64')]


def load_split(split, columns):
    # Reading only the needed columns avoids loading the full parquet into memory
    return pq.read_table(DATA_FILE, columns=columns, filters=[('split', '==', split)]).to_pandas()


def get_selected_features(positive_class_name, num_features):
    feature_cols = numeric_feature_cols()
    train_df = load_split('train', feature_cols + ['label'])
    train_df['binary_label'] = binarize(train_df['label'].values, positive_class_name)
    return get_feature_list(train_df, n_features=num_features, binary_y=train_df['binary_label'].values)


def predict_logits(model, tokenizer, texts, deltas=None, batch_size=64, desc=""):
    device = next(model.parameters()).device
    all_logits = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=desc):
            enc = tokenizer(texts[start:start + batch_size], truncation=True, padding=True,
                            max_length=128, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in enc.items()}
            if deltas is None:
                logits = model(**inputs).logits
            else:
                delta = torch.tensor(deltas[start:start + batch_size], dtype=torch.float32, device=device)
                logits = model(**inputs, delta=delta)
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits)


def get_val_logits(positive_class_name, sample=None):
    suffix = f"_n{sample}" if sample else ""
    cache = {
        "plain":  os.path.join(CACHE_DIR, f"val_logits_plain_binary_{positive_class_name}{suffix}.npy"),
        "delta":  os.path.join(CACHE_DIR, f"val_logits_delta_binary_{positive_class_name}{suffix}.npy"),
        "labels": os.path.join(CACHE_DIR, f"val_labels_binary_{positive_class_name}{suffix}.npy"),
    }
    if all(os.path.exists(p) for p in cache.values()):
        print(f"--- Loading validation logits for '{positive_class_name}' from cache ---")
        return {k: np.load(p) for k, p in cache.items()}

    import json
    delta_dir = os.path.join(PROJECT_ROOT, "out", f"delta_binary_{positive_class_name}_manual")
    baseline_dir = os.path.join(PROJECT_ROOT, "out", f"baseline_binary_{positive_class_name}_manual")
    with open(os.path.join(delta_dir, 'config.json')) as f:
        num_features = json.load(f).get('custom_num_features', 50)

    selected_features = get_selected_features(positive_class_name, num_features)
    print(f"Selected {len(selected_features)} features: {selected_features}")

    val_df = load_split('val', ['text', 'label'] + selected_features)
    if sample and sample < len(val_df):
        # Random rows, not head(): rows are grouped by user, so head() would cover only a few users
        val_df = val_df.sample(n=sample, random_state=42)
    texts = val_df['text'].tolist()
    deltas = val_df[selected_features].fillna(0).values.astype(np.float32)
    labels = binarize(val_df['label'].values, positive_class_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    baseline_model = AutoModelForSequenceClassification.from_pretrained(baseline_dir).to(device).eval()
    plain = predict_logits(baseline_model, tokenizer, texts, desc=f"Val baseline ({positive_class_name})")
    del baseline_model

    delta_model = AdvancedDeltaModel(base_name=MODEL_NAME, num_labels=2, num_features=len(selected_features)).to(device)
    delta_model.load_state_dict(torch.load(os.path.join(delta_dir, "pytorch_model.bin"), map_location=device))
    delta_model.eval()
    delta = predict_logits(delta_model, tokenizer, texts, deltas=deltas, desc=f"Val delta ({positive_class_name})")

    out = {"plain": plain, "delta": delta, "labels": labels}
    for k, p in cache.items():
        np.save(p, out[k])
    return out


def positive_proba(logits):
    # Softmax over 2 classes == sigmoid of the logit difference
    return 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))


def best_threshold(y, p, average):
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y, (p >= t).astype(int), average=average, zero_division=0) for t in thresholds]
    i = int(np.argmax(scores))
    return thresholds[i], scores[i]


def main(args):
    pos = args.positive_class
    val = get_val_logits(pos, sample=args.sample)

    test_labels = np.load(os.path.join(CACHE_DIR, f"labels_binary_{pos}.npy"))
    test_logits = {m: np.load(os.path.join(CACHE_DIR, f"proba_{m}_binary_{pos}.npy")) for m in ("plain", "delta")}
    target_names = ["control", "mental_health"] if pos == 'mental_health' else ["other", pos]

    rows = []
    for m in ("plain", "delta"):
        p_val = positive_proba(val[m])
        p_test = positive_proba(test_logits[m])
        t_macro, _ = best_threshold(val["labels"], p_val, 'macro')
        t_pos, _ = best_threshold(val["labels"], p_val, 'binary')

        for name, t in (("0.50 (default)", 0.5), ("val-tuned macro", t_macro), ("val-tuned pos-F1", t_pos)):
            pred = (p_test >= t).astype(int)
            rows.append({
                "model": m, "threshold": name, "t": round(float(t), 2),
                "test_pos_F1": f1_score(test_labels, pred, average='binary', zero_division=0),
                "test_macro_F1": f1_score(test_labels, pred, average='macro', zero_division=0),
                "pred_pos_rate": pred.mean(),
            })

        pred = (p_test >= t_macro).astype(int)
        print(f"\n--- {m.upper()} model, test set, threshold {t_macro:.2f} (tuned for macro F1 on val) ---")
        print(classification_report(test_labels, pred, target_names=target_names, digits=4))
        print(pd.DataFrame(confusion_matrix(test_labels, pred), index=target_names, columns=target_names))

    print(f"\nTrue positive rate on test: {test_labels.mean():.2f}")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_class", type=str, required=True, choices=['mental_health', 'depression', 'anxiety', 'bipolar'])
    parser.add_argument("--sample", type=int, default=10000, help="Random val rows to use (0 = full val split). CPU inference is ~3s per 64 rows per model.")
    args = parser.parse_args()
    main(args)
