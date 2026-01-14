"""
Debug script to identify issues with training
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=" * 70)
print("DEBUGGING TRAINING ISSUES")
print("=" * 70)

# 1. Check data file
print("\n1. CHECKING DATA FILE:")
print("-" * 40)
try:
    df = pd.read_parquet("data/deviation_advanced_final.parquet")
    print(f"✓ Data loaded: {len(df):,} samples")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Splits: {df['split'].value_counts().to_dict()}")
    
    # Check for data issues
    print(f"  Null text: {df['text'].isna().sum()}")
    print(f"  Empty text: {(df['text'].str.len() == 0).sum()}")
    print(f"  Max text length: {df['text'].str.len().max()}")
    
    # Check label distribution
    print("\n  Label distribution:")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        counts = split_df['label'].value_counts().sort_index()
        print(f"    {split}: {counts.to_dict()}")
        
except Exception as e:
    print(f"✗ Error loading data: {e}")
    print("  Run merge_blip.py first!")

# 2. Check class weights
print("\n2. CHECKING CLASS WEIGHTS:")
print("-" * 40)
try:
    weights = np.load("data/class_weights.npy")
    print(f"✓ Class weights loaded: {weights}")
    
    # Check if weights are reasonable
    if weights.min() <= 0:
        print("✗ WARNING: Negative or zero weights detected!")
    if weights.max() / weights.min() > 10:
        print("✗ WARNING: Extreme weight imbalance (ratio > 10)!")
        
except Exception as e:
    print(f"✗ Error loading weights: {e}")
    print("  Computing balanced weights...")
    
    from sklearn.utils.class_weight import compute_class_weight
    train_df = df[df['split'] == 'train']
    classes = np.unique(train_df['label'])
    weights = compute_class_weight('balanced', classes=classes, y=train_df['label'])
    print(f"  Computed weights: {weights}")
    np.save("data/class_weights.npy", weights)

# 3. Check model loading
print("\n3. CHECKING MODEL LOADING:")
print("-" * 40)

# Try both models
models_to_test = [
    "roberta-base",
    "mental/mental-roberta-base"
]

for model_name in models_to_test:
    print(f"\nTesting {model_name}:")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=4,
            ignore_mismatched_sizes=True  # Important for mental-roberta
        )
        print(f"  ✓ Model loaded successfully")
        print(f"    Vocab size: {tokenizer.vocab_size}")
        print(f"    Max length: {tokenizer.model_max_length}")
        
        # Test tokenization
        test_text = "I feel anxious and depressed today"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"    Test tokenization: {tokens['input_ids'].shape}")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(**tokens)
            print(f"    Forward pass shape: {outputs.logits.shape}")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

# 4. Check for common issues
print("\n4. CHECKING COMMON ISSUES:")
print("-" * 40)

# Check GPU memory
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
else:
    print("✗ No GPU available")

# Check sample data
print("\n5. SAMPLE DATA CHECK:")
print("-" * 40)
train_df = df[df['split'] == 'train'].head(5)
for idx, row in train_df.iterrows():
    print(f"Sample {idx}:")
    print(f"  Label: {row['label']} ({row['label_name']})")
    print(f"  Text length: {len(row['text'])}")
    print(f"  Text preview: {row['text'][:100]}...")
    print()

# 6. Recommendations
print("\n6. RECOMMENDATIONS:")
print("-" * 40)

# Check if using mental-roberta would help
if 'mental' not in models_to_test[0]:
    print("• Consider using 'mental/mental-roberta-base' instead of 'roberta-base'")

# Check for class imbalance
train_df = df[df['split'] == 'train']
label_counts = train_df['label'].value_counts()
if label_counts.max() / label_counts.min() > 3:
    print("• Severe class imbalance detected. Consider:")
    print("  - Oversampling minority classes")
    print("  - Using focal loss instead of cross-entropy")
    print("  - Adjusting class weights further")

# Check for feature issues
feature_cols = [c for c in df.columns if c not in ['text', 'label', 'label_name', 'split', 'user_id', 'tweet_id', 'created_at']] # updated to include created_at
if len(feature_cols) > 0:
    print(f"• Found {len(feature_cols)} feature columns")
    
    # Check for non-numeric, NaN, or infinite values
    numeric_feature_data = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    
    non_numeric_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"  ✗ WARNING: Found non-numeric columns: {non_numeric_cols}")

    nan_cols = numeric_feature_data.columns[numeric_feature_data.isna().any()].tolist()
    if nan_cols:
        print(f"  ✗ Columns with NaN/Nulls: {nan_cols[:10]}...")
    
    inf_cols = numeric_feature_data.columns[np.isinf(numeric_feature_data.values).any(axis=0)].tolist()
    if inf_cols:
        print(f"  ✗ Columns with Inf values: {inf_cols[:5]}...")

print("\n" + "=" * 70)
print("DEBUGGING COMPLETE")
print("=" * 70)