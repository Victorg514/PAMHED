
import pandas as pd
import numpy as np
import random

# --- CONFIGURATION ---
DATA_FILE = "data/final.parquet"
# List of key features that should show variation from tweet to tweet
FEATURES_TO_CHECK = [
    'len_z',
    'sent_compound_z',
    'circadian_deviation',
    'anomaly_score',
    'tpd_7d_dev',
    'emotion_volatility_10tw'
]
# How many unique values a feature should have to be considered "dynamic"
UNIQUENESS_THRESHOLD = 0.5 # Expect at least 50% of values to be unique

def validate_parquet(file_path: str):
    """
    Loads the processed parquet file and runs a series of checks to validate
    the quality and dynamism of the behavioral features.
    """
    print(f"--- Validating Processed Data File: {file_path} ---")
    
    # 1. Load Data
    try:
        df = pd.read_parquet(file_path)
        print(f"✓ Successfully loaded {len(df):,} rows.")
    except Exception as e:
        print(f"✗ FAILED to load parquet file: {e}")
        return

    # 2. Select a Random User from the Test Set
    test_users = df[df['split'] == 'test']['user_id'].unique().tolist()
    if not test_users:
        print("✗ FAILED: No users found in the 'test' split.")
        return
        
    target_user_id = random.choice(test_users)
    user_df = df[df['user_id'] == target_user_id].copy()
    
    print(f"\n--- Running checks on a random test user: {target_user_id} ---")
    print(f"User has {len(user_df)} tweets in the processed dataset.")
    
    if len(user_df) < 2:
        print("✗ WARNING: Selected user has fewer than 2 tweets. Cannot check for variation. Please re-run.")
        return
        
    # 3. Perform Validation Checks
    all_checks_passed = True
    
    print("\n--- Checking for Dynamic Feature Variation ---")
    for feature in FEATURES_TO_CHECK:
        if feature not in user_df.columns:
            print(f"  ? SKIPPED: Feature '{feature}' not found in the dataframe.")
            continue
            
        feature_series = user_df[feature]
        
        # Check 1: Are all values identical?
        num_unique = feature_series.nunique()
        uniqueness_ratio = num_unique / len(feature_series)
        
        print(f"  - Checking feature: '{feature}'...")
        if num_unique == 1:
            print(f"    ✗ FAILED: All values for '{feature}' are identical ({feature_series.iloc[0]:.4f}). This indicates a static calculation.")
            all_checks_passed = False
        elif uniqueness_ratio < UNIQUENESS_THRESHOLD:
             print(f"    - WARNING: Feature '{feature}' has low variation. Only {num_unique} unique values ({uniqueness_ratio:.1%}).")
        else:
            print(f"    ✓ PASSED: Feature '{feature}' is dynamic with {num_unique} unique values.")

    # 4. Check for NaN or Infinite values
    print("\n--- Checking for Invalid Numerical Values ---")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    nan_count = df[numeric_cols].isnull().sum().sum()
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    
    if nan_count > 0:
        print(f"✗ FAILED: Found {nan_count} NaN (null) values in numerical columns.")
        all_checks_passed = False
    else:
        print("✓ PASSED: No NaN values found.")
        
    if inf_count > 0:
        print(f"✗ FAILED: Found {inf_count} infinite values in numerical columns.")
        all_checks_passed = False
    else:
        print("✓ PASSED: No infinite values found.")
        
    # 5. Final Summary
    print("\n" + "="*50)
    if all_checks_passed:
        print("✅ VALIDATION SUCCESSFUL ✅")
        print("The features appear to be correctly calculated and dynamic.")
    else:
        print("❌ VALIDATION FAILED ❌")
        print("One or more checks failed. Please review the errors above before re-training.")
    print("="*50)
    
    # 6. Display a sample of the data for manual inspection
    print("\n--- Sample of processed data for the user ---")
    display_cols = ['text'] + [f for f in FEATURES_TO_CHECK if f in user_df.columns]
    print(user_df[display_cols].head(20))


if __name__ == "__main__":
    validate_parquet(DATA_FILE)