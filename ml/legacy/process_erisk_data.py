import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib
from sklearn.preprocessing import StandardScaler
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import warnings
warnings.filterwarnings('ignore')

def _clean_text(txt: str) -> str:
    return re.sub(r"http\S+|@\w+", "", str(txt)).strip()

def generate_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    MIN_POSTS = 10
    BASE_FRAC = 0.40
    
    user_counts = df["user_id"].value_counts()
    df = df[df["user_id"].isin(user_counts[user_counts >= MIN_POSTS].index)]
    print(f"Filtered to {df['user_id'].nunique()} eRisk users with >= {MIN_POSTS} posts.")
    
    sia = SentimentIntensityAnalyzer()

    def _process_user_timeline(group):
        n_base = max(5, int(len(group) * BASE_FRAC))
        baseline_df = group.head(n_base)
        later_df = group.iloc[n_base:].copy()

        if later_df.empty: return pd.DataFrame()

        base_text_lens = baseline_df['text'].str.len().fillna(0)
        base_sentiments = [sia.polarity_scores(t) for t in baseline_df['text']]
        base_sent_df = pd.DataFrame(base_sentiments)
        days_in_baseline = (baseline_df['created'].max() - baseline_df['created'].min()).days or 1
        base_hours = baseline_df['created'].dt.hour
        
        baseline_stats = {
            'len_mu': base_text_lens.mean(), 'len_std': base_text_lens.std() + 1e-8,
            'sent_compound_mu': base_sent_df['compound'].mean(), 'sent_compound_std': base_sent_df['compound'].std() + 1e-8,
            'tpd_mu': len(baseline_df) / days_in_baseline,
            'hour_median': base_hours.median(), 'hour_std': base_hours.std() + 1e-8,
        }

        # Add dummy caption token for consistency
        later_df['text'] = later_df['text'].apply(lambda x: (_clean_text(x) + " [IMG_CAP] ")[:4096])

        # --- FULL VECTORIZED FEATURE CALCULATION (Matching merge_blip_final) ---
        later_df['len_val'] = later_df['text'].str.len()
        sentiments = [sia.polarity_scores(t) for t in later_df['text']]
        sent_df = pd.DataFrame(sentiments, index=later_df.index)
        later_df = pd.concat([later_df, sent_df], axis=1)

        later_df['len_dev'] = later_df['len_val'] - baseline_stats['len_mu']
        later_df['len_z'] = later_df['len_dev'] / baseline_stats['len_std']
        later_df['sent_compound_dev'] = later_df['compound'] - baseline_stats['sent_compound_mu']
        later_df['sent_compound_z'] = later_df['sent_compound_dev'] / baseline_stats['sent_compound_std']
        
        later_df['hour'] = later_df['created'].dt.hour
        later_df['circadian_deviation'] = (later_df['hour'] - baseline_stats['hour_median']) / baseline_stats['hour_std']
        later_df['late_night'] = ((later_df['hour'] >= 23) | (later_df['hour'] <= 3)).astype(float)
        
        intervals = later_df['created'].diff().dt.total_seconds().fillna(0) / 3600
        later_df['burst_posting'] = (intervals < 1).astype(float)
        later_df['silence_period'] = (intervals > 72).astype(float)

        later_indexed = later_df.set_index('created')
        for days in [3, 7, 14, 30]:
            tweet_counts = later_indexed['text'].rolling(f'{days}D').count()
            later_df[f'tpd_{days}d'] = tweet_counts.values / float(days)
            later_df[f'tpd_{days}d_dev'] = later_df[f'tpd_{days}d'] - baseline_stats['tpd_mu']
        
        later_df['emotion_volatility_10tw'] = later_df['compound'].rolling(window=10, min_periods=2).std().fillna(0)
        later_df['polarity_shift'] = (later_df['compound'] * later_df['compound'].shift(1) < 0).astype(float).fillna(0)

        later_df['self_focus'] = later_df['text'].str.lower().str.count(r'\b(i|me|my|myself)\b') / (later_df['text'].str.split().str.len() + 1e-8)
        later_df['exclamation_count'] = later_df['text'].str.count('!')
        # Reddit doesn't really use @mentions in the same way, but we calculate it to keep shape consistent (likely 0)
        later_df['mention_count'] = later_df['text'].str.count('@') 
        later_df['isolation_score'] = later_df['text'].str.lower().str.count(r'\b(alone|lonely|nobody)\b')
        later_df['anomaly_score'] = later_df[['len_z', 'sent_compound_z', 'circadian_deviation']].abs().sum(axis=1)

        later_df['manic_indicator'] = (later_df['burst_posting'] > 0) & (later_df['compound'] > 0.5) & (later_df['late_night'] > 0)
        later_df['depressive_indicator'] = (later_df['silence_period'] > 0) & (later_df['compound'] < -0.3)
        
        return later_df

    print("\nProcessing eRisk timelines with full feature set...")
    all_user_dfs = [_process_user_timeline(group) for _, group in tqdm(df.groupby('user_id'), desc="Processing eRisk Users")]
    df_features = pd.concat(all_user_dfs, ignore_index=True)
    
    # --- Global Standardization (Matching logic) ---
    print("\nApplying global standardization...")
    pre_scaled_features = ['len_z', 'sent_compound_z', 'circadian_deviation', 'anomaly_score']
    features_to_scale = [
        c for c in df_features.columns 
        if df_features[c].dtype in ['float64', 'int64'] 
        and c not in pre_scaled_features 
        and c not in ['user_id', 'label', 'hour']
    ]
    
    df_features[features_to_scale] = df_features[features_to_scale].fillna(0)
    scaler = StandardScaler()
    df_features[features_to_scale] = scaler.fit_transform(df_features[features_to_scale])
    
    # Sanitize
    df_features.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df_features

if __name__ == '__main__':
    INPUT_PARQUET_PATH = "data/erisk_parsed_raw.parquet"
    OUTPUT_PROCESSED_PATH = "data/erisk_processed_for_testing.parquet"
    
    raw_erisk_df = pd.read_parquet(INPUT_PARQUET_PATH)
    processed_erisk_df = generate_deviation_features(raw_erisk_df)
    processed_erisk_df.to_parquet(OUTPUT_PROCESSED_PATH, index=False)
    
    print(f"\nSaved processed eRisk data ready for evaluation to: {OUTPUT_PROCESSED_PATH}")
    print(f"Total columns generated: {len(processed_erisk_df.columns)}")