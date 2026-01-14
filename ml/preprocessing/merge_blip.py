"""
• Loads FOUR nested-JSON files: depression, anxiety, bipolar, and controls
• Appends BLIP captions per tweet_id
• For each user, processes their timeline tweet-by-tweet.
• For each tweet, it computes deviation features relative to that user's baseline (first 40%).
• Creates a clean, training-ready Parquet file where EACH ROW IS ONE TWEET.
• Incorporates: temporal rhythms, emotional dynamics, linguistic complexity, etc.
• Saves a training-ready Parquet with one row per tweet.
"""
import json
import pathlib
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
FILES = {
    "data/user_timelines.json": "depression",
    "data/user_timelinesA.json": "anxiety",
    "data/user_timelinesB.json": "bipolar",
    "data/all_controls_timelines.json": "control",
}
LABEL_ID = {"control": 0, "depression": 1, "anxiety": 2, "bipolar": 3}
CAP_DIR = pathlib.Path("data/captions")
OUT_PARQUET = "data/final.parquet"
MIN_TWEETS = 50
BASE_FRAC = 0.40
VAL_PCT = 0.10
TEST_PCT = 0.20
RANDOM_SEED = 42
MAX_NONEN_FRAC = 0.30   # drop users if >30% of their tweets are non-English


class DataProcessor:
    def __init__(self):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        self.sia = SentimentIntensityAnalyzer()
        self.tweet_captions = self._load_all_captions()

    def _clean_text(self, txt: str) -> str:
        return re.sub(r"http\S+|@\w+", "", str(txt)).strip()

    def _parse_date(self, s: str) -> pd.Timestamp:
        s = s.replace(" UTC", "").replace(" · ", " ")
        return pd.to_datetime(s, errors='coerce')

    def _load_all_captions(self) -> dict:
        """Loads all caption files from the CAP_DIR into a dictionary for fast lookup."""
        print("--- Loading all image captions into memory ---")
        captions_dict = {}
        caption_files = list(CAP_DIR.glob("*.txt"))
        for fp in tqdm(caption_files, desc="Loading Captions"):
            try:
                key = fp.stem.split('_')[0]
                caption_text = fp.read_text(encoding='utf-8', errors='ignore').strip()
                if key not in captions_dict: captions_dict[key] = []
                captions_dict[key].append(caption_text)
            except Exception: continue
        for key, val_list in captions_dict.items():
            captions_dict[key] = " ".join(val_list)
        print(f"Loaded captions for {len(captions_dict)} unique tweets.")
        return captions_dict
    
    def _is_english(self, text: str) -> bool:
        """
        Robustly checks if a text is English, with a heuristic fallback.
        """
        try:
            from langdetect import detect, LangDetectException
            # Heuristic check first to avoid errors on short/weird text
            if len(text) < 15 and not any(word in text.lower() for word in ['the', 'a', 'is', 'i']):
                return True # Assume short/emoji-only text is fine
            return detect(text) == 'en'
        except (ImportError, LangDetectException):
            # Fallback to a simpler heuristic if langdetect fails or is not installed
            common_en_chars = "etaoinshrdlu"
            if not text or text.isnumeric(): return True
            alpha_chars = sum(c.isalpha() for c in text.lower())
            if alpha_chars < 5: return True # Too short to judge reliably
            en_char_count = sum(text.lower().count(c) for c in common_en_chars)
            return (en_char_count / alpha_chars) > 0.45 # Check if letter frequency is English-like

    def _heuristic_lang(self, text: str) -> str:
        """
        Heuristic fallback: if text looks sufficiently 'English', return 'en', else 'xx'.
        Criteria:
          - Ratio of [A-Za-z whitespace basic punct] to all chars
          - Presence of common English stopwords
        """
        import string, re
        cleaned = self._clean_text(text).lower()
        if not cleaned:
            return 'en'  # empty/emoji-only tweets won't penalize users

        # Remove urls/emails/handles (already mostly handled in _clean_text)
        cleaned = re.sub(r"http\S+|\S+@\S+|#\w+", "", cleaned)

        letters = sum(ch.isalpha() for ch in cleaned)
        latin_letters = sum(('a' <= ch <= 'z') or ('A' <= ch <= 'Z') for ch in cleaned)
        total = len(cleaned)
        if total == 0:
            return 'en'

        latin_ratio = latin_letters / max(1, letters or total)

        # light stopword signal
        tokens = re.findall(r"[a-z']+", cleaned)
        common_en = {"the","and","is","to","you","i","a","of","in","it","that","for","on","with","me","my"}
        stop_hit = len(common_en.intersection(tokens)) > 0

        # thresholds tuned to be forgiving but effective
        if latin_ratio >= 0.75 and stop_hit:
            return 'en'
        if latin_ratio >= 0.85:  # strong Latin signal even w/o stopwords
            return 'en'
        return 'xx'

    def _is_english(self, text: str) -> bool:
        lang = self._try_langdetect(text)
        if lang is None:
            lang = self._heuristic_lang(text)
        return lang == 'en'

    def process_user_timeline(self, group):
        """
        --- CORRECTED & FULLY-FEATURED VECTORIZED VERSION ---
        """
        n_base = max(10, int(len(group) * BASE_FRAC))
        baseline_df = group.head(n_base)
        later_df = group.iloc[n_base:].copy()

        if later_df.empty: return pd.DataFrame()

        # --- 1. Compute Comprehensive Baseline Statistics ---
        base_text_lens = baseline_df['text'].str.len().fillna(0)
        base_sentiments = [self.sia.polarity_scores(t) for t in baseline_df['text']]
        base_sent_df = pd.DataFrame(base_sentiments)
        days_in_baseline = (baseline_df['created'].max() - baseline_df['created'].min()).days or 1
        base_hours = baseline_df['created'].dt.hour
        baseline_stats = {
            'len_mu': base_text_lens.mean(), 'len_std': base_text_lens.std() + 1e-8,
            'sent_compound_mu': base_sent_df['compound'].mean(), 'sent_compound_std': base_sent_df['compound'].std() + 1e-8,
            'tpd_mu': len(baseline_df) / days_in_baseline,
            'hour_median': base_hours.median(), 'hour_std': base_hours.std() + 1e-8,
        }

        # --- 2. Pre-calculate features for the entire `later_df` at once ---
        def get_combined_text(row):
            caption = self.tweet_captions.get(row['tweet_id'], "")
            return (self._clean_text(row['text']) + " [IMG_CAP] " + caption)[:4096]
        later_df['text'] = later_df.apply(get_combined_text, axis=1)

        later_df['len_val'] = later_df['text'].str.len()
        sentiments = [self.sia.polarity_scores(t) for t in later_df['text']]
        sent_df = pd.DataFrame(sentiments, index=later_df.index)
        later_df = pd.concat([later_df, sent_df], axis=1)

        # --- 3. VECTORIZED FEATURE CALCULATION (ALL FEATURES RESTORED) ---
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
        later_df['mention_count'] = later_df['text'].str.count('@')
        later_df['isolation_score'] = later_df['text'].str.lower().str.count(r'\b(alone|lonely|nobody)\b')
        later_df['anomaly_score'] = later_df[['len_z', 'sent_compound_z', 'circadian_deviation']].abs().sum(axis=1)

        later_df['manic_indicator'] = (later_df['burst_posting'] > 0) & (later_df['compound'] > 0.5) & (later_df['late_night'] > 0)
        later_df['depressive_indicator'] = (later_df['silence_period'] > 0) & (later_df['compound'] < -0.3)
        
        return later_df

# In merge_blip_final.py

# ... (the rest of the class is the same) ...

    def run(self):
        """Executes the full data processing pipeline with DEFINITIVELY corrected standardization."""
        
        # Steps 1, 2, & 3 are the same (Load, Filter, Process Users)
        print("\n--- Loading and Merging Raw Tweet Data ---")
        rows = [];
        for path, label_name in FILES.items():
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
            for username, tweets in data.items():
                for tw in tweets:
                    try:
                        created = self._parse_date(tw["date"])
                        if pd.isna(created): continue
                        rows.append({"tweet_id": str(tw["id"]), "user_id": str(tw["user"]["profile_id"]), "username": username, "created": created, "text": tw.get("text", ""), "label_name": label_name, "label": LABEL_ID[label_name]})
                    except (KeyError, TypeError): continue
        df = pd.DataFrame(rows).sort_values(["user_id", "created"]).reset_index(drop=True)
        df['label'] = df.groupby('user_id')['label'].transform('first')
        df['label_name'] = df.groupby('user_id')['label_name'].transform('first')
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= MIN_TWEETS].index)]
        print(f"Filtered to {df['user_id'].nunique()} users with >= {MIN_TWEETS} tweets.")

        # Drop users with too many non-English tweets ---
        print("\n--- Screening users by language mix ---")
        # compute per-tweet englishness
        df['is_english'] = df['text'].apply(self._is_english)
        # fraction of non-English tweets per user
        nonen_frac = df.groupby('user_id')['is_english'].apply(lambda s: 1.0 - s.mean())
        keep_users = nonen_frac[nonen_frac <= MAX_NONEN_FRAC].index
        dropped_users = set(df['user_id'].unique()) - set(keep_users)
        df = df[df['user_id'].isin(keep_users)].copy()
        print(f"Dropped {len(dropped_users)} users exceeding MAX_NONEN_FRAC={MAX_NONEN_FRAC:.2f}.")
        print(f"Remaining users after language screen: {df['user_id'].nunique()}")


        print("\n--- Processing Timelines with Feature Engineering ---")
        all_user_dfs = [self.process_user_timeline(group) for _, group in tqdm(df.groupby('user_id'), desc="Processing Users")]
        df_features = pd.concat(all_user_dfs, ignore_index=True)
        

        # Standardize ONLY the features that are truly "raw"
        print("\n--- Applying Global Standardization (Definitively Corrected) ---")
        
        # Z-scored and deviation-like features should NOT be re-scaled globally.
        # We create an explicit list of columns to preserve.
        pre_scaled_features = ['len_z', 'sent_compound_z', 'circadian_deviation', 'anomaly_score']
        
        # Now we identify only the features that should be scaled.
        features_to_scale = [
            c for c in df_features.columns if 
            df_features[c].dtype in ['float64', 'int64'] and 
            c not in pre_scaled_features and
            c not in ['user_id', 'label', 'hour', 'dow']
        ]
        print(f"Standardizing {len(features_to_scale)} raw feature columns...")

        df_features[features_to_scale] = df_features[features_to_scale].fillna(0)
        scaler = StandardScaler()
        df_features[features_to_scale] = scaler.fit_transform(df_features[features_to_scale])

        # -----------------------------------------------

        # The rest of the steps are the same
        # Splits
        print("\n--- Creating Train/Val/Test Splits ---")
        train_u, val_u, test_u = [], [], []
        user_df = df_features.drop_duplicates('user_id')
        for lbl, users in user_df.groupby('label')['user_id'].unique().items():
            users = list(users); random.shuffle(users); n = len(users)
            n_train = max(1, int((1 - VAL_PCT - TEST_PCT) * n)); n_val = max(0 if n <= 2 else 1, int(VAL_PCT * n))
            train_u += users[:n_train]; val_u += users[n_train:n_train + n_val]; test_u += users[n_train + n_val:]
        split_map = {u: "train" for u in train_u} | {u: "val" for u in val_u} | {u: "test" for u in test_u}
        df_features['split'] = df_features['user_id'].map(split_map)
        df_features = df_features.dropna(subset=['split'])
        
        # Class Weights & Save
        print("\n--- Computing Class Weights ---")
        train_df = df_features[df_features['split'] == 'train']
        weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
        np.save("data/class_weights.npy", weights)
        print(f"Class weights computed and saved: {weights}")

        print(f"\n--- Saving Final Dataset to {OUT_PARQUET} ---")
        df_features.to_parquet(OUT_PARQUET, index=False)
        print("Preprocessing complete!")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.run()