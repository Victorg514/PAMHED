"""
• Loads FOUR nested-JSON files: depression, anxiety, bipolar, and controls
• Appends BLIP captions per tweet_id (model input text only)
• For each user, processes their timeline tweet-by-tweet.
• For each tweet, it computes deviation features relative to that user's baseline (first 40%).
• Creates a clean, training-ready Parquet file where EACH ROW IS ONE TWEET.
• Incorporates: temporal rhythms, emotional dynamics, linguistic complexity, etc.
• Features are saved UNSCALED: scaling is fit on the train split at training time (ml/features.py).

See FEATURE_FIXES.md for why the feature code changed (v2).
"""
import json
import pathlib
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
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
# Reuse the user -> split assignment of the previous dataset so old and new results share test users
PREVIOUS_SPLITS = pathlib.Path("data/backup_v1/final.parquet")
# Keep only users that were in the previous dataset, so v1 and v2 results differ only by the feature fixes
# (the earlier language screen was stricter; the extra users are less English)
# False since the retweet fix: the v1 user set contained the fake retweet "users", so matching it no
# longer makes sense. Real v1 users still keep their v1 split; new users get a stratified split.
RESTRICT_TO_PREVIOUS_USERS = False
MIN_HOUR_SPREAD = 1.0   # floor for the circular hour spread, in hours
# Floors for the baseline standard deviations: a user whose baseline tweets all have the same length or
# sentiment would otherwise get z-scores in the billions (this broke len_z in v1)
LEN_STD_FLOOR = 10.0     # characters
SENT_STD_FLOOR = 0.05    # VADER compound units
CAPTION_CACHE = pathlib.Path("data/captions_cache.parquet")   # one file instead of 630k small ones

SELF_WORDS = r'\b(i|me|my|myself)\b'
ISOLATION_WORDS = r'\b(alone|lonely|nobody)\b'


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
        if CAPTION_CACHE.exists():
            cache = pd.read_parquet(CAPTION_CACHE)
            print(f"Loaded captions for {len(cache)} tweets from {CAPTION_CACHE} (delete it to re-read {CAP_DIR})")
            return dict(zip(cache['tweet_id'], cache['caption']))
        print("--- Loading all image captions into memory ---")
        captions_dict = {}
        caption_files = sorted(CAP_DIR.glob("*.txt"))
        for fp in tqdm(caption_files, desc="Loading Captions"):
            try:
                key = fp.stem.split('_')[0]
                caption_text = fp.read_text(encoding='utf-8', errors='ignore').strip()
                if key not in captions_dict: captions_dict[key] = []
                captions_dict[key].append(caption_text)
            except Exception: continue
        # Sorted file names keep multi-image captions in a stable order
        for key, val_list in captions_dict.items():
            captions_dict[key] = " ".join(val_list)
        print(f"Loaded captions for {len(captions_dict)} unique tweets.")
        pd.DataFrame({'tweet_id': list(captions_dict), 'caption': list(captions_dict.values())}).to_parquet(CAPTION_CACHE, index=False)
        return captions_dict
    
    def _try_langdetect(self, text: str):
        """Returns langdetect's language code, or None if it is unavailable or cannot decide."""
        try:
            from langdetect import detect, DetectorFactory, LangDetectException
        except ImportError:
            return None
        DetectorFactory.seed = 0  # langdetect is random otherwise
        cleaned = self._clean_text(text)
        # Short/emoji-only text is too short to judge; let the heuristic decide
        if len(cleaned) < 15:
            return None
        try:
            return detect(cleaned)
        except LangDetectException:
            return None

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

    @staticmethod
    def _circular_hour_stats(hours: pd.Series):
        """Circular mean and spread of posting hours, both in hours (23:00 and 01:00 are 2h apart)."""
        angles = hours.to_numpy(dtype=float) * 2 * np.pi / 24
        s, c = np.sin(angles).mean(), np.cos(angles).mean()
        center = np.arctan2(s, c) * 24 / (2 * np.pi)  # any value works: differences are wrapped later
        r = min(max(np.hypot(s, c), 1e-8), 1.0)
        spread = np.sqrt(-2 * np.log(r)) * 24 / (2 * np.pi)
        return center, max(spread, MIN_HOUR_SPREAD)

    def process_user_timeline(self, group):
        """
        Computes every feature on the FULL timeline, then keeps only the post-baseline tweets.
        - All text features (baseline and later) use the same cleaned tweet text, without the caption.
        - Rolling/diff features see the baseline tweets too, so the first later tweets have no warm-up artifacts.
        """
        n_base = max(10, int(len(group) * BASE_FRAC))
        if len(group) <= n_base: return pd.DataFrame()
        df = group.copy()

        # --- 1. Per-tweet text features, from the cleaned text WITHOUT the caption ---
        feat_text = df['text'].fillna("").map(self._clean_text)
        n_words = feat_text.str.split().str.len().fillna(0)
        df['len_val'] = feat_text.str.len()
        sent_df = pd.DataFrame([self.sia.polarity_scores(t) for t in feat_text], index=df.index)
        df = pd.concat([df, sent_df], axis=1)

        df['self_focus'] = feat_text.str.lower().str.count(SELF_WORDS) / (n_words + 1)
        df['exclamation_rate'] = feat_text.str.count('!') / (n_words + 1)
        df['isolation_rate'] = feat_text.str.lower().str.count(ISOLATION_WORDS) / (n_words + 1)
        # Mentions are stripped by _clean_text, so count them on the raw text
        df['mention_count'] = df['text'].fillna("").str.count(r'@\w+')

        # --- 2. Timing and activity (full timeline) ---
        df['hour'] = df['created'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 3)).astype(float)

        intervals = df['created'].diff().dt.total_seconds() / 3600
        df['burst_posting'] = (intervals < 1).astype(float)        # first tweet: NaN -> 0
        df['silence_period'] = (intervals > 72).astype(float)

        indexed = df.set_index('created')
        for days in [3, 7, 14, 30]:
            df[f'tpd_{days}d'] = indexed['len_val'].rolling(f'{days}D').count().values / float(days)

        df['emotion_volatility_10tw'] = df['compound'].rolling(window=10, min_periods=2).std().fillna(0)
        df['polarity_shift'] = (df['compound'] * df['compound'].shift(1) < 0).astype(float)

        # --- 3. Baseline statistics (first n_base tweets) ---
        base = df.iloc[:n_base]
        days_in_baseline = (base['created'].max() - base['created'].min()).days or 1
        hour_center, hour_spread = self._circular_hour_stats(base['hour'])
        stats = {
            'len_mu': base['len_val'].mean(), 'len_std': max(base['len_val'].std(), LEN_STD_FLOOR),
            'sent_compound_mu': base['compound'].mean(), 'sent_compound_std': max(base['compound'].std(), SENT_STD_FLOOR),
            'tpd_mu': len(base) / days_in_baseline,
        }

        # --- 4. Keep the later tweets and compute deviations from the baseline ---
        later_df = df.iloc[n_base:].copy()
        later_df['len_dev'] = later_df['len_val'] - stats['len_mu']
        later_df['len_z'] = later_df['len_dev'] / stats['len_std']
        later_df['sent_compound_dev'] = later_df['compound'] - stats['sent_compound_mu']
        later_df['sent_compound_z'] = later_df['sent_compound_dev'] / stats['sent_compound_std']
        # Signed circular difference in [-12, 12) hours, scaled by the user's circular spread
        hour_diff = (later_df['hour'] - hour_center + 12) % 24 - 12
        later_df['circadian_deviation'] = hour_diff / hour_spread
        for days in [3, 7, 14, 30]:
            later_df[f'tpd_{days}d_dev'] = later_df[f'tpd_{days}d'] - stats['tpd_mu']
        later_df['anomaly_score'] = later_df[['len_z', 'sent_compound_z', 'circadian_deviation']].abs().sum(axis=1)

        later_df['manic_indicator'] = ((later_df['burst_posting'] > 0) & (later_df['compound'] > 0.5) & (later_df['late_night'] > 0)).astype(float)
        later_df['depressive_indicator'] = ((later_df['silence_period'] > 0) & (later_df['compound'] < -0.3)).astype(float)

        # --- 5. Model input text: cleaned tweet + image caption ---
        captions = later_df['tweet_id'].map(self.tweet_captions).fillna("")
        later_df['text'] = (later_df['text'].fillna("").map(self._clean_text) + " [IMG_CAP] " + captions).str.slice(0, 4096)

        return later_df

    def _assign_splits(self, df_features):
        """Reuses the previous dataset's user splits when available; new users get a stratified random split."""
        split_map = {}
        if PREVIOUS_SPLITS.exists():
            prev = pd.read_parquet(PREVIOUS_SPLITS, columns=['user_id', 'split']).drop_duplicates('user_id')
            split_map = dict(zip(prev['user_id'], prev['split']))
            print(f"Reusing splits for {len(split_map)} users from {PREVIOUS_SPLITS}")

        rng = random.Random(RANDOM_SEED)
        user_df = df_features.drop_duplicates('user_id')
        new_users = user_df[~user_df['user_id'].isin(split_map)]
        for lbl, users in new_users.groupby('label')['user_id'].unique().items():
            users = sorted(users); rng.shuffle(users); n = len(users)
            n_train = max(1, int((1 - VAL_PCT - TEST_PCT) * n)); n_val = max(0 if n <= 2 else 1, int(VAL_PCT * n))
            split_map |= {u: "train" for u in users[:n_train]}
            split_map |= {u: "val" for u in users[n_train:n_train + n_val]}
            split_map |= {u: "test" for u in users[n_train + n_val:]}
        print(f"Assigned new splits to {len(new_users)} users.")
        return df_features['user_id'].map(split_map)

    def run(self):
        """Executes the full data processing pipeline with DEFINITIVELY corrected standardization."""
        
        # Steps 1, 2, & 3 are the same (Load, Filter, Process Users)
        print("\n--- Loading and Merging Raw Tweet Data ---")
        # Column lists instead of one dict per tweet, and each JSON file freed before the next: keeps peak RAM low
        # Every tweet belongs to the TIMELINE OWNER, not to tw["user"] (the author). For a retweet the
        # author is someone else, and keying on the author created fake "users" made of many people's
        # retweets (v1 and the first v2 build). Retweets are someone else's words, so they are dropped.
        cols = {k: [] for k in ["tweet_id", "user_id", "username", "created", "text", "label_name"]}
        n_retweets = n_no_owner = 0
        for path, label_name in FILES.items():
            print(f"Reading {path}...")
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
            for username, tweets in data.items():
                owner = username.lstrip("@").lower()
                # The owner's profile_id comes from a tweet they wrote themselves
                owner_id = next((str(u["profile_id"]) for tw in tweets if isinstance(tw, dict)
                                 for u in [tw.get("user") or {}] if u.get("profile_id")
                                 and str(u.get("username", "")).lstrip("@").lower() == owner), None)
                if owner_id is None:
                    n_no_owner += 1
                    continue
                for tw in tweets:
                    try:
                        if str(tw.get("is-retweet", "False")).lower() == "true":
                            n_retweets += 1
                            continue
                        row = (str(tw["id"]), owner_id, username, tw["date"], tw.get("text", ""), label_name)
                    except (KeyError, TypeError, AttributeError): continue
                    for k, v in zip(cols, row): cols[k].append(v)
            del data
        print(f"Dropped {n_retweets} retweets; skipped {n_no_owner} timelines without any own tweet.")
        df = pd.DataFrame(cols); del cols
        df['created'] = pd.to_datetime(df['created'].str.replace(" UTC", "", regex=False).str.replace(" · ", " ", regex=False), format='mixed', errors='coerce')
        df = df.dropna(subset=['created']).drop_duplicates(['user_id', 'tweet_id'])
        # A person scraped into two condition files has no single label: drop them
        n_labels = df.groupby('user_id')['label_name'].nunique()
        ambiguous = set(n_labels[n_labels > 1].index)
        df = df[~df['user_id'].isin(ambiguous)]
        print(f"Dropped {len(ambiguous)} users that appear in more than one condition file.")
        df['label'] = df['label_name'].map(LABEL_ID)
        df = df.sort_values(["user_id", "created"]).reset_index(drop=True)
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
        del all_user_dfs

        # No global standardization here: features are scaled with statistics from the train split
        # at training time (ml/features.py), so test users never influence the scaling.
        num_cols = df_features.select_dtypes(include='number').columns.drop(['label'])
        df_features[num_cols] = df_features[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        print("\n--- Creating Train/Val/Test Splits ---")
        df_features['split'] = self._assign_splits(df_features)
        df_features = df_features.dropna(subset=['split'])
        if RESTRICT_TO_PREVIOUS_USERS and PREVIOUS_SPLITS.exists():
            prev_users = set(pd.read_parquet(PREVIOUS_SPLITS, columns=['user_id'])['user_id'])
            df_features = df_features[df_features['user_id'].isin(prev_users)]
            print(f"Kept {df_features['user_id'].nunique()} users that are in the previous dataset.")
        print(df_features.groupby('split')['user_id'].nunique().to_string())
        
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