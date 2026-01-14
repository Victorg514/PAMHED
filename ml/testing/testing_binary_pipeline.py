# testing_pipeline_final.py

import os
import json
import re
import requests
import pathlib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
from PIL import Image

# Core ML/NLP Libraries
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, RobertaModel, AutoModelForSequenceClassification, AutoConfig
)
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_selection import mutual_info_classif
from datasets import Dataset
from torch.utils.data import DataLoader
import shap

# Scraper (assuming ntscraper is in the environment)
from ntscraper import Nitter
import ntscraper.nitter as ntr


# --- MODEL DEFINITION ---
# This MUST be the stable, gated model class used in your final binary delta training script
class AdvancedDeltaModel(nn.Module):
    def __init__(self, base_name="mental/mental-roberta-base", num_labels=2, num_features=50, dropout_rate=0.3):
        super().__init__(); self.num_labels = num_labels; self.base = RobertaModel.from_pretrained(base_name); hidden_size = self.base.config.hidden_size; self.feature_encoder = nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(dropout_rate)); self.fusion_gate = nn.Sequential(nn.Linear(hidden_size + 64, 64), nn.ReLU(), nn.Linear(64, 64), nn.Sigmoid()); self.classifier = nn.Linear(hidden_size + 64, num_labels)
    def forward(self, input_ids=None, attention_mask=None, delta=None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask); text_features = outputs.last_hidden_state[:, 0]; behavioral_features_encoded = self.feature_encoder(delta.float()); gate_input = torch.cat([text_features, behavioral_features_encoded], dim=1); gate_values = self.fusion_gate(gate_input); gated_behavioral_features = behavioral_features_encoded * gate_values; final_combined_features = torch.cat([text_features, gated_behavioral_features], dim=1); logits = self.classifier(final_combined_features)
        return logits


class TwitterMentalHealthPipeline:
    """
    End-to-end pipeline for scraping, processing, predicting, and explaining
    Twitter user data for mental health analysis.
    """
    def __init__(self,
                 base_dir: str = "pipeline_data",
                 nitter_instance: str = "http://localhost:8080",
                 min_tweets: int = 50,
                 baseline_frac: float = 0.40):
        
        self.base_dir = pathlib.Path(base_dir)
        self.nitter_instance = nitter_instance
        self.min_tweets = min_tweets
        self.baseline_frac = baseline_frac
        
        self.data_dir = self.base_dir / "data"
        self.images_dir = self.base_dir / "images"
        self.captions_dir = self.base_dir / "captions"
        self.parquet_dir = self.base_dir / "parquets"
        for dir_path in [self.data_dir, self.images_dir, self.captions_dir, self.parquet_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self._blip_processor = None
        self._blip_model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sia = SentimentIntensityAnalyzer()
        self._apply_nitter_patch()
        self.tweet_captions = {} # Will be populated during processing

    # (Helper properties and patch are the same)
    def _apply_nitter_patch(self):
        # ... (same as your previous script)
        _orig_get_user = ntr.Nitter._get_user
        def _safe_get_user(self, tweet, is_encrypted):
            try: return _orig_get_user(self, tweet, is_encrypted)
            except IndexError:
                uname = tweet.find("a", class_="username"); fname = tweet.find("a", class_="fullname")
                return {"id": None, "username": uname.text.lstrip("@") if uname else "unknown", "fullname": fname.text if fname else "unknown", "avatar_url": None}
        ntr.Nitter._get_user = _safe_get_user
        
    @property
    def blip_processor(self):
        # ... (same as your previous script)
        if self._blip_processor is None: self._blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        return self._blip_processor

    @property
    def blip_model(self):
        # ... (same as your previous script)
        if self._blip_model is None: self._blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self._device)
        return self._blip_model

    # (scrape_user and caption_images are mostly the same)
    def scrape_user(self, username: str, max_tweets: int = -1) -> Dict:
        # ... (same as your previous script)
        username = username.lstrip("@"); user_dir = self.images_dir / username; user_dir.mkdir(parents=True, exist_ok=True)
        print(f"▸ Scraping timeline for @{username}")
        scraper = Nitter(log_level=1, skip_instance_check=True)
        try:
            result = scraper.get_tweets(username, mode="user", number=max_tweets, instance=self.nitter_instance)
            timeline = result.get("tweets", [])
        except Exception as e: print(f"  • Error fetching tweets: {e}"); raise
        print(f"  • {len(timeline)} tweets fetched")
        json_path = self.data_dir / f"{username}_raw.json"
        with open(json_path, 'w', encoding='utf-8') as f: json.dump(timeline, f, ensure_ascii=False, indent=2)
        print(f"  • Timeline written to {json_path}")
        img_urls, tweet_images = [], {}
        for tw in timeline:
            tweet_id = str(tw.get("id")); tweet_images[tweet_id] = []
            for url in tw.get("photos", []):
                if url not in img_urls: img_urls.append(url); tweet_images[tweet_id].append(len(img_urls) - 1)
        print(f"  • {len(img_urls)} unique images to download")
        sess = requests.Session(); downloaded_images = {}
        for idx, url in enumerate(img_urls):
            ext = pathlib.Path(url).suffix.split("?")[0] or ".jpg"; fname = user_dir / f"{idx:04d}{ext}"
            try:
                with sess.get(url, timeout=30, stream=True) as r:
                    r.raise_for_status()
                    with open(fname, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536): f.write(chunk)
                downloaded_images[idx] = str(fname)
            except Exception as e: print(f"    ✗ Failed {url} → {e}"); downloaded_images[idx] = None
        return {"username": username, "tweets": timeline, "tweet_images": tweet_images, "downloaded_images": downloaded_images}

    def caption_images(self, scrape_data: Dict):
        # ... (same as your previous script, now populates self.tweet_captions)
        username = scrape_data["username"]; tweet_images = scrape_data["tweet_images"]; downloaded_images = scrape_data["downloaded_images"]
        print(f"\n▸ Generating image captions for @{username}")
        image_captions = {}
        for img_idx, img_path in tqdm(downloaded_images.items(), desc="Captioning"):
            if img_path is None: continue
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = self.blip_processor(images=img, return_tensors="pt").to(self._device)
                out = self.blip_model.generate(**inputs, max_new_tokens=20)
                image_captions[img_idx] = self.blip_processor.decode(out[0], skip_special_tokens=True)
            except Exception as e: print(f"Failed to caption {img_path}: {e}"); image_captions[img_idx] = ""
        for tweet_id, img_indices in tweet_images.items():
            captions = [image_captions[idx] for idx in img_indices if idx in image_captions and image_captions[idx]]
            self.tweet_captions[tweet_id] = " ".join(captions)
    
    # --- UPDATED: Full feature engineering logic from merge_blip_final.py ---
    def _clean_text(self, txt: str) -> str: return re.sub(r"http\S+|@\w+", "", str(txt)).strip()
    def _parse_date(self, s: str) -> pd.Timestamp: return pd.to_datetime(s.replace(" UTC", "").replace(" · ", " "), errors='coerce')

    def _process_user_timeline(self, group):
        # (This is the complete, vectorized feature engineering function)
        n_base = max(10, int(len(group) * self.baseline_frac))
        baseline_df = group.head(n_base)
        later_df = group.iloc[n_base:].copy()
        if later_df.empty: return pd.DataFrame()
        base_text_lens = baseline_df['text'].str.len().fillna(0)
        base_sentiments = [self.sia.polarity_scores(t) for t in baseline_df['text']]
        base_sent_df = pd.DataFrame(base_sentiments)
        days_in_baseline = (baseline_df['created'].max() - baseline_df['created'].min()).days or 1
        base_hours = baseline_df['created'].dt.hour
        baseline_stats = {'len_mu': base_text_lens.mean(), 'len_std': base_text_lens.std() + 1e-8, 'sent_compound_mu': base_sent_df['compound'].mean(), 'sent_compound_std': base_sent_df['compound'].std() + 1e-8, 'tpd_mu': len(baseline_df) / days_in_baseline, 'hour_median': base_hours.median(), 'hour_std': base_hours.std() + 1e-8}
        def get_combined_text(row):
            caption = self.tweet_captions.get(str(row['tweet_id']), "")
            return (self._clean_text(row['text']) + " [IMG_CAP] " + caption)[:4096]
        later_df['text'] = later_df.apply(get_combined_text, axis=1)
        later_df['len_val'] = later_df['text'].str.len()
        sentiments = [self.sia.polarity_scores(t) for t in later_df['text']]; sent_df = pd.DataFrame(sentiments, index=later_df.index); later_df = pd.concat([later_df, sent_df], axis=1)
        later_df['len_z'] = (later_df['len_val'] - baseline_stats['len_mu']) / baseline_stats['len_std']
        later_df['sent_compound_z'] = (later_df['compound'] - baseline_stats['sent_compound_mu']) / baseline_stats['sent_compound_std']
        later_df['hour'] = later_df['created'].dt.hour
        later_df['circadian_deviation'] = (later_df['hour'] - baseline_stats['hour_median']) / baseline_stats['hour_std']
        intervals = later_df['created'].diff().dt.total_seconds().fillna(0) / 3600
        later_df['burst_posting'] = (intervals < 1).astype(float)
        later_df['silence_period'] = (intervals > 72).astype(float)
        later_indexed = later_df.set_index('created')
        for days in [3, 7, 14, 30]:
            tweet_counts = later_indexed['text'].rolling(f'{days}D').count(); later_df[f'tpd_{days}d'] = tweet_counts.values / float(days); later_df[f'tpd_{days}d_dev'] = later_df[f'tpd_{days}d'] - baseline_stats['tpd_mu']
        later_df['emotion_volatility_10tw'] = later_df['compound'].rolling(window=10, min_periods=2).std().fillna(0)
        later_df['polarity_shift'] = (later_df['compound'] * later_df['compound'].shift(1) < 0).astype(float).fillna(0)
        later_df['self_focus'] = later_df['text'].str.lower().str.count(r'\b(i|me|my|myself)\b') / (later_df['text'].str.split().str.len() + 1e-8)
        later_df['anomaly_score'] = later_df[['len_z', 'sent_compound_z', 'circadian_deviation']].abs().sum(axis=1)
        later_df['manic_indicator'] = (later_df['burst_posting'] > 0) & (later_df['compound'] > 0.5) & (later_df['late_night'] > 0)
        later_df['depressive_indicator'] = (later_df['silence_period'] > 0) & (later_df['compound'] < -0.3)
        return later_df

    def process_user(self, username: str, label: int, max_tweets: int = -1) -> str:
        username = username.lstrip("@"); print(f"\n{'='*60}\nProcessing user: @{username}\n{'='*60}")
        scrape_data = self.scrape_user(username, max_tweets)
        self.caption_images(scrape_data)
        
        # Create initial DataFrame from raw tweets
        rows = []
        for tw in scrape_data["tweets"]:
            try:
                created = self._parse_date(tw["date"])
                if pd.isna(created): continue
                rows.append({"tweet_id": str(tw["id"]), "user_id": str(tw["user"]["profile_id"]), "username": username, "created": created, "text": tw.get("text", ""), "label": label})
            except (KeyError, TypeError): continue
        df = pd.DataFrame(rows).sort_values("created").reset_index(drop=True)
        
        # Process features
        df_features = self._process_user_timeline(df)
        if df_features.empty:
            print(f"✗ Pipeline halted for @{username} due to insufficient data.")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_path = self.parquet_dir / f"{username}_{timestamp}.parquet"
        df_features.to_parquet(parquet_path, index=False)
        print(f"\n✓ Feature pipeline complete! Saved to: {parquet_path}")
        return str(parquet_path)

    def predict_and_explain(self, parquet_path: str, binary_task: str, full_training_data_path: str):
        """
        --- NEW: Loads BINARY models, predicts, and generates SHAP explanations. ---
        """
        if not parquet_path:
            print("Parquet path is empty, skipping analysis.")
            return
            
        print(f"\n--- Analysis for Binary Task: {binary_task.upper()} vs. OTHERS ---")
        
        # --- Load Models ---
        baseline_dir = f"out/baseline_binary_{binary_task}_manual"
        delta_dir = f"out/delta_binary_{binary_task}_manual"
        
        baseline_model = AutoModelForSequenceClassification.from_pretrained(baseline_dir).to(self._device).eval()
        
        delta_config = AutoConfig.from_pretrained(delta_dir)
        num_features = delta_config.custom_num_features
        delta_model = AdvancedDeltaModel(base_name=delta_config._name_or_path, num_labels=2, num_features=num_features).to(self._device).eval()
        weights_path = os.path.join(delta_dir, "pytorch_model.bin")
        delta_model.load_state_dict(torch.load(weights_path, map_location=self._device))

        # --- Prepare Data ---
        user_df = pd.read_parquet(parquet_path)
        tokenizer = AutoTokenizer.from_pretrained(delta_config._name_or_path)
        
        # Get the same feature list the model was trained with
        training_df = pd.read_parquet(full_training_data_path)
        LABEL_MAP = {"control": 0, "depression": 1, "anxiety": 2, "bipolar": 3}
        if binary_task == 'mental_health':
            training_df['binary_label'] = training_df['label'].apply(lambda x: 0 if x == LABEL_MAP['control'] else 1)
        else:
            positive_label_id = LABEL_MAP[binary_task]
            training_df['binary_label'] = training_df['label'].apply(lambda x: 1 if x == positive_label_id else 0)
        
        train_split = training_df[training_df['split'] == 'train']
        from sklearn.feature_selection import mutual_info_classif
        def get_feature_list(df, n_features, binary_y):
             exclude_cols = ['tweet_id', 'user_id', 'text', 'label', 'label_name', 'split', 'created', 'binary_label']; feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']];
             if not feature_cols: return []; X = df[feature_cols].fillna(0).values; y = binary_y; mi_scores = mutual_info_classif(X, y, random_state=42);
             feature_scores = pd.DataFrame({'feature': feature_cols, 'score': mi_scores}).sort_values('score', ascending=False);
             return feature_scores.head(n_features)['feature'].tolist()
        selected_features = get_feature_list(train_split, n_features=num_features, binary_y=train_split['binary_label'].values)
        
        # --- Run Inference ---
        # (This part is simplified for clarity, can be expanded to full baseline/delta comparison)
        def tok_plus(batch):
            encoding = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
            encoding["delta"] = [[float(batch[c][i]) if c in batch and pd.notna(batch[c][i]) else 0.0 for c in selected_features] for i in range(len(batch["text"]))]
            return encoding
        
        user_ds = Dataset.from_pandas(user_df).map(tok_plus, batched=True)
        user_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'delta'])
        user_loader = DataLoader(user_ds, batch_size=8)
        
        all_logits = []
        with torch.no_grad():
            for batch in tqdm(user_loader, desc=f"Predicting for @{user_df['username'].iloc[0]}"):
                batch = {k: v.to(self._device) for k, v in batch.items()}
                all_logits.append(delta_model(**batch).cpu().numpy())
        
        proba_delta = torch.softmax(torch.tensor(np.concatenate(all_logits)), dim=1)
        pred_delta = proba_delta.argmax(axis=1)

        print("\n--- Prediction Summary ---")
        pred_counts = pd.Series(pred_delta).value_counts()
        print(f"Predicted 'Other': {pred_counts.get(0, 0)} tweets")
        print(f"Predicted '{binary_task.upper()}': {pred_counts.get(1, 0)} tweets")

        # --- SHAP Explanation for a single, high-confidence tweet ---
        if 1 in pred_counts and pred_counts[1] > 0:
            # Find a tweet predicted as positive
            positive_pred_indices = np.where(pred_delta == 1)[0]
            # Pick the one with the highest confidence
            idx_to_explain = positive_pred_indices[proba_delta[positive_pred_indices, 1].argmax()]

            print(f"\n--- Generating SHAP explanation for a high-confidence '{binary_task}' tweet ---")
            
            # (SHAP logic adapted from your notebook)
            background_df = training_df.sample(50, random_state=42)
            background_ds = Dataset.from_pandas(background_df).map(tok_plus, batched=True)
            background_ds.set_format(type='torch', columns=['input_ids', 'delta'])

            instance_to_explain = user_ds[int(idx_to_explain)]
            
            def predict_for_shap(input_ids, delta):
                logits = delta_model(input_ids=input_ids, attention_mask=(input_ids != tokenizer.pad_token_id), delta=delta)
                return torch.softmax(logits, dim=-1)

            # NOTE: SHAP on transformers is complex. This is a simplified conceptual placeholder.
            # A full implementation requires careful handling of input formats.
            print("SHAP visualization would be generated here.")
            print("Example text:", user_df.iloc[int(idx_to_explain)]['text'])


if __name__ == '__main__':
    # --- DEMO CONFIGURATION ---
    TARGET_USERNAME = "anxietytxtmsgs"
    # This is the "true" label for the user, for context. It's not used by the model.
    TRUE_LABEL = 2 # 0=control, 1=depression, 2=anxiety, 3=bipolar
    # This is the binary task we want to test
    BINARY_TASK = "anxiety" 
    
    # Path to the full dataset, needed to get the training feature list
    FULL_DATASET_PATH = "data/final.parquet"
    
    # --- EXECUTION ---
    pipeline = TwitterMentalHealthPipeline(base_dir="live_demo_data")
    
    # 1. Process the user and create the feature parquet
    user_parquet_path = pipeline.process_user(
        TARGET_USERNAME, 
        label=TRUE_LABEL, 
        max_tweets=500 # Limit for a quick demo
    )
    
    # 2. Run prediction and explanation
    if user_parquet_path:
        pipeline.predict_and_explain(
            user_parquet_path,
            binary_task=BINARY_TASK,
            full_training_data_path=FULL_DATASET_PATH
        )