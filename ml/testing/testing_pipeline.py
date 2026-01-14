"""
Mental Health Prediction Pipeline
A modular pipeline for scraping Twitter data, captioning images, and preparing features
for mental health prediction models.
"""

import os
import sys
import json
import re
import requests
import pathlib
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import zscore
from ntscraper import Nitter
import ntscraper.nitter as ntr
from bs4 import BeautifulSoup


class TwitterMentalHealthPipeline:
    """
    Pipeline for processing Twitter user data for mental health prediction.
    
    Usage:
        pipeline = TwitterMentalHealthPipeline()
        parquet_path = pipeline.process_user("username", label=0)
    """
    
    def __init__(self, 
                 base_dir: str = "pipeline_data",
                 nitter_instance: str = "http://localhost:8080",
                 min_tweets: int = 50,
                 baseline_frac: float = 0.20):
        """
        Initialize the pipeline with configuration parameters.
        
        Args:
            base_dir: Base directory for storing all pipeline data
            nitter_instance: URL of the Nitter instance to use
            min_tweets: Minimum tweets required for processing
            baseline_frac: Fraction of tweets to use for baseline calculation
        """
        self.base_dir = pathlib.Path(base_dir)
        self.nitter_instance = nitter_instance
        self.min_tweets = min_tweets
        self.baseline_frac = baseline_frac
        
        # Create directory structure
        self.data_dir = self.base_dir / "data"
        self.images_dir = self.base_dir / "images"
        self.captions_dir = self.base_dir / "captions"
        self.parquet_dir = self.base_dir / "parquets"
        
        for dir_path in [self.data_dir, self.images_dir, self.captions_dir, self.parquet_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize BLIP model (lazy loading)
        self._blip_processor = None
        self._blip_model = None
        self._device = None
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Apply Nitter monkey patch
        self._apply_nitter_patch()
    
    def _apply_nitter_patch(self):
        """Apply the avatar bug guard patch to Nitter."""
        _orig_get_user = ntr.Nitter._get_user
        def _safe_get_user(self, tweet, is_encrypted):
            try:
                return _orig_get_user(self, tweet, is_encrypted)
            except IndexError:
                uname = tweet.find("a", class_="username")
                fname = tweet.find("a", class_="fullname")
                return {
                    "id": None,
                    "username": uname.text.lstrip("@") if uname else "unknown",
                    "fullname": fname.text if fname else "unknown",
                    "avatar_url": None,
                }
        ntr.Nitter._get_user = _safe_get_user
    
    @property
    def blip_processor(self):
        """Lazy load BLIP processor."""
        if self._blip_processor is None:
            self._blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
        return self._blip_processor
    
    @property
    def blip_model(self):
        """Lazy load BLIP model."""
        if self._blip_model is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self._device)
        return self._blip_model
    
    def scrape_user(self, username: str, max_tweets: int = -1) -> Dict:
        """
        Scrape a user's timeline and download images.
        
        Args:
            username: Twitter username (without @)
            max_tweets: Maximum tweets to fetch (-1 for all)
            
        Returns:
            Dictionary with tweet data and metadata
        """
        username = username.lstrip("@")
        user_dir = self.images_dir / username
        user_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"▸ Scraping timeline for @{username}")
        
        # Initialize scraper WITHOUT instance parameter
        scraper = Nitter(
            log_level=1, 
            skip_instance_check=True
        )
        
        # Get tweets - pass instance to get_tweets method instead!
        try:
            result = scraper.get_tweets(
                username, 
                mode="user", 
                number=max_tweets,
                instance=self.nitter_instance  # Pass instance HERE
            )
            timeline = result.get("tweets", [])
        except Exception as e:
            print(f"  • Error fetching tweets: {e}")
            raise
        
        print(f"  • {len(timeline)} tweets fetched")
        
        # Save raw tweets
        json_path = self.data_dir / f"{username}_raw.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(timeline, f, ensure_ascii=False, indent=2)
        print(f"  • Timeline written to {json_path}")
        
        # Collect and download images
        img_urls = []
        tweet_images = {}  # Map tweet_id to image indices
        
        for tw in timeline:
            tweet_id = tw.get("id")
            tweet_images[tweet_id] = []
            
            for url in tw.get("photos", []):
                if url not in img_urls:
                    img_urls.append(url)
                    tweet_images[tweet_id].append(len(img_urls) - 1)
        
        print(f"  • {len(img_urls)} unique images to download")
        
        # Download images
        sess = requests.Session()
        downloaded_images = {}
        
        for idx, url in enumerate(img_urls):
            ext = pathlib.Path(url).suffix.split("?")[0] or ".jpg"
            fname = user_dir / f"{idx:04d}{ext}"
            
            try:
                with sess.get(url, timeout=30, stream=True) as r:
                    r.raise_for_status()
                    with open(fname, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                downloaded_images[idx] = str(fname)
                print(f"    ✓ {fname.name}")
            except Exception as e:
                print(f"    ✗ Failed {url} → {e}")
                downloaded_images[idx] = None
        
        return {
            "username": username,
            "tweets": timeline,
            "tweet_images": tweet_images,
            "downloaded_images": downloaded_images,
            "json_path": str(json_path)
        }
    
    def caption_images(self, scrape_data: Dict) -> Dict:
        """
        Generate captions for all images using BLIP.
        
        Args:
            scrape_data: Output from scrape_user()
            
        Returns:
            Dictionary mapping tweet_id to concatenated captions
        """
        username = scrape_data["username"]
        tweet_images = scrape_data["tweet_images"]
        downloaded_images = scrape_data["downloaded_images"]
        
        print(f"\n▸ Generating image captions for @{username}")
        
        # Caption each image
        image_captions = {}
        
        for img_idx, img_path in tqdm(downloaded_images.items(), desc="Captioning"):
            if img_path is None:
                continue
                
            caption_file = self.captions_dir / f"{username}_{img_idx:04d}.txt"
            
            # Skip if already captioned
            if caption_file.exists():
                try:
                    caption = caption_file.read_text(encoding='utf-8').strip()
                except UnicodeDecodeError:
                    caption = caption_file.read_text(encoding='utf-8', errors='ignore').strip()
                image_captions[img_idx] = caption
                continue
            
            # Generate caption
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = self.blip_processor(images=img, return_tensors="pt").to(self._device)
                out = self.blip_model.generate(**inputs, max_new_tokens=20)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                
                # Save caption
                caption_file.write_text(caption, encoding='utf-8')
                image_captions[img_idx] = caption
                
            except Exception as e:
                print(f"Failed to caption {img_path}: {e}")
                image_captions[img_idx] = ""
        
        # Map tweet_id to concatenated captions
        tweet_captions = {}
        for tweet_id, img_indices in tweet_images.items():
            captions = []
            for idx in img_indices:
                if idx in image_captions and image_captions[idx]:
                    captions.append(image_captions[idx])
            tweet_captions[tweet_id] = " ".join(captions)
        
        return tweet_captions
    
    def clean_text(self, text: str) -> str:
        """Remove URLs and mentions from text."""
        return re.sub(r"http\S+|@\w+", "", text).strip()
    
    def compute_features(self, username: str, tweets: List[Dict], 
                        tweet_captions: Dict, label: int = 0) -> pd.DataFrame:
        """
        Compute deviation features for tweets.
        
        Args:
            username: Twitter username
            tweets: List of tweet dictionaries
            tweet_captions: Dictionary mapping tweet_id to captions
            label: Label for the user (0 for control, 1 for anxiety)
            
        Returns:
            DataFrame with computed features
        """
        print(f"\n▸ Computing features for @{username}")
        
        # Create initial dataframe
        rows = []
        for tweet in tweets:
            try:
                # Parse date
                date_str = tweet["date"].replace(" UTC", "").replace(" · ", " ")
                
                # Combine text with captions
                caption = tweet_captions.get(tweet["id"], "")
                text = self.clean_text(tweet["text"])
                if caption:
                    text = f"{text} [IMG_CAP] {caption}"
                
                rows.append({
                    "tweet_id": tweet["id"],
                    "user_id": tweet["user"]["profile_id"],
                    "username": username,
                    "created": pd.to_datetime(date_str),
                    "text": text,
                    "label": label
                })
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping tweet due to error: {e}")
                continue
        
        df = pd.DataFrame(rows).sort_values("created").reset_index(drop=True)
        
        if len(df) < self.min_tweets:
            print(f"Warning: User has only {len(df)} tweets (minimum: {self.min_tweets})")
            return df
        
        # Compute baseline statistics
        n_base = max(10, int(len(df) * self.baseline_frac))
        base_df = df.head(n_base)
        
        # Calculate baseline metrics
        base_stats = {
            "len_mu": base_df["text"].str.len().mean(),
            "sent_mu": base_df["text"].apply(
                lambda t: self.sia.polarity_scores(t)["compound"]
            ).mean(),
            "n_base": n_base
        }
        
        # Calculate tweets per day
        days_span = (base_df["created"].iloc[-1] - base_df["created"].iloc[0]).days
        if days_span == 0:
            days_span = 1
        base_stats["tpd_mu"] = len(base_df) / days_span
        
        # Compute deviation features for remaining tweets
        later_df = df.iloc[n_base:].copy()
        
        if len(later_df) == 0:
            print("Warning: No tweets after baseline period")
            return df
        
        # Raw features
        later_df["len_raw"] = later_df["text"].str.len()
        later_df["sent_raw"] = later_df["text"].apply(
            lambda t: self.sia.polarity_scores(t)["compound"]
        )
        
        # Rolling tweets per day (7-day window)
        later_indexed = later_df.set_index("created")
        tpd7 = later_indexed["text"].rolling("7D").count().fillna(0)
        later_df["tpd_raw"] = tpd7.values
        
        # Compute deviations
        len_diff = later_df["len_raw"] - base_stats["len_mu"]
        sent_diff = later_df["sent_raw"] - base_stats["sent_mu"]
        tpd_diff = later_df["tpd_raw"] - base_stats["tpd_mu"]
        
        # Z-score normalization
        later_df["len_delta"] = zscore(len_diff) if len_diff.std() > 0 else 0
        later_df["sent_delta"] = zscore(sent_diff) if sent_diff.std() > 0 else 0
        later_df["tpd_delta"] = zscore(tpd_diff) if tpd_diff.std() > 0 else 0
        
        # Select final columns
        final_df = later_df[[
            "tweet_id", "user_id", "username", "text", "label",
            "len_delta", "sent_delta", "tpd_delta"
        ]]
        
        return final_df
    
    def process_user(self, username: str, label: int = 0, 
                    max_tweets: int = -1) -> str:
        """
        Complete pipeline to process a single user.
        
        Args:
            username: Twitter username (without @)
            label: Label for the user (0 for control, 1 for anxiety)
            max_tweets: Maximum tweets to fetch (-1 for all)
            
        Returns:
            Path to the generated parquet file
        """
        username = username.lstrip("@")
        print(f"\n{'='*60}")
        print(f"Processing user: @{username}")
        print(f"{'='*60}")
        
        # Step 1: Scrape user data
        scrape_data = self.scrape_user(username, max_tweets)
        
        # Step 2: Caption images
        tweet_captions = self.caption_images(scrape_data)
        
        # Step 3: Compute features
        df = self.compute_features(
            username, 
            scrape_data["tweets"], 
            tweet_captions, 
            label
        )
        
        # Step 4: Save to parquet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_path = self.parquet_dir / f"{username}_{timestamp}.parquet"
        df.to_parquet(parquet_path, index=False)
        
        print(f"\n✓ Pipeline complete!")
        print(f"  • Processed {len(df)} tweets with deviation features")
        print(f"  • Saved to: {parquet_path}")
        
        return str(parquet_path)
    
    def process_multiple_users(self, user_list: List[Tuple[str, int]], 
                             max_tweets: int = -1) -> pd.DataFrame:
        """
        Process multiple users and combine into a single dataset.
        
        Args:
            user_list: List of (username, label) tuples
            max_tweets: Maximum tweets per user
            
        Returns:
            Combined DataFrame with all users
        """
        all_dfs = []
        
        for username, label in user_list:
            try:
                parquet_path = self.process_user(username, label, max_tweets)
                df = pd.read_parquet(parquet_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error processing @{username}: {e}")
                continue
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Add train/val/test split
            combined_df = self._add_splits(combined_df)
            
            # Save combined dataset
            combined_path = self.parquet_dir / f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            combined_df.to_parquet(combined_path, index=False)
            
            print(f"\n✓ Combined dataset saved to: {combined_path}")
            return combined_df
        
        return pd.DataFrame()
    
    def _add_splits(self, df: pd.DataFrame, val_pct: float = 0.1, 
                   test_pct: float = 0.2) -> pd.DataFrame:
        """Add stratified train/val/test splits by user."""
        import random
        
        train_u, val_u, test_u = [], [], []
        
        for label in df["label"].unique():
            users = df[df["label"] == label]["user_id"].unique().tolist()
            random.shuffle(users)
            
            n = len(users)
            n_train = max(1, int((1 - val_pct - test_pct) * n))
            n_val = max(0 if n <= 2 else 1, int(val_pct * n))
            
            train_u.extend(users[:n_train])
            val_u.extend(users[n_train:n_train + n_val])
            test_u.extend(users[n_train + n_val:])
        
        # Map users to splits
        split_map = {}
        for u in train_u:
            split_map[u] = "train"
        for u in val_u:
            split_map[u] = "val"
        for u in test_u:
            split_map[u] = "test"
        
        df["split"] = df["user_id"].map(split_map)
        
        return df


# Example notebook usage function
def quick_process_user(username: str, label: int = 0, 
                      base_dir: str = "pipeline_data") -> pd.DataFrame:
    """
    Convenience function for notebook usage.
    
    Example:
        df = quick_process_user("elonmusk", label=0)
    """
    pipeline = TwitterMentalHealthPipeline(base_dir=base_dir)
    parquet_path = pipeline.process_user(username, label)
    return pd.read_parquet(parquet_path)