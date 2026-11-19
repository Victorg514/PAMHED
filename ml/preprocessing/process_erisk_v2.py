"""
Builds the eRisk (Reddit) cross-platform test set with EXACTLY the v2 Twitter feature code.

  python ml/preprocessing/process_erisk_v2.py   # data/erisk_parsed_raw.parquet -> data/erisk_v2.parquet

- Every feature comes from DataProcessor.process_user_timeline in merge_blip.py, so Reddit posts get
  the same baseline (first 40%), floors and circular timing as the tweets.
- Reddit posts have no images: the model text is "<cleaned post> [IMG_CAP] ", the same as a tweet without one.
- Features stay UNSCALED. Each model applies its own Twitter train-split scaling (config.json), so
  nothing is fit on eRisk (the v1 script fit a StandardScaler on eRisk itself).
- Same minimum timeline length as the Twitter users (MIN_TWEETS = 50).
"""
import pathlib
import sys

import pandas as pd
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from merge_blip import DataProcessor, MIN_TWEETS

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "erisk_parsed_raw.parquet"
OUT = ROOT / "data" / "erisk_v2.parquet"


def main():
    raw = pd.read_parquet(RAW)
    raw = raw.dropna(subset=["created"]).sort_values(["user_id", "created"], kind="stable").reset_index(drop=True)
    sizes = raw.groupby("user_id").size()
    raw = raw[raw["user_id"].isin(sizes[sizes >= MIN_TWEETS].index)]
    raw["tweet_id"] = raw["user_id"] + "_" + raw.groupby("user_id").cumcount().astype(str)
    users = raw.drop_duplicates("user_id")["label"].value_counts().to_dict()
    print(f"eRisk users with >= {MIN_TWEETS} posts: {users} (0 = control, 1 = depression), {len(raw)} posts")

    # Feature code only; no captions to load for Reddit
    proc = DataProcessor.__new__(DataProcessor)
    proc.sia = SentimentIntensityAnalyzer()
    proc.tweet_captions = {}

    parts = [proc.process_user_timeline(g) for _, g in tqdm(raw.groupby("user_id", sort=False), desc="eRisk users")]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    num = df.select_dtypes("number").columns
    df[num] = df[num].replace([float("inf"), float("-inf")], 0).fillna(0)
    df["split"] = "erisk"
    df.to_parquet(OUT, index=False)
    print(f"Saved {len(df)} post-baseline posts from {df['user_id'].nunique()} users to {OUT}")


if __name__ == "__main__":
    main()
