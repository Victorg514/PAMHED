"""
update_timelines.py
Refresh every user already present in user_timelines.json, adding tweets
that were not captured in previous runs.

Assumptions
-----------
* user_timelines.json exists in the same directory.
* Each tweet dict contains a 'link' field like '/username/status/123456789...'.
* Your local Nitter instance is reachable at http://localhost:8080.
"""

import json
import os
import re
from time import sleep
from ntscraper import Nitter
from bs4 import BeautifulSoup   # ntscraper already depends on it
import ntscraper.nitter as ntr

_orig_get_user = ntr.Nitter._get_user

def _safe_get_user(self, tweet, is_encrypted):
    """Fallback when avatar path is missing."""
    try:
        return _orig_get_user(self, tweet, is_encrypted)
    except IndexError:
        # Minimal fallback: just grab username & name from anchors
        uname = tweet.find("a", class_="username")
        fname = tweet.find("a", class_="fullname")
        return {
            "id": None,
            "username": uname.text.lstrip("@") if uname else "unknown",
            "fullname": fname.text if fname else "unknown",
            "avatar_url": None,
        }

# Monkey-patch the class method
ntr.Nitter._get_user = _safe_get_user

# ─── CONFIG ────────────────────────────────────────────────────────────────
INSTANCE_URL  = "http://localhost:8080"
TWEETS_PER_PAGE = 500               # one call returns up to this many tweets
PAUSE_BETWEEN_USERS = 1             # seconds – be nice to your instance
OUT_FILE = "data/user_timelinesB.json"
RETWEET_CAP = 100 
FILTER = 50
# ───────────────────────────────────────────────────────────────────────────

def tid(tweet_dict):
    """Return the full status-ID segment as-is (no #m stripping)."""
    return tweet_dict["link"].split("/")[-1]


def load_previous_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found – run main scraper first.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}

    seen = {user: {tid(t) for t in tweets} for user, tweets in data.items()}
    return data, seen


# ──────────────────────────  RETWEET FILTER  ──────────────────────────────
def is_retweet(tweet: dict) -> bool:
    """Return True if this tweet is detected as a retweet."""
    return (
        tweet.get("is_retweet") is True
        or tweet.get("retweeted") is True
        or tweet.get("text", "").startswith("RT @")
    )
# ───────────────────────────────────────────────────────────────────────────


def scrape_new_for_user(
    scraper: Nitter, user: str, seen_set: set[str], cap: int
) -> list[dict]:
    """Return NEW tweets, limiting retweets to <cap>."""
    timeline = scraper.get_tweets(
        user,
        "user",
        TWEETS_PER_PAGE,
        instance=INSTANCE_URL,
    ).get("tweets", [])

    new_tweets, rt_count = [], 0
    for t in timeline:
        if tid(t) in seen_set:
            continue

        if is_retweet(t):
            if rt_count >= cap:
                continue
            rt_count += 1

        new_tweets.append(t)

    return new_tweets


def main():
    scraper = Nitter(log_level=1, skip_instance_check=True)
    print(f"Using Nitter instance {INSTANCE_URL}")

    data, seen_ids = load_previous_data(OUT_FILE)
    print(f"Loaded {len(data)} users from {OUT_FILE}\n")

    for user in data.keys():
        if len(data[user]) >= FILTER:
            continue
        print(f"↳ Refreshing @{user}")

        try:
            fresh = scrape_new_for_user(scraper, user, seen_ids[user], RETWEET_CAP)
            if fresh:
                data[user].extend(fresh)
                seen_ids[user].update(tid(t) for t in fresh)
                print(
                    f"   • added {len(fresh)} tweets "
                    f"(retweets capped at {RETWEET_CAP})"
                )
            else:
                print("   • no new tweets")
        except Exception as e:
            print(f"   • failed: {e}")

        sleep(PAUSE_BETWEEN_USERS)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\nDone. File updated → {OUT_FILE}")


if __name__ == "__main__":
    main()